"""Export DPO preference pairs to Together AI fine-tuning format.

Reads preference_pairs.jsonl (internal format with prompt/chosen/rejected) and
exports to dpo_train.jsonl in the format that Together AI's DPO fine-tuning API
actually accepts.

Together AI DPO format (per https://docs.together.ai/docs/fine-tuning/data-preparation):
  {
    "input": {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ]
    },
    "preferred_output": [
      {"role": "assistant", "content": "..."}
    ],
    "non_preferred_output": [
      {"role": "assistant", "content": "..."}
    ]
  }

Each output must contain exactly one assistant message. The input.messages is a
conversational context (system + user turns). The model's chat template is
applied automatically during training.

Control tokens ([FAST], [THINK], [VERIFY], [RECOVER], [ESCALATE]) are prepended
to the assistant responses so the DPO-trained model learns to emit them.

Validation rules:
  - No empty strings in prompt, chosen, or rejected
  - chosen != rejected (after stripping whitespace)
  - Reasonable length bounds (prompt and responses within configurable limits)

Usage:
    cd /path/to/mac-ai-os
    uv run python -m training.v4_pipeline.dpo_export [--limit N] [--output PATH]

Inputs:
    dpo/preference_pairs.jsonl  — internal preference pairs

Output:
    dpo/dpo_train.jsonl         — Together AI DPO training file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PIPELINE_DIR = Path(__file__).resolve().parent
DPO_DIR = _PIPELINE_DIR / "dpo"

INPUT_PATH = DPO_DIR / "preference_pairs.jsonl"
OUTPUT_PATH = DPO_DIR / "dpo_train.jsonl"

# ---------------------------------------------------------------------------
# Canonical system prompt (matches v3 SFT training — see runs/v3_run_002)
# ---------------------------------------------------------------------------

CANONICAL_SYSTEM_PROMPT = (
    "You are an adaptive engineering operator. You inspect systems, reason through "
    "problems, take actions, verify results, recover from failures, and decide when "
    "deeper thinking is necessary. You work from evidence, not assumptions. You "
    "document failures as carefully as successes."
)

# ---------------------------------------------------------------------------
# Control tokens
# ---------------------------------------------------------------------------

CONTROL_TOKENS = ["[FAST]", "[THINK]", "[VERIFY]", "[RECOVER]", "[ESCALATE]"]


def ensure_control_token(text: str, default_token: str = "[THINK]") -> str:
    """Ensure the assistant response starts with a control token.

    If the text already starts with a control token (after stripping leading
    whitespace), it is returned as-is. Otherwise, the default token is prepended.
    """
    stripped = text.lstrip()
    for token in CONTROL_TOKENS:
        if stripped.startswith(token):
            return stripped
    return f"{default_token} {stripped}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

MIN_PROMPT_LEN = 5
MIN_RESPONSE_LEN = 10
MAX_PROMPT_LEN = 4000
MAX_RESPONSE_LEN = 8000


class ValidationError(Exception):
    """Raised when a preference pair fails validation."""


def validate_pair(pair: dict[str, Any]) -> tuple[str, str, str]:
    """Validate a preference pair and return (prompt, chosen, rejected).

    Raises ValidationError if the pair is invalid.
    """
    prompt = pair.get("prompt", "")
    chosen = pair.get("chosen", "")
    rejected = pair.get("rejected", "")

    if not prompt or not prompt.strip():
        raise ValidationError("empty prompt")
    if not chosen or not chosen.strip():
        raise ValidationError("empty chosen response")
    if not rejected or not rejected.strip():
        raise ValidationError("empty rejected response")

    if len(prompt.strip()) < MIN_PROMPT_LEN:
        raise ValidationError(f"prompt too short ({len(prompt.strip())} chars)")
    if len(chosen.strip()) < MIN_RESPONSE_LEN:
        raise ValidationError(f"chosen too short ({len(chosen.strip())} chars)")
    if len(rejected.strip()) < MIN_RESPONSE_LEN:
        raise ValidationError(f"rejected too short ({len(rejected.strip())} chars)")

    if len(prompt) > MAX_PROMPT_LEN:
        raise ValidationError(f"prompt too long ({len(prompt)} chars)")
    if len(chosen) > MAX_RESPONSE_LEN:
        raise ValidationError(f"chosen too long ({len(chosen)} chars)")
    if len(rejected) > MAX_RESPONSE_LEN:
        raise ValidationError(f"rejected too long ({len(rejected)} chars)")

    if chosen.strip() == rejected.strip():
        raise ValidationError("chosen and rejected are identical")

    return prompt.strip(), chosen.strip(), rejected.strip()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def pair_to_together_format(
    pair: dict[str, Any],
    system_prompt: str = CANONICAL_SYSTEM_PROMPT,
) -> dict[str, Any] | None:
    """Convert an internal preference pair to Together AI DPO format.

    Returns None if the pair fails validation.
    """
    try:
        prompt, chosen, rejected = validate_pair(pair)
    except ValidationError as e:
        pid = pair.get("prompt_id", "?")
        print(f"  SKIP prompt_id={pid}: {e}", file=sys.stderr)
        return None

    # Prepend control tokens to assistant responses
    chosen_text = ensure_control_token(chosen)
    rejected_text = ensure_control_token(rejected)

    return {
        "input": {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        },
        "preferred_output": [
            {"role": "assistant", "content": chosen_text},
        ],
        "non_preferred_output": [
            {"role": "assistant", "content": rejected_text},
        ],
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping blank/malformed lines."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: {path.name}:{lineno} invalid JSON — {e}", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_export_stats(
    output_path: Path,
    total_in: int,
    total_out: int,
    skipped: int,
) -> None:
    """Print statistics about the export."""
    print(f"\n{'=' * 60}")
    print(f"DPO Export Statistics")
    print(f"{'=' * 60}")
    print(f"  Input pairs:      {total_in}")
    print(f"  Exported:         {total_out}")
    print(f"  Skipped (invalid):{skipped}")
    if total_in > 0:
        print(f"  Pass rate:        {total_out / total_in * 100:.1f}%")
    print(f"  Output file:      {output_path}")
    if output_path.exists():
        print(f"  File size:        {output_path.stat().st_size / 1024:.1f} KB")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export DPO preference pairs to Together AI fine-tuning format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Input preference_pairs.jsonl path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output dpo_train.jsonl path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of pairs to export (0 = no limit). Useful for testing.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=CANONICAL_SYSTEM_PROMPT,
        help="System prompt to include in the input messages.",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Omit the system message from the input (some models don't use it).",
    )
    args = parser.parse_args()

    # --- Load input ---
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        print("Run dpo_generator.py first to create preference pairs.", file=sys.stderr)
        return 1

    print(f"Loading preference pairs from {args.input}...")
    pairs = load_jsonl(args.input)
    print(f"  Loaded {len(pairs)} pairs")

    if not pairs:
        print("ERROR: No preference pairs to export.", file=sys.stderr)
        return 1

    # --- Apply limit ---
    if args.limit > 0:
        pairs = pairs[: args.limit]
        print(f"  Limited to {len(pairs)} pairs (--limit {args.limit})")

    # --- Export ---
    system_prompt = "" if args.no_system_prompt else args.system_prompt

    print(f"\nExporting to Together AI DPO format: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            if args.no_system_prompt:
                # Build without system message
                try:
                    prompt, chosen, rejected = validate_pair(pair)
                except ValidationError as e:
                    pid = pair.get("prompt_id", "?")
                    print(f"  SKIP prompt_id={pid}: {e}", file=sys.stderr)
                    skipped += 1
                    continue
                chosen_text = ensure_control_token(chosen)
                rejected_text = ensure_control_token(rejected)
                record = {
                    "input": {
                        "messages": [
                            {"role": "user", "content": prompt},
                        ],
                    },
                    "preferred_output": [
                        {"role": "assistant", "content": chosen_text},
                    ],
                    "non_preferred_output": [
                        {"role": "assistant", "content": rejected_text},
                    ],
                }
            else:
                record = pair_to_together_format(pair, system_prompt=system_prompt)
                if record is None:
                    skipped += 1
                    continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    # --- Stats ---
    print_export_stats(args.output, len(pairs), exported, skipped)

    if exported == 0:
        print("WARNING: No pairs exported. All pairs failed validation.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
