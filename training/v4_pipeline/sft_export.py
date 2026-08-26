#!/usr/bin/env python3
"""Convert improved-responses JSONL into Together AI SFT format.

Reads an "improved responses" JSONL file (one record per line) and emits the
OpenAI-compatible chat format that Together AI expects for supervised
fine-tuning:

    {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": "<prompt>"},
        {"role": "assistant", "content": "[<MODE>] <response>"},
    ]}

Input record schema (flexible -- missing fields fall back to safe defaults):

    {
        "prompt":          "<user task>",            # required
        "response":        "<improved assistant text>", # required
        "mode":            "FAST|THINK|VERIFY|RECOVER|ESCALATE",  # optional
        "control_token":   "[FAST]",                 # optional (overrides mode)
        "system":          "<override system prompt>" # optional
    }

The system prompt teaches control tokens and tool use, matching the
qwen_inference.py / api_provider.py runtime contract so the fine-tuned model
emits the right control token at the start of every generation and calls tools
in the structured tool-call format the runtime parses.

Usage:
    cd /path/to/mac-ai-os
    uv run python -m training.v4_pipeline.sft_export [--limit N]

    # or directly:
    uv run python training/v4_pipeline/sft_export.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# --- Paths ------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
INPUT_PATH = PIPELINE_DIR / "sft" / "improved_responses.jsonl"
OUTPUT_PATH = PIPELINE_DIR / "sft" / "sft_train.jsonl"

# --- Defaults ---------------------------------------------------------------

DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_MIN_RESPONSE_LENGTH = 10  # chars
DEFAULT_MIN_SCORE = 0.0  # 0.0 = no quality filtering


# --- System prompt ----------------------------------------------------------

# Build the tool-call format tokens with chr() so the source contains no
# literal angle-bracket tags (mirrors app/models/api_provider.py).
_TC_OPEN = chr(60) + "tool_call" + chr(62)
_TC_CLOSE = chr(60) + "/tool_call" + chr(62)

# Canonical operator system prompt (matches the v3 production dataset and the
# qwen_inference.py / api_provider.py runtime). Teaches control tokens and the
# structured tool-call format so the model learns to self-select compute mode
# and emit parseable tool calls.
SYSTEM_PROMPT = (
    "You are an adaptive engineering operator. You inspect systems, reason "
    "through problems, take actions, verify results, recover from failures, "
    "and decide when deeper thinking is necessary. You work from evidence, "
    "not assumptions. You document failures as carefully as successes.\n\n"
    "## Compute Modes (control tokens)\n"
    "Begin every response with exactly one control token to signal how much "
    "reasoning the task needs:\n"
    "  [FAST]     -- direct, low-latency action. Use for simple tool calls, "
    "file reads, status checks, and routine edits.\n"
    "  [THINK]    -- extended reasoning before acting. Use for design, "
    "debugging, refactoring, multi-step planning, and ambiguous problems.\n"
    "  [VERIFY]   -- act, then verify the result before declaring success. "
    "Use when a change must be confirmed (tests pass, file written, deploy "
    "succeeded).\n"
    "  [RECOVER]  -- a previous attempt failed; reassess and try a different "
    "strategy. Use after errors, broken builds, or unexpected output.\n"
    "  [ESCALATE] -- the task exceeds safe autonomous scope; stop and surface "
    "the situation for a human. Use for irreversible or high-risk actions.\n\n"
    "## Tool Use\n"
    "When a task requires a tool, emit a structured tool call in this exact "
    "format:\n"
    f"  {_TC_OPEN}\n"
    '  {"name": "tool_name", "arguments": {"param": "value"}}\n'
    f"  {_TC_CLOSE}\n\n"
    "After each tool call you will receive the result. Use it to decide your "
    "next action. When the task is complete, give a concise final answer "
    "with no tool calls."
)

# Valid control tokens (bracketed form, emitted at the start of the assistant
# turn). Order matters only for the validation regex.
CONTROL_TOKENS = ("FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE")
_TOKEN_RE = re.compile(r"^\s*\[(FAST|THINK|VERIFY|RECOVER|ESCALATE)\]")


# --- Helpers ----------------------------------------------------------------


def _ensure_token(content: str, mode: str | None, token: str | None) -> str:
    """Ensure assistant content starts with a control token.

    If the content already begins with a token, return it unchanged (after
    stripping leading whitespace). Otherwise prepend the appropriate token,
    preferring an explicit ``token`` then ``mode`` then a default of THINK.
    """
    stripped = content.lstrip()
    if _TOKEN_RE.match(stripped):
        return stripped

    if token:
        label = token.strip().strip("[]").upper()
        if label in CONTROL_TOKENS:
            return f"[{label}] {stripped}"

    if mode:
        label = str(mode).strip().upper().strip("[]")
        if label in CONTROL_TOKENS:
            return f"[{label}] {stripped}"

    # Safe default -- THINK is the conservative mode.
    return f"[THINK] {stripped}"


def _coerce_str(value: Any) -> str:
    """Coerce a value to a non-empty string, returning '' for None/missing."""
    if value is None:
        return ""
    return str(value).strip()


# --- Token estimation -------------------------------------------------------

# Lazily-loaded tiktoken encoder (optional; falls back to char-based estimate).
_encoder: Any = None
_encoder_checked = False


def _get_encoder() -> Any:
    """Return a tiktoken encoder if available, else None.

    tiktoken gives an accurate token count for the Qwen/BPE family. When it
    is not installed we fall back to a rough ``len(text) // 4`` heuristic.
    """
    global _encoder, _encoder_checked
    if not _encoder_checked:
        _encoder_checked = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = None
    return _encoder


def estimate_token_count(text: str) -> int:
    """Estimate the token count for a string.

    Uses tiktoken (cl100k_base) when available for an accurate BPE count.
    Falls back to ``len(text) // 4`` as a rough heuristic (~4 chars/token).
    """
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_example_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the total token count for a chat-formatted example.

    Sums the token counts of each message's content plus a small per-message
    overhead (4 tokens, approximating chat-template role markers).
    """
    total = 0
    for msg in messages:
        total += estimate_token_count(msg.get("content", ""))
        total += 4  # role/template overhead per message
    return total


def compute_overall_score(scores: Any) -> float | None:
    """Compute an overall review score as the mean of individual scores.

    ``scores`` is expected to be a dict of ``{criterion: score}`` where each
    score is numeric. Returns ``None`` if no numeric scores are present.
    """
    if not isinstance(scores, dict) or not scores:
        return None
    values: list[float] = []
    for v in scores.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            values.append(float(v))
    if not values:
        return None
    return sum(values) / len(values)


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single improved-response record to SFT chat format.

    Returns ``{"messages": [...]}`` or ``None`` if the record is unusable
    (missing prompt or response).
    """
    prompt = _coerce_str(record.get("prompt"))
    response = _coerce_str(record.get("response"))
    if not prompt or not response:
        return None

    mode = record.get("mode")
    token = record.get("control_token") or record.get("token")
    system = _coerce_str(record.get("system")) or SYSTEM_PROMPT

    assistant_content = _ensure_token(response, mode=mode, token=token)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def export_sft(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    limit: int | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    min_response_length: int = DEFAULT_MIN_RESPONSE_LENGTH,
    min_score: float = DEFAULT_MIN_SCORE,
) -> int:
    """Convert an improved-responses JSONL file to Together AI SFT format.

    Args:
        input_path: Source JSONL with improved-response records.
        output_path: Destination JSONL in {"messages": [...]} format.
        limit: If set, only convert the first N usable records.
        max_seq_length: Skip examples whose estimated token count exceeds this.
        min_response_length: Skip examples whose assistant response (chars) is
            shorter than this. Filters trivially short responses.
        min_score: Skip records whose mean review score is below this. 0.0
            disables quality filtering. Scores are read from the ``scores``
            dict in each input record.

    Returns:
        The number of examples written.
    """
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    skipped_length = 0
    skipped_response_len = 0
    skipped_score = 0
    seen_prompts: set[str] = set()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  skip: invalid JSON ({e})", file=sys.stderr)
                skipped += 1
                skipped_invalid += 1
                continue

            # Quality filtering: skip records below the score threshold.
            if min_score > 0.0:
                overall = compute_overall_score(record.get("scores"))
                if overall is None or overall < min_score:
                    skipped += 1
                    skipped_score += 1
                    continue

            example = convert_record(record)
            if example is None:
                skipped += 1
                skipped_invalid += 1
                continue

            # Min response length: skip trivially short assistant responses.
            assistant_content = example["messages"][2]["content"]
            if len(assistant_content) < min_response_length:
                skipped += 1
                skipped_response_len += 1
                continue

            # Sequence length validation: skip examples that exceed max_seq_length.
            token_count = estimate_example_tokens(example["messages"])
            if token_count > max_seq_length:
                skipped += 1
                skipped_length += 1
                continue

            # Deduplicate by prompt text to avoid overfitting.
            prompt_key = example["messages"][1]["content"]
            if prompt_key in seen_prompts:
                skipped += 1
                skipped_duplicate += 1
                continue
            seen_prompts.add(prompt_key)

            fout.write(json.dumps(example) + "\n")
            written += 1

            if limit is not None and written >= limit:
                break

    print(f"Exported {written} SFT examples -> {output_path}")
    if skipped:
        print(f"Skipped {skipped} records total:")
        if skipped_invalid:
            print(f"  invalid/empty:    {skipped_invalid}")
        if skipped_duplicate:
            print(f"  duplicate:        {skipped_duplicate}")
        if skipped_length:
            print(f"  too long (>{max_seq_length} tokens): {skipped_length}")
        if skipped_response_len:
            print(f"  response too short (<{min_response_length} chars): {skipped_response_len}")
        if skipped_score:
            print(f"  below min_score ({min_score}): {skipped_score}")

    # Report control-token distribution for a quick sanity check.
    if written:
        dist: dict[str, int] = {}
        with open(output_path) as f:
            for ln in f:
                ex = json.loads(ln)
                m = _TOKEN_RE.match(ex["messages"][2]["content"].lstrip())
                label = m.group(1) if m else "NONE"
                dist[label] = dist.get(label, 0) + 1
        print("\nControl-token distribution:")
        for label in (*CONTROL_TOKENS, "NONE"):
            if label in dist:
                pct = dist[label] / written * 100
                print(f"  [{label}]: {dist[label]} ({pct:.1f}%)")

    return written


# --- CLI --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert improved responses to Together AI SFT format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help=f"Input JSONL (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output JSONL (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only export the first N usable records (for testing).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Skip examples whose estimated token count exceeds this "
             f"(default: {DEFAULT_MAX_SEQ_LENGTH}). Uses tiktoken when "
             f"available, else len(text)//4 heuristic.",
    )
    parser.add_argument(
        "--min-response-length",
        type=int,
        default=DEFAULT_MIN_RESPONSE_LENGTH,
        help=f"Skip examples whose assistant response is shorter than this "
             f"many characters (default: {DEFAULT_MIN_RESPONSE_LENGTH}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Skip records whose mean review score (from the 'scores' dict) "
             f"is below this threshold. 0.0 disables quality filtering "
             f"(default: {DEFAULT_MIN_SCORE}).",
    )
    args = parser.parse_args()

    count = export_sft(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_seq_length=args.max_seq_length,
        min_response_length=args.min_response_length,
        min_score=args.min_score,
    )
    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
