"""DPO preference pair generator for the v4 post-training pipeline.

Creates Direct Preference Optimization (DPO) preference pairs from two sources:
  a) Qwen v3.1 original responses  — the weaker / rejected responses
  b) Devin-improved responses       — the stronger / chosen responses

Preference pairs are produced in two ways:

1. **Cross-model pairs** — match qwen_responses.jsonl and improved_responses.jsonl
   by prompt_id. The improved response is "chosen", the original is "rejected".

2. **Self-preference pairs** — when qwen_responses.jsonl contains multiple
   responses for the same prompt at different temperatures, the low-temperature
   response is "chosen" and the high-temperature response is "rejected". Higher
   temperature sampling produces more erratic outputs, making this a cheap
   source of preference signal without a second model.

Output format (preference_pairs.jsonl), one JSON object per line:
  {"prompt": "...", "chosen": "...", "rejected": "...",
   "prompt_id": "...", "category": "...", "pair_type": "cross_model"|"self_preference"}

Usage:
    cd /path/to/mac-ai-os
    uv run python -m training.v4_pipeline.dpo_generator [--resume] [--limit N]

Input files (produced by the SFT generation step):
    sft/qwen_responses.jsonl     — original Qwen v3.1 responses
    sft/improved_responses.jsonl — Devin-improved responses (same prompt IDs)

Output:
    dpo/preference_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PIPELINE_DIR = Path(__file__).resolve().parent
SFT_DIR = _PIPELINE_DIR / "sft"
DPO_DIR = _PIPELINE_DIR / "dpo"

QWEN_RESPONSES_PATH = SFT_DIR / "qwen_responses.jsonl"
IMPROVED_RESPONSES_PATH = SFT_DIR / "improved_responses.jsonl"
OUTPUT_PATH = DPO_DIR / "preference_pairs.jsonl"

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

MIN_RESPONSE_LEN = 10  # characters — shorter responses are not useful for DPO
MAX_RESPONSE_LEN = 8000  # characters — cap to keep training tractable
MIN_PROMPT_LEN = 5


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts, skipping blank/malformed lines."""
    if not path.exists():
        return []
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


def load_existing_prompt_ids(path: Path) -> set[str]:
    """Load prompt_ids already present in the output file (for --resume)."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("prompt_id")
                ptype = rec.get("pair_type", "")
                # Resume key is prompt_id + pair_type so self-preference pairs
                # with the same prompt_id are not skipped.
                ids.add(f"{pid}::{ptype}")
            except json.JSONDecodeError:
                continue
    return ids


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Append records to a JSONL file. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_valid_response(text: str) -> bool:
    """Check that a response is non-empty and within length bounds."""
    if not text or not text.strip():
        return False
    if len(text.strip()) < MIN_RESPONSE_LEN:
        return False
    return True


def _is_valid_prompt(text: str) -> bool:
    """Check that a prompt is non-empty and within length bounds."""
    if not text or not text.strip():
        return False
    if len(text.strip()) < MIN_PROMPT_LEN:
        return False
    return True


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_cross_model_pairs(
    qwen_records: list[dict[str, Any]],
    improved_records: list[dict[str, Any]],
    seen: set[str],
) -> list[dict[str, Any]]:
    """Create preference pairs by matching original vs improved responses.

    Matches on prompt_id. The improved response is "chosen", the original is
    "rejected". If the same prompt_id appears multiple times in qwen_responses
    (e.g. at different temperatures), the first occurrence is used for the
    cross-model pair.
    """
    # Index improved responses by prompt_id (first occurrence wins)
    improved_by_id: dict[str, dict[str, Any]] = {}
    for rec in improved_records:
        pid = rec.get("prompt_id") or rec.get("id")
        if pid and pid not in improved_by_id:
            improved_by_id[pid] = rec

    # Index qwen responses by prompt_id (first occurrence for cross-model)
    qwen_by_id: dict[str, dict[str, Any]] = {}
    for rec in qwen_records:
        pid = rec.get("prompt_id") or rec.get("id")
        if pid and pid not in qwen_by_id:
            qwen_by_id[pid] = rec

    pairs: list[dict[str, Any]] = []
    for pid, qwen_rec in qwen_by_id.items():
        resume_key = f"{pid}::cross_model"
        if resume_key in seen:
            continue

        improved_rec = improved_by_id.get(pid)
        if not improved_rec:
            continue

        prompt = qwen_rec.get("prompt", "")
        # response field may be a dict (from qwen_inference) or a string
        raw_rejected = qwen_rec.get("response", "")
        if isinstance(raw_rejected, dict):
            rejected = raw_rejected.get("text", "")
        else:
            rejected = raw_rejected
        raw_chosen = improved_rec.get("improved_response") or improved_rec.get("response", "")
        if isinstance(raw_chosen, dict):
            chosen = raw_chosen.get("text", "")
        else:
            chosen = raw_chosen
        category = qwen_rec.get("category", improved_rec.get("category", "unknown"))

        if not _is_valid_prompt(prompt):
            continue
        if not _is_valid_response(chosen) or not _is_valid_response(rejected):
            continue
        if chosen.strip() == rejected.strip():
            # No preference signal if responses are identical
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
            "prompt_id": pid,
            "category": category,
            "pair_type": "cross_model",
        })

    return pairs


def generate_self_preference_pairs(
    qwen_records: list[dict[str, Any]],
    seen: set[str],
) -> list[dict[str, Any]]:
    """Create self-preference pairs from temperature variation.

    When qwen_responses.jsonl contains multiple responses for the same prompt
    at different temperatures, the lowest-temperature response is "chosen"
    and the highest-temperature response is "rejected". Higher temperature
    sampling tends to produce more erratic, lower-quality outputs.
    """
    # Group qwen responses by prompt_id
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in qwen_records:
        pid = rec.get("prompt_id") or rec.get("id")
        if pid:
            by_prompt[pid].append(rec)

    pairs: list[dict[str, Any]] = []
    for pid, recs in by_prompt.items():
        resume_key = f"{pid}::self_preference"
        if resume_key in seen:
            continue

        # Need at least 2 responses at different temperatures
        if len(recs) < 2:
            continue

        # Sort by temperature (ascending). Responses without a temperature
        # field are treated as temperature 0.7 (a common default).
        def _temp(rec: dict[str, Any]) -> float:
            return float(rec.get("temperature", 0.7))

        sorted_recs = sorted(recs, key=_temp)
        low_rec = sorted_recs[0]
        high_rec = sorted_recs[-1]

        low_temp = _temp(low_rec)
        high_temp = _temp(high_rec)

        # Only create a pair if there's a meaningful temperature gap
        if high_temp - low_temp < 0.2:
            continue

        prompt = low_rec.get("prompt", "")
        raw_chosen = low_rec.get("response", "")
        raw_rejected = high_rec.get("response", "")
        chosen = raw_chosen.get("text", "") if isinstance(raw_chosen, dict) else raw_chosen
        rejected = raw_rejected.get("text", "") if isinstance(raw_rejected, dict) else raw_rejected
        category = low_rec.get("category", "unknown")

        if not _is_valid_prompt(prompt):
            continue
        if not _is_valid_response(chosen) or not _is_valid_response(rejected):
            continue
        if chosen.strip() == rejected.strip():
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
            "prompt_id": pid,
            "category": category,
            "pair_type": "self_preference",
            "chosen_temp": low_temp,
            "rejected_temp": high_temp,
        })

    return pairs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(path: Path) -> None:
    """Print statistics about the preference pairs file."""
    records = load_jsonl(path)
    if not records:
        print("No preference pairs found.")
        return

    total = len(records)
    cat_dist = Counter(r.get("category", "unknown") for r in records)
    type_dist = Counter(r.get("pair_type", "unknown") for r in records)

    chosen_lens = [len(r.get("chosen", "")) for r in records]
    rejected_lens = [len(r.get("rejected", "")) for r in records]
    prompt_lens = [len(r.get("prompt", "")) for r in records]

    avg_chosen = sum(chosen_lens) / len(chosen_lens) if chosen_lens else 0
    avg_rejected = sum(rejected_lens) / len(rejected_lens) if rejected_lens else 0
    avg_prompt = sum(prompt_lens) / len(prompt_lens) if prompt_lens else 0

    print(f"\n{'=' * 60}")
    print(f"DPO Preference Pairs Statistics")
    print(f"{'=' * 60}")
    print(f"  Total pairs:      {total}")
    print(f"  Output file:      {path}")
    print(f"  File size:        {path.stat().st_size / 1024:.1f} KB")
    print()
    print(f"  Pair types:")
    for ptype, count in type_dist.most_common():
        print(f"    {ptype:20s} {count:5d}  ({count / total * 100:5.1f}%)")
    print()
    print(f"  Category distribution:")
    for cat, count in cat_dist.most_common():
        print(f"    {cat:25s} {count:5d}  ({count / total * 100:5.1f}%)")
    print()
    print(f"  Average text length (chars):")
    print(f"    prompt:   {avg_prompt:7.0f}")
    print(f"    chosen:   {avg_chosen:7.0f}")
    print(f"    rejected: {avg_rejected:7.0f}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs from Qwen v3.1 and Devin-improved responses.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompt_ids already present in the output file.",
    )
    parser.add_argument(
        "--qwen-responses",
        type=Path,
        default=QWEN_RESPONSES_PATH,
        help="Path to qwen_responses.jsonl (original Qwen v3.1 responses).",
    )
    parser.add_argument(
        "--improved-responses",
        type=Path,
        default=IMPROVED_RESPONSES_PATH,
        help="Path to improved_responses.jsonl (Devin-improved responses).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output path for preference_pairs.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of pairs to generate (0 = no limit).",
    )
    parser.add_argument(
        "--no-self-preference",
        action="store_true",
        help="Skip self-preference pair generation (temperature-based).",
    )
    args = parser.parse_args()

    # --- Load inputs ---
    print(f"Loading Qwen responses from {args.qwen_responses}...")
    qwen_records = load_jsonl(args.qwen_responses)
    print(f"  Loaded {len(qwen_records)} records")

    print(f"Loading improved responses from {args.improved_responses}...")
    improved_records = load_jsonl(args.improved_responses)
    print(f"  Loaded {len(improved_records)} records")

    if not qwen_records:
        print("ERROR: No Qwen responses found. Run the SFT generation step first.", file=sys.stderr)
        return 1

    # --- Resume support ---
    seen: set[str] = set()
    if args.resume and args.output.exists():
        seen = load_existing_prompt_ids(args.output)
        print(f"Resume: {len(seen)} pairs already in output file")

    # --- Generate pairs ---
    all_pairs: list[dict[str, Any]] = []

    # 1. Cross-model pairs (Qwen original vs Devin improved)
    print("\nGenerating cross-model preference pairs...")
    cross_pairs = generate_cross_model_pairs(qwen_records, improved_records, seen)
    print(f"  Generated {len(cross_pairs)} cross-model pairs")
    all_pairs.extend(cross_pairs)

    # 2. Self-preference pairs (temperature variation within Qwen responses)
    if not args.no_self_preference:
        print("Generating self-preference pairs (temperature variation)...")
        self_pairs = generate_self_preference_pairs(qwen_records, seen)
        print(f"  Generated {len(self_pairs)} self-preference pairs")
        all_pairs.extend(self_pairs)

    # --- Apply limit ---
    if args.limit > 0 and len(all_pairs) > args.limit:
        print(f"\nLimiting to {args.limit} pairs (from {len(all_pairs)})")
        all_pairs = all_pairs[: args.limit]

    if not all_pairs:
        print("\nNo new preference pairs generated.", file=sys.stderr)
        if args.output.exists():
            print_stats(args.output)
        return 0

    # --- Write output ---
    written = append_jsonl(args.output, all_pairs)
    print(f"\nWrote {written} preference pairs to {args.output}")

    # --- Stats ---
    print_stats(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
