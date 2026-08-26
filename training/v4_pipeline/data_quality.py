#!/usr/bin/env python3
"""Data quality filtering for the v4 post-training pipeline.

Applies four stages of cleaning to an improved-responses JSONL dataset before
it is exported for supervised fine-tuning:

1. **Deduplication** -- exact and fuzzy (n-gram Jaccard, threshold 0.85).
   When two records are near-duplicates the higher-quality version (by review
   score) is kept.
2. **N-gram decontamination** -- removes training prompts whose 8-gram overlap
   with any evaluation test-case prompt exceeds 0.3, preventing memorisation of
   eval answers.
3. **Length filtering** -- enforces professional min/max bounds on prompt and
   response length.
4. **Quality filtering** -- drops records whose review ``scores`` fall below a
   minimum overall score (records with no scores are kept).

The module is importable::

    from training.v4_pipeline.data_quality import filter_dataset, deduplicate, decontaminate

and runnable as a CLI::

    uv run python -m training.v4_pipeline.data_quality \\
      --input sft/improved_responses.jsonl \\
      --output sft/filtered_responses.jsonl \\
      --dedup-threshold 0.85 \\
      --decontaminate \\
      --min-score 5.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Eval test cases (for decontamination) ----------------------------------

try:
    from training.v4_pipeline.eval.test_cases import ALL_TEST_CASES
except Exception:  # pragma: no cover - eval module may be unavailable in slim envs
    ALL_TEST_CASES = []


# --- Constants --------------------------------------------------------------

#: N-gram size used for decontamination (8-gram overlap with eval prompts).
NGRAM_SIZE = 8

#: N-gram size used for fuzzy dedup (smaller so short prompts still compare).
DEDUP_NGRAM_SIZE = 3

#: Default Jaccard similarity threshold above which two prompts are duplicates.
DEFAULT_DEDUP_THRESHOLD = 0.85

#: Default maximum 8-gram overlap with eval prompts before a record is dropped.
DEFAULT_DECONTAM_THRESHOLD = 0.3

#: Default minimum overall review score.
DEFAULT_MIN_SCORE = 5.0

#: Length bounds (characters).
MIN_PROMPT_LEN = 10
MAX_PROMPT_LEN = 4000
MIN_RESPONSE_LEN = 10
MAX_RESPONSE_LEN = 8000

#: Number of MinHash-style bands used to bucket candidate duplicates.
_LSH_BANDS = 20


# --- N-gram utilities -------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase and split *text* into word tokens.

    Non-alphanumeric characters are treated as separators. Returns an empty
    list for blank/whitespace-only input.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], n: int = NGRAM_SIZE) -> set[str]:
    """Return the set of *n*-grams (joined by space) from a token list.

    If the token list is shorter than *n*, a single gram containing all tokens
    is returned so that very short prompts still produce a comparable signature.
    """
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two n-gram sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _hash_gram(gram: str) -> int:
    """Stable 64-bit-ish hash of an n-gram string (stdlib only)."""
    return int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")


def _minhash_signature(grams: set[str], num_perm: int = _LSH_BANDS) -> tuple[int, ...]:
    """Compute a MinHash signature of *grams* using *num_perm* permutations.

    Each permutation is simulated by xor-ing the gram hash with a fixed salt,
    then taking the minimum. This is a lightweight, dependency-free MinHash.
    """
    if not grams:
        return tuple(0 for _ in range(num_perm))
    hashed = [_hash_gram(g) for g in grams]
    salts = [((i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) for i in range(num_perm)]
    return tuple(min(h ^ salt for h in hashed) for salt in salts)


def _signature_band(sig: tuple[int, ...], band: int, bands: int = _LSH_BANDS) -> int:
    """Return a bucket key for one LSH band of a signature."""
    rows_per_band = max(1, len(sig) // bands)
    start = band * rows_per_band
    end = start + rows_per_band
    return hash(sig[start:end])


# --- Record helpers ---------------------------------------------------------


def _get_prompt(record: dict[str, Any]) -> str:
    return str(record.get("prompt", "") or "").strip()


def _get_response(record: dict[str, Any]) -> str:
    return str(record.get("response", "") or "").strip()


def _overall_score(record: dict[str, Any]) -> float | None:
    """Extract an overall review score from a record.

    Looks for ``scores["overall"]`` first, then ``scores["score"]``, then a
    top-level ``score``. Returns ``None`` when no score is present.
    """
    scores = record.get("scores")
    if isinstance(scores, dict):
        for key in ("overall", "score", "quality", "rating"):
            val = scores.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    val = record.get("score")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _record_quality(record: dict[str, Any]) -> float:
    """Numeric quality used to pick the better of two duplicates.

    Falls back to -1.0 when no score is available so scored records always win
    over unscored ones.
    """
    score = _overall_score(record)
    return score if score is not None else -1.0


# --- Public filtering functions ---------------------------------------------


@dataclass
class FilterStats:
    """Counts describing what a filter stage removed."""

    total_input: int = 0
    total_output: int = 0
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    decontaminated_removed: int = 0
    length_filtered: int = 0
    quality_filtered: int = 0
    kept_no_score: int = 0

    def merge(self, other: FilterStats) -> FilterStats:
        """Return a new stats object summing corresponding fields."""
        return FilterStats(
            total_input=self.total_input,
            total_output=other.total_output,
            exact_duplicates_removed=self.exact_duplicates_removed + other.exact_duplicates_removed,
            fuzzy_duplicates_removed=self.fuzzy_duplicates_removed + other.fuzzy_duplicates_removed,
            decontaminated_removed=self.decontaminated_removed + other.decontaminated_removed,
            length_filtered=self.length_filtered + other.length_filtered,
            quality_filtered=self.quality_filtered + other.quality_filtered,
            kept_no_score=self.kept_no_score + other.kept_no_score,
        )

    def report(self) -> None:
        """Print a human-readable summary of the filtering run."""
        removed = self.total_input - self.total_output
        print("\n=== Data Quality Filter Summary ===")
        print(f"  Total input records:      {self.total_input}")
        print(f"  Total output records:     {self.total_output}")
        print(f"  Total removed:            {removed}")
        print(f"  Exact duplicates removed: {self.exact_duplicates_removed}")
        print(f"  Fuzzy duplicates removed: {self.fuzzy_duplicates_removed}")
        print(f"  Decontaminated removed:   {self.decontaminated_removed}")
        print(f"  Length-filtered removed:  {self.length_filtered}")
        print(f"  Quality-filtered removed: {self.quality_filtered}")
        print(f"  Kept (no score):          {self.kept_no_score}")
        print("====================================\n")


def deduplicate(
    records: list[dict[str, Any]],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Remove exact and near-duplicate records.

    Exact duplicates are removed first (keeping the highest-quality copy). Then
    fuzzy duplicates are detected via MinHash/LSH with n-gram Jaccard
    similarity >= *threshold*; the higher-quality record is retained.

    Args:
        records: Input records (each must have ``prompt``/``response``).
        threshold: Jaccard similarity at/above which two prompts are duplicates.

    Returns:
        ``(deduped_records, stats)``.
    """
    stats = FilterStats(total_input=len(records))

    # Stage 1: exact prompt dedup.
    exact_best: dict[str, dict[str, Any]] = {}
    exact_order: list[str] = []
    for rec in records:
        key = _get_prompt(rec)
        if not key:
            # Keep records without a prompt as-is (length filter handles them).
            exact_order.append("__no_prompt__" + str(id(rec)))
            exact_best[exact_order[-1]] = rec
            continue
        if key in exact_best:
            stats.exact_duplicates_removed += 1
            if _record_quality(rec) > _record_quality(exact_best[key]):
                exact_best[key] = rec
        else:
            exact_order.append(key)
            exact_best[key] = rec

    exact_records = [exact_best[k] for k in exact_order]

    # Stage 2: fuzzy dedup via MinHash/LSH.
    signatures: list[tuple[int, ...]] = []
    gram_sets: list[set[str]] = []
    for rec in exact_records:
        grams = _ngrams(_tokenize(_get_prompt(rec)), n=DEDUP_NGRAM_SIZE)
        gram_sets.append(grams)
        signatures.append(_minhash_signature(grams))

    # Bucket records by LSH band; candidates share at least one band.
    bands = _LSH_BANDS
    candidate_pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[int, list[int]] = {}
        for idx, sig in enumerate(signatures):
            bkey = _signature_band(sig, band, bands)
            buckets.setdefault(bkey, []).append(idx)
        for group in buckets.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    candidate_pairs.add((group[i], group[j]))

    removed: set[int] = set()
    for i, j in candidate_pairs:
        if i in removed or j in removed:
            continue
        sim = _jaccard(gram_sets[i], gram_sets[j])
        if sim >= threshold:
            # Keep the higher-quality record; drop the other.
            if _record_quality(exact_records[i]) >= _record_quality(exact_records[j]):
                removed.add(j)
            else:
                removed.add(i)
            stats.fuzzy_duplicates_removed += 1

    deduped = [rec for idx, rec in enumerate(exact_records) if idx not in removed]
    stats.total_output = len(deduped)
    return deduped, stats


def _eval_gram_sets() -> list[set[str]]:
    """Pre-compute 8-gram sets for every eval test-case prompt."""
    return [_ngrams(_tokenize(tc.prompt)) for tc in ALL_TEST_CASES]


def decontaminate(
    records: list[dict[str, Any]],
    threshold: float = DEFAULT_DECONTAM_THRESHOLD,
    eval_grams: list[set[str]] | None = None,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Remove records whose prompt 8-gram overlap with eval prompts is too high.

    Args:
        records: Input records.
        threshold: Maximum allowed 8-gram Jaccard overlap with any eval prompt.
        eval_grams: Pre-computed eval n-gram sets (computed from
            ``ALL_TEST_CASES`` when ``None``).

    Returns:
        ``(clean_records, stats)``.
    """
    stats = FilterStats(total_input=len(records))
    if eval_grams is None:
        eval_grams = _eval_gram_sets()

    clean: list[dict[str, Any]] = []
    for rec in records:
        prompt_grams = _ngrams(_tokenize(_get_prompt(rec)))
        contaminated = False
        for eg in eval_grams:
            if _jaccard(prompt_grams, eg) > threshold:
                contaminated = True
                break
        if contaminated:
            stats.decontaminated_removed += 1
        else:
            clean.append(rec)
    stats.total_output = len(clean)
    return clean, stats


def filter_by_length(
    records: list[dict[str, Any]],
    min_prompt: int = MIN_PROMPT_LEN,
    max_prompt: int = MAX_PROMPT_LEN,
    min_response: int = MIN_RESPONSE_LEN,
    max_response: int = MAX_RESPONSE_LEN,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Filter records by professional prompt/response length bounds.

    Args:
        records: Input records.
        min_prompt / max_prompt: Prompt length bounds (characters).
        min_response / max_response: Response length bounds (characters).

    Returns:
        ``(kept_records, stats)``.
    """
    stats = FilterStats(total_input=len(records))
    kept: list[dict[str, Any]] = []
    for rec in records:
        plen = len(_get_prompt(rec))
        rlen = len(_get_response(rec))
        if plen < min_prompt or plen > max_prompt or rlen < min_response or rlen > max_response:
            stats.length_filtered += 1
            continue
        kept.append(rec)
    stats.total_output = len(kept)
    return kept, stats


def filter_by_quality(
    records: list[dict[str, Any]],
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Filter records by minimum overall review score.

    Records that carry a ``scores`` dict (or top-level ``score``) below
    *min_score* are dropped. Records with **no** score are kept (counted in
    ``kept_no_score``).

    Args:
        records: Input records.
        min_score: Minimum overall score required to keep a scored record.

    Returns:
        ``(kept_records, stats)``.
    """
    stats = FilterStats(total_input=len(records))
    kept: list[dict[str, Any]] = []
    for rec in records:
        score = _overall_score(rec)
        if score is None:
            stats.kept_no_score += 1
            kept.append(rec)
            continue
        if score < min_score:
            stats.quality_filtered += 1
            continue
        kept.append(rec)
    stats.total_output = len(kept)
    return kept, stats


@dataclass
class FilterOptions:
    """Options controlling :func:`filter_dataset`."""

    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    decontaminate: bool = False
    decontam_threshold: float = DEFAULT_DECONTAM_THRESHOLD
    min_score: float = DEFAULT_MIN_SCORE
    min_prompt: int = MIN_PROMPT_LEN
    max_prompt: int = MAX_PROMPT_LEN
    min_response: int = MIN_RESPONSE_LEN
    max_response: int = MAX_RESPONSE_LEN
    extra: dict[str, Any] = field(default_factory=dict)


def filter_dataset(
    records: list[dict[str, Any]],
    options: FilterOptions | None = None,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Run the full data-quality pipeline over *records*.

    Stages run in order: dedup -> decontamination -> length -> quality.
    Each stage receives the output of the previous one. Aggregate stats are
    returned so callers can report what was removed at every step.

    Args:
        records: Input JSONL records (dicts with ``prompt``/``response``).
        options: Filtering configuration. Defaults to :class:`FilterOptions`.

    Returns:
        ``(filtered_records, aggregate_stats)``.
    """
    if options is None:
        options = FilterOptions()

    aggregate = FilterStats(total_input=len(records))

    # 1. Deduplication.
    records, stats = deduplicate(records, threshold=options.dedup_threshold)
    aggregate = aggregate.merge(stats)

    # 2. Decontamination (optional).
    if options.decontaminate:
        records, stats = decontaminate(records, threshold=options.decontam_threshold)
        aggregate = aggregate.merge(stats)

    # 3. Length filtering.
    records, stats = filter_by_length(
        records,
        min_prompt=options.min_prompt,
        max_prompt=options.max_prompt,
        min_response=options.min_response,
        max_response=options.max_response,
    )
    aggregate = aggregate.merge(stats)

    # 4. Quality filtering.
    records, stats = filter_by_quality(records, min_score=options.min_score)
    aggregate = aggregate.merge(stats)

    aggregate.total_output = len(records)
    return records, aggregate


# --- I/O helpers ------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (skipping blank/invalid lines)."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  warn: skipping invalid JSON on line {lineno}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write *records* to *path* as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training.v4_pipeline.data_quality",
        description="Apply data-quality filtering to an improved-responses JSONL dataset.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEFAULT_DEDUP_THRESHOLD,
        help=f"Jaccard threshold for fuzzy dedup (default: {DEFAULT_DEDUP_THRESHOLD}).",
    )
    parser.add_argument(
        "--decontaminate",
        action="store_true",
        help="Remove prompts with high n-gram overlap with eval test cases.",
    )
    parser.add_argument(
        "--decontam-threshold",
        type=float,
        default=DEFAULT_DECONTAM_THRESHOLD,
        help=f"Max 8-gram overlap with eval prompts (default: {DEFAULT_DECONTAM_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum overall review score (default: {DEFAULT_MIN_SCORE}).",
    )
    parser.add_argument(
        "--min-prompt",
        type=int,
        default=MIN_PROMPT_LEN,
        help=f"Min prompt length in chars (default: {MIN_PROMPT_LEN}).",
    )
    parser.add_argument(
        "--max-prompt",
        type=int,
        default=MAX_PROMPT_LEN,
        help=f"Max prompt length in chars (default: {MAX_PROMPT_LEN}).",
    )
    parser.add_argument(
        "--min-response",
        type=int,
        default=MIN_RESPONSE_LEN,
        help=f"Min response length in chars (default: {MIN_RESPONSE_LEN}).",
    )
    parser.add_argument(
        "--max-response",
        type=int,
        default=MAX_RESPONSE_LEN,
        help=f"Max response length in chars (default: {MAX_RESPONSE_LEN}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Reads ``--input`` JSONL, runs the full filter pipeline, writes
    ``--output`` JSONL, and prints a summary.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    records = _read_jsonl(args.input)
    print(f"Read {len(records)} records from {args.input}")

    options = FilterOptions(
        dedup_threshold=args.dedup_threshold,
        decontaminate=args.decontaminate,
        decontam_threshold=args.decontam_threshold,
        min_score=args.min_score,
        min_prompt=args.min_prompt,
        max_prompt=args.max_prompt,
        min_response=args.min_response,
        max_response=args.max_response,
    )

    filtered, stats = filter_dataset(records, options=options)
    _write_jsonl(args.output, filtered)
    print(f"Wrote {len(filtered)} records to {args.output}")
    stats.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
