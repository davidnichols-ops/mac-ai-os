"""SFT data generation orchestrator for the v4 post-training pipeline.

Ties prompt_generator output together with qwen_inference to produce the
initial Qwen v3.1 responses that downstream subagent review steps will refine
into supervised fine-tuning data.

Usage:
    uv run python -m training.v4_pipeline.sft_generator --input prompts/prompts_1000.jsonl
    uv run python -m training.v4_pipeline.sft_generator --input prompts/prompts_1000.jsonl --limit 50
    uv run python -m training.v4_pipeline.sft_generator --input prompts/prompts_1000.jsonl --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from training.v4_pipeline.qwen_inference import (
    QwenInferenceClient,
    load_api_key,
)

logger = logging.getLogger(__name__)

# --- Defaults ---

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PIPELINE_DIR / "prompts" / "prompts_1000.jsonl"
DEFAULT_OUTPUT = PIPELINE_DIR / "sft" / "qwen_responses.jsonl"
PROGRESS_EVERY = 100


def _iter_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue


def _load_processed_ids(output_path: Path) -> set[str]:
    """Return the set of prompt ids already present in the output file."""
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id") or rec.get("prompt_id")
            if pid:
                ids.add(pid)
    return ids


def _print_summary(stats: dict[str, Any]) -> None:
    """Print human-readable summary statistics."""
    written = stats["written"]
    errors = stats["errors"]
    total = written + stats["skipped"]
    error_rate = (errors / written * 100) if written else 0.0
    tool_call_rate = (stats["tool_calls"] / written) if written else 0.0
    avg_latency = (stats["total_latency"] / written) if written else 0.0

    print("\n" + "=" * 60)
    print("SFT Generation Summary")
    print("=" * 60)
    print(f"  Total prompts read : {stats['total_read']}")
    print(f"  Responses written  : {written}")
    print(f"  Skipped (resume)   : {stats['skipped']}")
    print(f"  Errors             : {errors} ({error_rate:.1f}%)")
    print(f"  Tool calls emitted : {stats['tool_calls']} ({tool_call_rate:.2f}/response)")
    print(f"  Avg latency        : {avg_latency:.2f}s")
    print(f"  Total tokens       : {stats['total_tokens']}")
    print(f"  Output file        : {stats['output_path']}")
    print()

    mode_counts = stats["mode_counts"]
    mode_total = sum(mode_counts.values()) or 1
    print("  Mode distribution:")
    for mode in ("FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE"):
        count = mode_counts.get(mode, 0)
        pct = count / mode_total * 100
        print(f"    {mode:10s}: {count:5d} ({pct:5.1f}%)")
    other = mode_total - sum(mode_counts.get(m, 0) for m in ("FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE"))
    if other:
        print(f"    {'UNKNOWN':10s}: {other:5d} ({other / mode_total * 100:5.1f}%)")
    print("=" * 60)


def _process_one(
    client: QwenInferenceClient,
    rec: dict[str, Any],
    *,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    """Process a single prompt record. Returns output record or None on skip."""
    pid = rec.get("id") or rec.get("prompt_id")
    prompt_text = rec.get("prompt")
    if not prompt_text:
        logger.warning("Skipping record without a prompt: %s", pid)
        return None
    result = client.generate_response(
        prompt_text,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    out_rec = dict(rec)
    out_rec["response"] = result.to_dict()
    return out_rec


def run(
    input_path: Path,
    output_path: Path,
    *,
    limit: Optional[int] = None,
    resume: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the SFT generation orchestrator.

    For each prompt in the input JSONL:
      1. Call qwen_inference.generate_response() to get Qwen v3.1's response.
      2. Write the response (plus original prompt metadata) to the output JSONL.

    Resume support skips prompts whose ids are already in the output file.
    Progress is logged every PROGRESS_EVERY items. Summary stats are returned
    and printed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = _load_processed_ids(output_path) if resume else set()
    if processed:
        logger.info("Resume: %d prompts already processed in %s", len(processed), output_path)

    client = QwenInferenceClient()  # temperature is set per-call in generate_response()

    # Collect pending prompts
    pending: list[dict[str, Any]] = []
    total = 0
    for rec in _iter_jsonl(input_path):
        total += 1
        pid = rec.get("id") or rec.get("prompt_id")
        if pid and pid in processed:
            continue
        if limit is not None and len(pending) >= limit:
            break
        pending.append(rec)

    skipped = total - len(pending)
    written = 0
    errors = 0
    mode_counts: dict[str, int] = {}
    tool_call_count = 0
    total_latency = 0.0
    total_tokens = 0
    start_time = time.monotonic()
    write_lock = threading.Lock()

    print(f"  Workers: {workers}, Pending: {len(pending)}, Skipped (resume): {skipped}", flush=True)

    def _handle_result(out_rec: dict[str, Any] | None) -> None:
        nonlocal written, errors, tool_call_count, total_latency, total_tokens
        if out_rec is None:
            return
        result_dict = out_rec.get("response", {})
        with write_lock:
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            if result_dict.get("error"):
                errors += 1
            mode = result_dict.get("mode")
            if mode:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            tool_call_count += len(result_dict.get("tool_calls", []))
            total_latency += result_dict.get("latency", 0)
            total_tokens += result_dict.get("total_tokens", 0)
            if written % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - start_time
                rate = written / elapsed if elapsed else 0.0
                logger.info(
                    "Progress: written=%d skipped=%d errors=%d rate=%.1f/s tokens=%d",
                    written,
                    skipped,
                    errors,
                    rate,
                    total_tokens,
                )
                print(
                    f"  [{written}] written={written} skipped={skipped} "
                    f"errors={errors} rate={rate:.1f}/s tokens={total_tokens}",
                    flush=True,
                )

    with output_path.open("a", encoding="utf-8") as out_f:
        if workers <= 1:
            for rec in pending:
                out_rec = _process_one(client, rec, temperature=temperature, max_tokens=max_tokens)
                _handle_result(out_rec)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process_one, client, rec, temperature=temperature, max_tokens=max_tokens): rec
                    for rec in pending
                }
                for fut in as_completed(futures):
                    try:
                        out_rec = fut.result()
                    except Exception as exc:
                        logger.error("Worker error: %s", exc)
                        out_rec = None
                    _handle_result(out_rec)

    elapsed = time.monotonic() - start_time
    stats = {
        "total_read": total,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "mode_counts": mode_counts,
        "tool_calls": tool_call_count,
        "total_latency": round(total_latency, 3),
        "total_tokens": total_tokens,
        "elapsed": round(elapsed, 3),
        "output_path": str(output_path),
    }
    _print_summary(stats)
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SFT data by calling Qwen v3.1 on prompt_generator output.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Input JSONL of prompts (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL of responses (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only this many prompts (for testing).",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Skip prompts already present in the output file (default: true).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume; reprocess all prompts (appends to output).",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max completion tokens (default: 2048).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel API workers (default: 1). The H100 endpoint batches concurrent requests efficiently.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    # Validate API key early so we fail fast.
    try:
        load_api_key()
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: cannot load API key: {exc}", file=sys.stderr)
        return 1

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Limit : {args.limit if args.limit else 'all'}")
    print(f"Resume: {args.resume}")
    print()

    run(
        input_path,
        output_path,
        limit=args.limit,
        resume=args.resume,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
