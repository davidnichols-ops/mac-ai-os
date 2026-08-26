#!/usr/bin/env python3
"""Poll a Together AI fine-tuning job until it completes or fails.

Usage:
    uv run python -m training.v4_pipeline.poll_job --job-id <id> [--interval 60] [--timeout 7200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load TOGETHER_API_KEY from repo .env if present
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if line.startswith("TOGETHER_API_KEY="):
            os.environ.setdefault("TOGETHER_API_KEY", line.split("=", 1)[1].strip().strip('"'))
            break


def poll(job_id: str, interval: int = 60, timeout: int = 7200) -> int:
    import together

    client = together.Together()
    start = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nTimeout after {elapsed:.0f}s", flush=True)
            return 1

        try:
            job = client.fine_tuning.retrieve(id=job_id)
        except Exception as e:
            print(f"  [error] retrieve failed: {e}", flush=True)
            time.sleep(interval)
            continue

        status = job.status
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] status={status} "
                  f"tokens={job.token_count} steps={job.steps_completed}/{job.total_steps} "
                  f"epochs={job.epochs_completed} price=${job.total_price}", flush=True)
            last_status = status

        if status in ("completed", "failed", "cancelled"):
            print(f"\nFinal status: {status}", flush=True)
            print(f"  Output model: {getattr(job, 'model_output_name', 'N/A')}", flush=True)
            print(f"  Total price: ${job.total_price}", flush=True)
            print(f"  Token count: {job.token_count}", flush=True)

            # List checkpoints
            try:
                checkpoints = client.fine_tuning.list_checkpoints(id=job_id)
                print(f"  Checkpoints: {checkpoints}", flush=True)
            except Exception as e:
                print(f"  Checkpoints error: {e}", flush=True)

            return 0 if status == "completed" else 1

        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Together AI fine-tuning job")
    parser.add_argument("--job-id", required=True, help="Job ID to poll")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval (seconds)")
    parser.add_argument("--timeout", type=int, default=7200, help="Max wait (seconds)")
    args = parser.parse_args()
    return poll(args.job_id, args.interval, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
