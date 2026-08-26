"""Training conversion pipeline — convert trajectories to training formats.

Supports three output formats:
  1. SFT (Supervised Fine-Tuning) — behavioral imitation
  2. Preference — chosen/rejected pairs for proxy wars
  3. Compute Policy — FAST/THINK/VERIFY/RECOVER classification

Usage:
  uv run python training/prepare_dataset.py --trajectories data/trajectories/ --format sft --output training/output/sft.jsonl
  uv run python training/prepare_dataset.py --trajectories data/trajectories/ --format preference --output training/output/preference.jsonl
  uv run python training/prepare_dataset.py --trajectories data/trajectories/ --format compute_policy --output training/output/policy.jsonl
  uv run python training/prepare_dataset.py --trajectories data/trajectories/ --format all --output-dir training/output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import both app and training modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.formats.compute_policy import convert_trajectories_to_compute_policy  # noqa: E402
from training.formats.preference import convert_trajectories_to_preference  # noqa: E402
from training.formats.sft import convert_trajectories_to_sft  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert MAOS trajectories to training data formats."
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--format",
        choices=["sft", "preference", "compute_policy", "all"],
        default="sft",
        help="Output format (default: sft)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (for single format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (for 'all' format)",
    )
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include thinking blocks in SFT output",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score for SFT (0.0 = all)",
    )
    parser.add_argument(
        "--winners-only",
        action="store_true",
        default=True,
        help="Only include successful trajectories in SFT (default: True)",
    )

    args = parser.parse_args()

    if not args.trajectories.is_dir():
        print(f"Error: {args.trajectories} is not a directory", file=sys.stderr)
        return 1

    if args.format == "all":
        if not args.output_dir:
            print("Error: --output-dir required for --format all", file=sys.stderr)
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)

        sft_path = args.output_dir / "sft.jsonl"
        pref_path = args.output_dir / "preference.jsonl"
        policy_path = args.output_dir / "compute_policy.jsonl"

        print(f"Converting trajectories from {args.trajectories}...")
        sft_count = convert_trajectories_to_sft(
            args.trajectories, sft_path,
            include_thinking=args.include_thinking,
            winners_only=args.winners_only,
            min_quality=args.min_quality,
        )
        print(f"  SFT: {sft_count} samples -> {sft_path}")

        pref_count = convert_trajectories_to_preference(args.trajectories, pref_path)
        print(f"  Preference: {pref_count} pairs -> {pref_path}")

        policy_count = convert_trajectories_to_compute_policy(args.trajectories, policy_path)
        print(f"  Compute Policy: {policy_count} examples -> {policy_path}")

        print(f"\nTotal: {sft_count + pref_count + policy_count} training samples")
        return 0

    if not args.output:
        print("Error: --output required for single format", file=sys.stderr)
        return 1

    if args.format == "sft":
        count = convert_trajectories_to_sft(
            args.trajectories, args.output,
            include_thinking=args.include_thinking,
            winners_only=args.winners_only,
            min_quality=args.min_quality,
        )
    elif args.format == "preference":
        count = convert_trajectories_to_preference(args.trajectories, args.output)
    elif args.format == "compute_policy":
        count = convert_trajectories_to_compute_policy(args.trajectories, args.output)
    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        return 1

    print(f"Wrote {count} samples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
