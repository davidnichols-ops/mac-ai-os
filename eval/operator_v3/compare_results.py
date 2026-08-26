#!/usr/bin/env python3
"""Compare v2 baseline vs v3 evaluation results.

Generates a side-by-side comparison report showing improvements
(or regressions) across all evaluation categories.

Usage:
    cd /path/to/mac-ai-os
    uv run python eval/operator_v3/compare_results.py \
        --v2 eval/results/v2_baseline.json \
        --v3 eval/results/v3_eval.json \
        --output eval/results/v3_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON."""
    if not path.exists():
        print(f"ERROR: Results file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def generate_comparison(v2: dict[str, Any], v3: dict[str, Any]) -> str:
    """Generate a markdown comparison report."""
    lines: list[str] = []
    lines.append("# Adaptive Operator v2 vs v3 — Evaluation Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Model info
    v2_model = v2.get("model", "unknown")
    v3_model = v3.get("model", "unknown")
    lines.append("## Models")
    lines.append("")
    lines.append("| | v2 | v3 |")
    lines.append("|---|---|---|")
    lines.append(f"| Model | {v2_model} | {v3_model} |")
    lines.append("")

    # Overall results
    v2_tasks = v2.get("tasks", [])
    v3_tasks = v3.get("tasks", [])

    v2_passed = sum(1 for t in v2_tasks if t.get("passed"))
    v3_passed = sum(1 for t in v3_tasks if t.get("passed"))

    lines.append("## Overall Results")
    lines.append("")
    lines.append("| Metric | v2 | v3 | Delta |")
    lines.append("|--------|----|----|-------|")
    lines.append(f"| Total tasks | {len(v2_tasks)} | {len(v3_tasks)} | {len(v3_tasks) - len(v2_tasks):+d} |")
    lines.append(f"| Passed | {v2_passed} | {v3_passed} | {v3_passed - v2_passed:+d} |")
    lines.append(f"| Pass rate | {v2_passed/max(len(v2_tasks),1)*100:.1f}% | {v3_passed/max(len(v3_tasks),1)*100:.1f}% | {(v3_passed/max(len(v3_tasks),1) - v2_passed/max(len(v2_tasks),1))*100:+.1f}% |")
    lines.append("")

    # Per-category breakdown
    categories = set()
    for t in v2_tasks + v3_tasks:
        if t.get("category"):
            categories.add(t["category"])

    if categories:
        lines.append("## Per-Category Results")
        lines.append("")
        lines.append("| Category | v2 Pass | v3 Pass | v2 Rate | v3 Rate | Delta |")
        lines.append("|----------|---------|---------|---------|---------|-------|")

        for cat in sorted(categories):
            v2_cat = [t for t in v2_tasks if t.get("category") == cat]
            v3_cat = [t for t in v3_tasks if t.get("category") == cat]
            v2_cat_pass = sum(1 for t in v2_cat if t.get("passed"))
            v3_cat_pass = sum(1 for t in v3_cat if t.get("passed"))
            v2_rate = v2_cat_pass / max(len(v2_cat), 1) * 100
            v3_rate = v3_cat_pass / max(len(v3_cat), 1) * 100
            delta = v3_rate - v2_rate
            lines.append(f"| {cat} | {v2_cat_pass}/{len(v2_cat)} | {v3_cat_pass}/{len(v3_cat)} | {v2_rate:.0f}% | {v3_rate:.0f}% | {delta:+.0f}% |")

        lines.append("")

    # Per-task comparison
    lines.append("## Per-Task Comparison")
    lines.append("")
    lines.append("| Task | Category | v2 | v3 | Change |")
    lines.append("|------|----------|----|----|--------|")

    v2_by_name = {t.get("name", ""): t for t in v2_tasks}
    v3_by_name = {t.get("name", ""): t for t in v3_tasks}
    all_names = sorted(set(v2_by_name) | set(v3_by_name))

    for name in all_names:
        v2_t = v2_by_name.get(name, {})
        v3_t = v3_by_name.get(name, {})
        cat = v2_t.get("category") or v3_t.get("category", "")
        v2_pass = "PASS" if v2_t.get("passed") else "FAIL"
        v3_pass = "PASS" if v3_t.get("passed") else "FAIL"
        if v2_t.get("passed") and not v3_t.get("passed"):
            change = "REGRESSION"
        elif not v2_t.get("passed") and v3_t.get("passed"):
            change = "IMPROVED"
        elif v2_t.get("passed") and v3_t.get("passed"):
            change = "both pass"
        else:
            change = "both fail"
        lines.append(f"| {name} | {cat} | {v2_pass} | {v3_pass} | {change} |")

    lines.append("")

    # Tool call comparison
    v2_tool_calls = sum(t.get("tool_calls", 0) for t in v2_tasks)
    v3_tool_calls = sum(t.get("tool_calls", 0) for t in v3_tasks)
    v2_steps = sum(t.get("steps", 0) for t in v2_tasks)
    v3_steps = sum(t.get("steps", 0) for t in v3_tasks)
    v2_duration = sum(t.get("duration_s", 0) for t in v2_tasks)
    v3_duration = sum(t.get("duration_s", 0) for t in v3_tasks)

    lines.append("## Execution Metrics")
    lines.append("")
    lines.append("| Metric | v2 | v3 | Delta |")
    lines.append("|--------|----|----|-------|")
    lines.append(f"| Total tool calls | {v2_tool_calls} | {v3_tool_calls} | {v3_tool_calls - v2_tool_calls:+d} |")
    lines.append(f"| Total steps | {v2_steps} | {v3_steps} | {v3_steps - v2_steps:+d} |")
    lines.append(f"| Total duration (s) | {v2_duration:.1f} | {v3_duration:.1f} | {v3_duration - v2_duration:+.1f} |")
    lines.append("")

    # Decision
    lines.append("## Decision")
    lines.append("")
    if v3_passed > v2_passed:
        lines.append("**v3 shows improvement.** Proceed to Phase 2 (routing removal).")
    elif v3_passed == v2_passed:
        lines.append("**v3 shows no improvement.** Investigate dataset quality and training mixture.")
    else:
        lines.append("**v3 shows regression.** Investigate training data, hyperparameters, and evaluation.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare v2 and v3 evaluation results")
    parser.add_argument("--v2", type=Path, required=True, help="v2 baseline results JSON")
    parser.add_argument("--v3", type=Path, required=True, help="v3 evaluation results JSON")
    parser.add_argument("--output", type=Path, default=Path("eval/results/v3_comparison.md"), help="Output report path")
    args = parser.parse_args()

    v2 = load_results(args.v2)
    v3 = load_results(args.v3)

    report = generate_comparison(v2, v3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Comparison report saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
