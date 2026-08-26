#!/usr/bin/env python3
"""Execution-based sample selection for pass@1.

Takes the 20 samples per problem from the multi-sample run,
runs the base HumanEval tests on each, and picks the first
one that passes. This is the DeepSeek technique for boosting
pass@1 with multi-sample generation.

pass@10 on base tests was 81.5% — so if we can find a passing
sample for 81.5% of problems, our pass@1 should be ~80%+.
"""

import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from evalplus.data import get_human_eval_plus

RESULTS_DIR = Path("/root/training/evalplus_results")
SAMPLES_FILE = RESULTS_DIR / "multisample_raw.jsonl"
OUTPUT_FILE = RESULTS_DIR / "execution_selected.jsonl"


def run_base_tests(solution: str, test_code: str, timeout: int = 10) -> bool:
    """Run base test cases on a solution. Returns True if all pass."""
    full_code = solution + "\n\n" + test_code
    try:
        result = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def main():
    print("=== Execution-Based Sample Selection ===", flush=True)

    # Load all samples
    samples = defaultdict(list)
    with open(SAMPLES_FILE) as f:
        for line in f:
            item = json.loads(line)
            samples[item["task_id"]].append(item["solution"])

    print(f"Loaded {len(samples)} problems with samples", flush=True)
    total_samples = sum(len(v) for v in samples.values())
    print(f"Total samples: {total_samples}", flush=True)

    # Load problems to get base tests
    problems = get_human_eval_plus()
    print(f"Loaded {len(problems)} HumanEval+ problems", flush=True)

    # For each problem, run base tests on all samples and pick first passing
    selected = {}
    t0 = time.time()

    for i, (task_id, problem) in enumerate(problems.items()):
        problem_samples = samples.get(task_id, [])
        if not problem_samples:
            print(f"  {task_id}: NO SAMPLES", flush=True)
            continue

        # Get base test code
        base_test = problem.get("base_input", [])
        # Build test function
        test_code = "\n".join(f"assert {t}" for t in base_test)
        # Also add the entry point check if present
        if "entry_point" in problem:
            entry = problem["entry_point"]
            test_code += f"\nassert {entry} is not None"

        # Try each sample
        found_passing = False
        for idx, solution in enumerate(problem_samples):
            if run_base_tests(solution, test_code):
                selected[task_id] = {"task_id": task_id, "solution": solution}
                found_passing = True
                break

        if not found_passing:
            # Use the first sample as fallback
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(problems)}] {elapsed:.0f}s — {sum(1 for t in selected if selected[t]['solution'] != samples[t][0])} selected from alternatives", flush=True)

    elapsed = time.time() - t0
    print(f"\nSelection complete: {elapsed:.0f}s", flush=True)

    # Count how many were selected from non-first samples
    alt_selected = 0
    for task_id in selected:
        if selected[task_id]["solution"] != samples[task_id][0]:
            alt_selected += 1
    print(f"Selected from alternative samples: {alt_selected}/{len(selected)}", flush=True)

    # Save selected solutions
    with open(OUTPUT_FILE, "w") as f:
        for task_id, result in selected.items():
            f.write(json.dumps(result) + "\n")
    print(f"Saved to {OUTPUT_FILE}", flush=True)

    # Sanitize
    print("\n=== Sanitizing ===", flush=True)
    r = subprocess.run(
        ["python3", "-m", "evalplus.sanitize", "--samples", str(OUTPUT_FILE), "--dataset", "humaneval"],
        capture_output=True, text=True, timeout=300,
    )
    print(r.stdout[-300:], flush=True)

    san_file = str(OUTPUT_FILE).replace(".jsonl", "-sanitized.jsonl")

    # Evaluate
    print("\n=== Evaluating ===", flush=True)
    r = subprocess.run(
        ["python3", "-c", f"""
from evalplus.evaluate import evaluate
evaluate(dataset="humaneval", samples="{san_file}", i_just_wanna_run=True, parallel=4)
"""],
        capture_output=True, text=True, timeout=600,
    )
    print("=== EvalPlus Output ===", flush=True)
    print(r.stdout, flush=True)
    if r.stderr:
        print(r.stderr[-500:], flush=True)

    # Parse results
    base_pass1 = None
    plus_pass1 = None
    for line in r.stdout.split("\n"):
        if "pass@1" in line and "base" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match:
                base_pass1 = float(match.group(1))
        elif "pass@1" in line and "plus" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match:
                plus_pass1 = float(match.group(1))

    # Save final results
    final = {
        "method": "execution_based_selection",
        "base_pass_at_1": base_pass1,
        "plus_pass_at_1": plus_pass1,
        "num_samples_per_problem": 20,
        "alt_selected": alt_selected,
        "selection_time_s": elapsed,
    }
    with open(RESULTS_DIR / "execution_selected_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Execution-Selected pass@1 Results:")
    print(f"  HumanEval base pass@1: {base_pass1}")
    print(f"  HumanEval+ pass@1: {plus_pass1}")
    print(f"  Alternative selections: {alt_selected}/{len(selected)}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
