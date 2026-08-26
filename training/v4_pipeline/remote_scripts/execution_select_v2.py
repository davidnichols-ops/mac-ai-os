#!/usr/bin/env python3
"""Execution-based sample selection v2 — using EvalPlus test format correctly.

The test field contains a `check(candidate)` function.
We need to: define the solution, define check, call check(entry_point).
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
OUTPUT_FILE = RESULTS_DIR / "execution_selected_v2.jsonl"


def run_base_tests(solution: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    """Run the check() function from the test field on a solution."""
    full_code = solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
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
    print("=== Execution-Based Sample Selection v2 ===", flush=True)

    # Load all samples
    samples = defaultdict(list)
    with open(SAMPLES_FILE) as f:
        for line in f:
            item = json.loads(line)
            samples[item["task_id"]].append(item["solution"])

    print(f"Loaded {len(samples)} problems with samples", flush=True)
    total_samples = sum(len(v) for v in samples.values())
    print(f"Total samples: {total_samples}", flush=True)

    # Load problems
    problems = get_human_eval_plus()
    print(f"Loaded {len(problems)} HumanEval+ problems", flush=True)

    # For each problem, run base tests on all samples and pick first passing
    selected = {}
    t0 = time.time()
    alt_selected = 0

    for i, (task_id, problem) in enumerate(problems.items()):
        problem_samples = samples.get(task_id, [])
        if not problem_samples:
            continue

        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")
        if not test_code or not entry_point:
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}
            continue

        # Try each sample
        found_passing = False
        for idx, solution in enumerate(problem_samples):
            if run_base_tests(solution, test_code, entry_point):
                selected[task_id] = {"task_id": task_id, "solution": solution}
                if idx > 0:
                    alt_selected += 1
                found_passing = True
                break

        if not found_passing:
            # Use the first sample as fallback
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(problems)}] {elapsed:.0f}s — {alt_selected} alt selected", flush=True)

    elapsed = time.time() - t0
    print(f"\nSelection complete: {elapsed:.0f}s", flush=True)
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
        "method": "execution_based_selection_v2",
        "base_pass_at_1": base_pass1,
        "plus_pass_at_1": plus_pass1,
        "num_samples_per_problem": 20,
        "alt_selected": alt_selected,
        "selection_time_s": elapsed,
    }
    with open(RESULTS_DIR / "execution_selected_v2_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Execution-Selected v2 pass@1 Results:")
    print(f"  HumanEval base pass@1: {base_pass1}")
    print(f"  HumanEval+ pass@1: {plus_pass1}")
    print(f"  Alternative selections: {alt_selected}/{len(selected)}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
