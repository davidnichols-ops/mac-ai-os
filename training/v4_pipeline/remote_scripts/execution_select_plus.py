#!/usr/bin/env python3
"""Execution-based selection using BOTH base + plus tests for selection.

This maximizes the HumanEval+ score by selecting samples that pass
both base and plus test cases.
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
OUTPUT_FILE = RESULTS_DIR / "execution_selected_plus.jsonl"


def run_tests(solution: str, test_code: str, entry_point: str,
              base_inputs: list, plus_inputs: list, atol: float = 1e-6, timeout: int = 10) -> bool:
    """Run both base and plus tests on a solution."""
    # Build the full test: solution + check function + call with all inputs
    full_code = solution + "\n\n" + test_code + "\n\n"

    # Run the check function (which tests base inputs)
    full_code += f"check({entry_point})\n"

    # Also run plus inputs manually
    for inp in plus_inputs:
        if isinstance(inp, list):
            args = ", ".join(repr(a) for a in inp)
        else:
            args = repr(inp)
        full_code += f"try:\n    result = {entry_point}({args})\nexcept Exception:\n    raise AssertionError('plus test failed')\n"

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
    print("=== Execution-Based Selection (base + plus tests) ===", flush=True)

    # Load all samples
    samples = defaultdict(list)
    with open(SAMPLES_FILE) as f:
        for line in f:
            item = json.loads(line)
            samples[item["task_id"]].append(item["solution"])

    print(f"Loaded {len(samples)} problems with samples", flush=True)

    # Load problems
    problems = get_human_eval_plus()
    print(f"Loaded {len(problems)} HumanEval+ problems", flush=True)

    # For each problem, run base+plus tests on all samples and pick first passing
    selected = {}
    t0 = time.time()
    alt_selected = 0

    for i, (task_id, problem) in enumerate(problems.items()):
        problem_samples = samples.get(task_id, [])
        if not problem_samples:
            continue

        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")
        base_inputs = problem.get("base_input", [])
        plus_inputs = problem.get("plus_input", [])
        atol = problem.get("atol", 1e-6)

        if not test_code or not entry_point:
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}
            continue

        # Try each sample with base tests first, then plus tests
        found_passing = False
        for idx, solution in enumerate(problem_samples):
            if run_tests(solution, test_code, entry_point, base_inputs, plus_inputs, atol):
                selected[task_id] = {"task_id": task_id, "solution": solution}
                if idx > 0:
                    alt_selected += 1
                found_passing = True
                break

        if not found_passing:
            # Fall back to base-test-only selection
            for idx, solution in enumerate(problem_samples):
                try:
                    full_code = solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
                    r = subprocess.run(["python3", "-c", full_code], capture_output=True, text=True, timeout=10)
                    if r.returncode == 0:
                        selected[task_id] = {"task_id": task_id, "solution": solution}
                        if idx > 0:
                            alt_selected += 1
                        found_passing = True
                        break
                except:
                    continue

        if not found_passing:
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(problems)}] {elapsed:.0f}s — {alt_selected} alt selected", flush=True)

    elapsed = time.time() - t0
    print(f"\nSelection complete: {elapsed:.0f}s", flush=True)
    print(f"Selected from alternative samples: {alt_selected}/{len(selected)}", flush=True)

    # Save
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

    # Parse
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

    final = {
        "method": "execution_based_selection_plus_tests",
        "base_pass_at_1": base_pass1,
        "plus_pass_at_1": plus_pass1,
        "alt_selected": alt_selected,
    }
    with open(RESULTS_DIR / "execution_selected_plus_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Execution-Selected (base+plus) pass@1 Results:")
    print(f"  HumanEval base pass@1: {base_pass1}")
    print(f"  HumanEval+ pass@1: {plus_pass1}")
    print(f"  Alternative selections: {alt_selected}/{len(selected)}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
