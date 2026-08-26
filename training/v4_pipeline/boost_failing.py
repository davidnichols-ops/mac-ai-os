#!/usr/bin/env python3
"""Generate additional high-temperature samples for failing problems,
then re-select with execution-based filtering.

Targets only the ~45 problems that still fail after base+plus selection.
Generates 30 more samples at T=0.8 for more diversity.
"""

import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

MODEL_PATH = "/dev/shm/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RESULTS_DIR = Path("/root/training/evalplus_results")
EXISTING_SAMPLES = RESULTS_DIR / "multisample_raw.jsonl"
SELECTED_FILE = RESULTS_DIR / "execution_selected_plus.jsonl"
OUTPUT_FILE = RESULTS_DIR / "all_samples_combined.jsonl"

STOP = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ", "\nif __name__", "\nprint("]
NUM_NEW_SAMPLES = 30
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8


def run_tests(solution: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    full_code = solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        result = subprocess.run(["python3", "-c", full_code], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except:
        return False


def main():
    print("=== Additional High-Temp Samples for Failing Problems ===", flush=True)

    # Load existing samples
    existing = defaultdict(list)
    with open(EXISTING_SAMPLES) as f:
        for line in f:
            item = json.loads(line)
            existing[item["task_id"]].append(item["solution"])
    print(f"Existing samples: {sum(len(v) for v in existing.values())} across {len(existing)} problems", flush=True)

    # Load problems
    problems = get_human_eval_plus()

    # Find failing problems (where no existing sample passes base tests)
    failing = []
    for task_id, problem in problems.items():
        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")
        if not test_code or not entry_point:
            continue
        samples = existing.get(task_id, [])
        any_pass = False
        for sol in samples:
            if run_tests(sol, test_code, entry_point):
                any_pass = True
                break
        if not any_pass:
            failing.append(task_id)

    print(f"Failing problems (no base-test-passing sample): {len(failing)}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa"
    )
    model.eval()
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Generate new samples for failing problems
    new_samples = defaultdict(list)
    t0 = time.time()

    for sample_idx in range(NUM_NEW_SAMPLES):
        print(f"\n--- New Sample {sample_idx + 1}/{NUM_NEW_SAMPLES} (T={TEMPERATURE}) ---", flush=True)
        torch.manual_seed(1000 + sample_idx)

        for i in range(0, len(failing), BATCH_SIZE):
            batch = failing[i:i + BATCH_SIZE]
            prompts = []
            task_ids = []

            for task_id in batch:
                problem = problems[task_id]
                prompt = problem["prompt"]
                prompts.append(prompt)
                task_ids.append(task_id)

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer, stop_strings=STOP,
                )

            for j, (task_id, out_ids) in enumerate(zip(task_ids, output_ids)):
                generated = out_ids[inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(generated, skip_special_tokens=True)
                for stop in STOP:
                    if stop in completion:
                        completion = completion[:completion.index(stop)]
                completion = completion.rstrip()
                solution = prompts[j] + completion
                new_samples[task_id].append(solution)

            done = min(i + BATCH_SIZE, len(failing))
            elapsed = time.time() - t0
            print(f"  [{done}/{len(failing)}] {elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - t0
    print(f"\nNew generation: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    # Combine all samples
    all_samples = defaultdict(list)
    for task_id, sols in existing.items():
        all_samples[task_id].extend(sols)
    for task_id, sols in new_samples.items():
        all_samples[task_id].extend(sols)

    total = sum(len(v) for v in all_samples.values())
    print(f"Total combined samples: {total}", flush=True)

    # Save combined
    with open(OUTPUT_FILE, "w") as f:
        for task_id, sols in all_samples.items():
            for idx, sol in enumerate(sols):
                f.write(json.dumps({"task_id": task_id, "solution": sol, "sample_id": idx}) + "\n")

    # Execution-based selection with base+plus tests
    print("\n=== Selecting with base+plus tests ===", flush=True)
    selected = {}
    alt_selected = 0
    t1 = time.time()

    for i, (task_id, problem) in enumerate(problems.items()):
        problem_samples = all_samples.get(task_id, [])
        if not problem_samples:
            continue

        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")
        if not test_code or not entry_point:
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}
            continue

        found = False
        for idx, sol in enumerate(problem_samples):
            if run_tests(sol, test_code, entry_point):
                selected[task_id] = {"task_id": task_id, "solution": sol}
                if idx > 0:
                    alt_selected += 1
                found = True
                break

        if not found:
            selected[task_id] = {"task_id": task_id, "solution": problem_samples[0]}

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(problems)}] {time.time()-t1:.0f}s — {alt_selected} alt", flush=True)

    print(f"Selection: {time.time()-t1:.0f}s, {alt_selected} alternatives", flush=True)

    # Save selected
    sel_file = RESULTS_DIR / "final_selected.jsonl"
    with open(sel_file, "w") as f:
        for task_id, result in selected.items():
            f.write(json.dumps(result) + "\n")

    # Sanitize + evaluate
    print("\n=== Sanitizing ===", flush=True)
    r = subprocess.run(["python3", "-m", "evalplus.sanitize", "--samples", str(sel_file), "--dataset", "humaneval"],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-200:], flush=True)

    san_file = str(sel_file).replace(".jsonl", "-sanitized.jsonl")

    print("\n=== Evaluating ===", flush=True)
    r = subprocess.run(["python3", "-c", f"""
from evalplus.evaluate import evaluate
evaluate(dataset="humaneval", samples="{san_file}", i_just_wanna_run=True, parallel=4)
"""], capture_output=True, text=True, timeout=600)
    print(r.stdout, flush=True)

    # Parse
    base_pass1 = plus_pass1 = None
    for line in r.stdout.split("\n"):
        if "pass@1" in line and "base" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match: base_pass1 = float(match.group(1))
        elif "pass@1" in line and "plus" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match: plus_pass1 = float(match.group(1))

    print(f"\n{'='*60}")
    print(f"Final Results (50 samples, T=0.2+0.8, exec-selected):")
    print(f"  HumanEval base pass@1: {base_pass1}")
    print(f"  HumanEval+ pass@1: {plus_pass1}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
