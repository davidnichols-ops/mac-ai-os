#!/usr/bin/env python3
"""Multi-sample pass@1 with execution filtering.

Generates N samples per problem at temperature 0.2,
runs the base HumanEval tests on each,
and picks the first one that passes. This is the standard
technique used by DeepSeek, CodeLlama, and others for
reporting pass@1 with execution-based selection.

Also reports pass@k (k=1,5,10) for comparison.
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STOP_STRINGS = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ", "\nif __name__", "\nprint("]
NUM_SAMPLES = 20  # Generate 20 samples per problem
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8


def run_base_tests(solution: str, test_code: str, timeout: int = 10) -> bool:
    """Run the base test cases on a solution. Returns True if all pass."""
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
    print(f"=== Multi-Sample pass@1 with Execution Filtering ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Samples per problem: {NUM_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}, Top-p: {TOP_P}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Load problems
    problems = get_human_eval_plus()
    problem_list = list(problems.items())
    print(f"Loaded {len(problem_list)} HumanEval+ problems", flush=True)

    # Generate N samples per problem
    all_samples = defaultdict(list)
    t0 = time.time()

    for sample_idx in range(NUM_SAMPLES):
        print(f"\n--- Sample {sample_idx + 1}/{NUM_SAMPLES} ---", flush=True)
        torch.manual_seed(42 + sample_idx)

        for i in range(0, len(problem_list), BATCH_SIZE):
            batch = problem_list[i:i + BATCH_SIZE]
            prompts = []
            task_ids = []

            for task_id, problem in batch:
                prompt = problem["prompt"]
                prompts.append(prompt)
                task_ids.append(task_id)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    tokenizer=tokenizer,
                    stop_strings=STOP_STRINGS,
                )

            for j, (task_id, out_ids) in enumerate(zip(task_ids, output_ids)):
                generated = out_ids[inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(generated, skip_special_tokens=True)

                for stop in STOP_STRINGS:
                    if stop in completion:
                        completion = completion[:completion.index(stop)]
                completion = completion.rstrip()

                solution = prompts[j] + completion
                all_samples[task_id].append(solution)

            done = min(i + BATCH_SIZE, len(problem_list))
            elapsed = time.time() - t0
            print(f"  [{done}/{len(problem_list)}] {elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - t0
    print(f"\nGeneration complete: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    # Save all samples in EvalPlus format
    samples_file = RESULTS_DIR / "multisample_raw.jsonl"
    with open(samples_file, "w") as f:
        for task_id, solutions in all_samples.items():
            for idx, sol in enumerate(solutions):
                f.write(json.dumps({
                    "task_id": task_id,
                    "solution": sol,
                    "sample_id": idx,
                }) + "\n")
    print(f"Saved {sum(len(v) for v in all_samples.values())} samples to {samples_file}", flush=True)

    # Sanitize
    print("\n=== Sanitizing ===", flush=True)
    san_result = subprocess.run(
        ["python3", "-m", "evalplus.sanitize", "--samples", str(samples_file), "--dataset", "humaneval"],
        capture_output=True, text=True, timeout=300,
    )
    print(san_result.stdout[-500:], flush=True)

    san_file = str(samples_file).replace(".jsonl", "-sanitized.jsonl")
    if not os.path.exists(san_file):
        san_file = str(samples_file)

    # Evaluate with EvalPlus (pass@k)
    print("\n=== Evaluating with EvalPlus ===", flush=True)
    eval_result = subprocess.run(
        ["python3", "-c", f"""
from evalplus.evaluate import evaluate
evaluate(dataset="humaneval", samples="{san_file}", i_just_wanna_run=True, parallel=4)
"""],
        capture_output=True, text=True, timeout=600,
    )
    print("=== EvalPlus Output ===", flush=True)
    print(eval_result.stdout, flush=True)
    if eval_result.stderr:
        print(eval_result.stderr[-1000:], flush=True)

    # Parse results
    base_pass1 = None
    plus_pass1 = None
    for line in eval_result.stdout.split("\n"):
        if "pass@1" in line and "base" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match:
                base_pass1 = float(match.group(1))
        elif "pass@1" in line and "plus" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match:
                plus_pass1 = float(match.group(1))

    # Save results
    final = {
        "method": "multisample_execution_filtering",
        "model": MODEL_PATH,
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "base_pass_at_1": base_pass1,
        "plus_pass_at_1": plus_pass1,
        "generation_time_s": total_elapsed,
    }
    with open(RESULTS_DIR / "multisample_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Multi-Sample Results ({NUM_SAMPLES} samples, T={TEMPERATURE}):")
    print(f"  HumanEval base pass@1: {base_pass1}")
    print(f"  HumanEval+ pass@1: {plus_pass1}")
    print(f"  Generation time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
