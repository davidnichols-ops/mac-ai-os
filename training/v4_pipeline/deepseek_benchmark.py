#!/usr/bin/env python3
"""DeepSeek-style HumanEval+ benchmark.

Key differences from our first attempt:
1. Raw completion mode (no chat template) — this is what HumanEval was designed for
2. Proper stop strings: ["\ndef", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
3. Greedy decoding (temp=0)
4. Max 512 new tokens
5. Solution = prompt + completion (self-contained)
6. EvalPlus sanitization before evaluation
7. Batched generation for speed
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

MODEL_PATH = "/dev/shm/merged_model"
RESULTS_DIR = Path("/root/training/evalplus_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STOP_STRINGS = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ", "\nif __name__", "\nprint("]


def main():
    print("=== DeepSeek-style HumanEval+ Benchmark ===", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    print(f"Mode: Raw completion (no chat template)", flush=True)
    print(f"Stop strings: {STOP_STRINGS}", flush=True)
    print(f"Decoding: Greedy (temp=0)", flush=True)
    print(f"Max new tokens: 512", flush=True)
    print()

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left padding for generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Load all 164 HumanEval+ problems
    problems = get_human_eval_plus()
    problem_list = list(problems.items())
    print(f"Loaded {len(problem_list)} HumanEval+ problems", flush=True)

    # Generate completions in batches using raw prompt (no chat template)
    BATCH_SIZE = 16
    MAX_NEW_TOKENS = 512
    results = {}

    print(f"\nGenerating with batch_size={BATCH_SIZE}...", flush=True)
    t0 = time.time()

    for i in range(0, len(problem_list), BATCH_SIZE):
        batch = problem_list[i:i + BATCH_SIZE]
        prompts = []
        task_ids = []

        for task_id, problem in batch:
            # Raw prompt — just the function signature + docstring
            # This is exactly what HumanEval was designed for
            prompt = problem["prompt"]
            prompts.append(prompt)
            task_ids.append(task_id)

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate with stop strings
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                tokenizer=tokenizer,
                stop_strings=STOP_STRINGS,
            )

        # Decode completions
        for j, (task_id, out_ids) in enumerate(zip(task_ids, output_ids)):
            prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            # For left-padded inputs, the actual prompt tokens are at the end
            generated = out_ids[inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(generated, skip_special_tokens=True)

            # Apply stop strings manually too (in case some slipped through)
            for stop in STOP_STRINGS:
                if stop in completion:
                    completion = completion[:completion.index(stop)]

            # Strip trailing whitespace/newlines
            completion = completion.rstrip()

            # Solution = prompt + completion (self-contained)
            results[task_id] = {
                "task_id": task_id,
                "solution": prompts[j] + completion,
            }

        elapsed = time.time() - t0
        done = min(i + BATCH_SIZE, len(problem_list))
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(problem_list) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(problem_list)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    total_elapsed = time.time() - t0
    print(f"\nGeneration complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    # Save raw results
    raw_file = RESULTS_DIR / "humaneval_raw_completion.jsonl"
    with open(raw_file, "w") as f:
        for task_id, result in results.items():
            f.write(json.dumps(result) + "\n")
    print(f"Raw results saved to {raw_file}", flush=True)

    # Run EvalPlus sanitization
    print("\n=== Running EvalPlus sanitization ===", flush=True)
    sanitize_result = subprocess.run(
        ["python3", "-m", "evalplus.sanitize", "--samples", str(raw_file), "--dataset", "humaneval"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(sanitize_result.stdout, flush=True)
    if sanitize_result.stderr:
        print(sanitize_result.stderr[-1000:], flush=True)

    # Find sanitized file
    sanitized_file = str(raw_file).replace(".jsonl", "-sanitized.jsonl")
    if not os.path.exists(sanitized_file):
        # Try alternate naming
        sanitized_file = RESULTS_DIR / "humaneval_raw_completion-sanitized.jsonl"
    if not os.path.exists(sanitized_file):
        print(f"WARNING: Sanitized file not found at {sanitized_file}", flush=True)
        sanitized_file = str(raw_file)  # Fall back to raw
    else:
        print(f"Sanitized file: {sanitized_file}", flush=True)

    # Run EvalPlus evaluation
    print("\n=== Running EvalPlus evaluation ===", flush=True)
    eval_result = subprocess.run(
        [
            "python3", "-c",
            f"""
from evalplus.evaluate import evaluate
evaluate(
    dataset="humaneval",
    samples="{sanitized_file}",
    i_just_wanna_run=True,
    parallel=4,
)
""",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("=== EvalPlus Output ===", flush=True)
    print(eval_result.stdout, flush=True)
    if eval_result.stderr:
        print("=== Stderr ===", flush=True)
        print(eval_result.stderr[-2000:], flush=True)

    # Parse pass@1
    plus_score = None
    base_score = None
    for line in eval_result.stdout.split("\n"):
        if "pass@1" in line.lower():
            if "plus" in line.lower():
                match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
                if match:
                    plus_score = float(match.group(1))
            elif "base" in line.lower():
                match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
                if match:
                    base_score = float(match.group(1))

    # Save final results
    final = {
        "method": "deepseek_style_raw_completion",
        "plus_pass_at_1": plus_score,
        "base_pass_at_1": base_score,
        "generation_time_s": total_elapsed,
        "num_problems": len(problem_list),
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "stop_strings": STOP_STRINGS,
        "sanitized": sanitized_file != str(raw_file),
    }
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS (DeepSeek-style raw completion):")
    print(f"  HumanEval base pass@1: {base_score}%")
    print(f"  HumanEval+ pass@1: {plus_score}%")
    print(f"  Generation time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Sanitized: {final['sanitized']}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
