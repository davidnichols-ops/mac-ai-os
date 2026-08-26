#!/usr/bin/env python3
"""Fast HumanEval+ benchmark using HF backend with batch generation.

EvalPlus's HF backend generates one problem at a time. This script:
1. Loads all 164 HumanEval problems
2. Generates completions in batches of 8
3. Runs EvalPlus evaluation on the results
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/dev/shm/merged_model"
RESULTS_DIR = Path("/root/training/evalplus_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HumanEval+ dataset
from evalplus.data import get_human_eval_plus

def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    
    # Load HumanEval+ problems
    problems = get_human_eval_plus()
    problem_list = list(problems.items())
    print(f"Loaded {len(problem_list)} HumanEval+ problems", flush=True)
    
    # Generate completions
    BATCH_SIZE = 8
    MAX_NEW_TOKENS = 512
    results = {}
    
    print(f"Generating with batch_size={BATCH_SIZE}, max_new_tokens={MAX_NEW_TOKENS}", flush=True)
    t0 = time.time()
    
    for i in range(0, len(problem_list), BATCH_SIZE):
        batch = problem_list[i:i+BATCH_SIZE]
        prompts = []
        task_ids = []
        
        for task_id, problem in batch:
            # Format prompt like EvalPlus does
            prompt = problem["prompt"]
            # Add docstring signature if needed
            entry_point = problem.get("entry_point", "")
            if entry_point and entry_point not in prompt:
                pass  # HumanEval prompts already include the signature
            
            # Apply chat template
            messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{prompt}\n```"}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                text = prompt  # Fallback to raw prompt
            
            prompts.append(text)
            task_ids.append(task_id)
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        for j, (task_id, out_ids) in enumerate(zip(task_ids, output_ids)):
            generated = out_ids[inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(generated, skip_special_tokens=True)
            
            # Extract code from completion
            # Try to extract from markdown code blocks first
            code_match = re.search(r"```python\n(.*?)```", completion, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = completion
            
            # Strip thinking tokens if present
            code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
            code = code.strip()
            
            results[task_id] = {
                "task_id": task_id,
                "solution": code,
            }
        
        elapsed = time.time() - t0
        done = min(i + BATCH_SIZE, len(problem_list))
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(problem_list) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(problem_list)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s ({rate:.1f} prob/s)", flush=True)
    
    total_elapsed = time.time() - t0
    print(f"\nGeneration complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    
    # Save results in EvalPlus format
    output_file = RESULTS_DIR / "humaneval_plus_results.jsonl"
    with open(output_file, "w") as f:
        for task_id, result in results.items():
            f.write(json.dumps(result) + "\n")
    
    print(f"Results saved to {output_file}", flush=True)
    
    # Now run EvalPlus evaluation
    print("\nRunning EvalPlus evaluation...", flush=True)
    import subprocess
    eval_cmd = [
        "python3", "-m", "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", str(output_file),
        "--greedy",
    ]
    print(f"Command: {' '.join(eval_cmd)}", flush=True)
    
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)
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
        "plus_pass_at_1": plus_score,
        "base_pass_at_1": base_score,
        "generation_time_s": total_elapsed,
        "num_problems": len(problem_list),
        "batch_size": BATCH_SIZE,
    }
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(final, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"HumanEval+ pass@1: {plus_score}%")
    print(f"HumanEval base pass@1: {base_score}%")
    print(f"Generation time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
