#!/usr/bin/env python3
"""Merge coding LoRA into base model and benchmark."""
import json, torch, re, time, subprocess, os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from evalplus.data import get_human_eval_plus

BASE_MODEL = "/dev/shm/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CODING_LORA = "/dev/shm/coding_lora"
MERGED_OUTPUT = "/dev/shm/coding_merged"
RESULTS_DIR = Path("/root/training/evalplus_results")
STOP = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ", "\nif __name__", "\nprint("]

def main():
    print("=== Merge Coding LoRA + Benchmark ===", flush=True)

    # Load base model
    print("Loading base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa"
    )

    # Load coding LoRA
    print("Loading coding LoRA...", flush=True)
    model = PeftModel.from_pretrained(model, CODING_LORA)
    model = model.merge_and_unload()
    print(f"Merged. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Save merged model
    print(f"Saving merged model to {MERGED_OUTPUT}...", flush=True)
    model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT)

    # Benchmark
    print("\n=== Benchmarking (raw completion, greedy) ===", flush=True)
    model.eval()
    problems = get_human_eval_plus()
    items = list(problems.items())
    BATCH = 16

    results = {}
    t0 = time.time()
    for i in range(0, len(items), BATCH):
        batch = items[i:i+BATCH]
        prompts = [p["prompt"] for _, p in batch]
        ids = [tid for tid, _ in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, temperature=1.0, top_p=1.0,
                pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer, stop_strings=STOP,
            )

        for j, (tid, o) in enumerate(zip(ids, out)):
            gen = o[inputs["input_ids"].shape[1]:]
            code = tokenizer.decode(gen, skip_special_tokens=True)
            for s in STOP:
                if s in code:
                    code = code[:code.index(s)]
            code = code.rstrip()
            results[tid] = {"task_id": tid, "solution": prompts[j] + code}

        done = min(i+BATCH, len(items))
        print(f"  [{done}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    print(f"\nGeneration: {time.time()-t0:.0f}s", flush=True)

    # Save
    out_file = str(RESULTS_DIR / "coding_merged_raw.jsonl")
    with open(out_file, "w") as f:
        for r in results.values():
            f.write(json.dumps(r) + "\n")

    # Sanitize
    print("\n=== Sanitizing ===", flush=True)
    r = subprocess.run(["python3", "-m", "evalplus.sanitize", "--samples", out_file, "--dataset", "humaneval"],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-300:], flush=True)

    san_file = out_file.replace(".jsonl", "-sanitized.jsonl")

    # Evaluate
    print("\n=== Evaluating ===", flush=True)
    r = subprocess.run(["python3", "-c", f"""
from evalplus.evaluate import evaluate
evaluate(dataset="humaneval", samples="{san_file}", i_just_wanna_run=True, parallel=4)
"""], capture_output=True, text=True, timeout=600)
    print(r.stdout, flush=True)
    if r.stderr:
        print(r.stderr[-500:], flush=True)

if __name__ == "__main__":
    main()
