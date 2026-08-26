#!/usr/bin/env python3
"""Merge DPO adapter into base model + smoke test.

Saves merged model to /dev/shm (167GB available) since overlay only has 32GB.
"""

import gc
import os
import sys
import time

BASE_MODEL = "Qwen/Qwen3.5-9B"
MODELS_DIR = "/root/training/models"
DPO_ADAPTER_DIR = f"{MODELS_DIR}/dpo_adapter"
MERGED_DIR = "/dev/shm/merged_model"  # Use shared memory — 149GB free

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, "/root/training")
from colab_utils import smoke_test_prompt, evaluate_response

print("Loading base model in fp16 for merge...", flush=True)
base_merge = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading DPO adapter from {DPO_ADAPTER_DIR}...", flush=True)
merged_model = PeftModel.from_pretrained(base_merge, DPO_ADAPTER_DIR)
print("Merging LoRA adapter into base model...", flush=True)
merged_model = merged_model.merge_and_unload()

os.makedirs(MERGED_DIR, exist_ok=True)
print(f"Saving merged model to {MERGED_DIR}...", flush=True)
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
merge_tokenizer = AutoTokenizer.from_pretrained(DPO_ADAPTER_DIR, trust_remote_code=True)
merge_tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}", flush=True)

del base_merge, merged_model
gc.collect()
torch.cuda.empty_cache()

# === Smoke test ===
print("Loading merged model for smoke test...", flush=True)
smoke_model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
smoke_tok = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)

messages = smoke_test_prompt()
input_text = smoke_tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = smoke_tok(input_text, return_tensors="pt").to(smoke_model.device)

print("Generating response...", flush=True)
with torch.no_grad():
    output_ids = smoke_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=smoke_tok.pad_token_id,
    )

generated = output_ids[0][inputs["input_ids"].shape[1]:]
response = smoke_tok.decode(generated, skip_special_tokens=True)

print("\n=== Generated response ===", flush=True)
print(response, flush=True)
print("===========================\n", flush=True)

result = evaluate_response(response, require_control_token=True, require_tool_call=True)
print("Evaluation result:", flush=True)
for k, v in result.items():
    print(f"  {k}: {v}", flush=True)

if result["passed"]:
    print("\nSMOKE TEST PASSED", flush=True)
else:
    print("\nSMOKE TEST ISSUES:", flush=True)
    for issue in result["issues"]:
        print(f"  - {issue}", flush=True)

print(f"\n=== Merge + Smoke test complete ===", flush=True)
print(f"Merged model: {MERGED_DIR}", flush=True)
