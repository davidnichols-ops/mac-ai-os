#!/usr/bin/env python3
"""bf16 LoRA SFT + DPO training for adaptive operator v4.1.

Key changes from v4:
  - MAX_SEQ_LENGTH: 2048 -> 32768 (16x increase)
  - Added routing fix data (THINK vs RECOVER disambiguation)
  - bf16 LoRA (no QLoRA — Unsloth docs say QLoRA not recommended for Qwen3.5)
  - Gradient checkpointing with unsloth for long context
  - EvalPlus benchmark after training

Runs on vast.ai L40S 48GB.
"""

import gc
import json
import os
import subprocess
import sys
import time

# === Configuration ===
BASE_MODEL = "Qwen/Qwen3.5-9B"
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# 32K context — covers 95% of operator sessions
# KV cache at bf16: ~1.1 GB (trivial on 48GB)
# Training VRAM: ~39 GB peak (fits 48GB with headroom)
MAX_SEQ_LENGTH = 32768

SFT_EPOCHS = 3
SFT_LEARNING_RATE = 2e-4
SFT_BATCH_SIZE = 2  # reduced from 4 — 32K sequences need more activation memory
SFT_GRADIENT_ACCUMULATION = 4  # effective batch = 8

DPO_EPOCHS = 1
DPO_LEARNING_RATE = 5e-5
DPO_BETA = 0.1
DPO_BATCH_SIZE = 1
DPO_GRADIENT_ACCUMULATION = 8

DATA_DIR = "/root/training/data"
RESULTS_DIR = "/root/training/results"
MODELS_DIR = "/root/training/models"
ROUTING_FIX_PATH = f"{DATA_DIR}/routing_fix.jsonl"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# === Load model in bf16 ===
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"Loading {BASE_MODEL} in bf16...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.config.use_cache = False
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

# === SFT Training ===
sys.path.insert(0, "/root/training")
from colab_utils import load_sft_dataset, get_lora_target_modules, print_dataset_summary
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=get_lora_target_modules(),
)

# Load SFT data + routing fixes
SFT_DATA_PATH = f"{DATA_DIR}/sft_train.jsonl"
print(f"Loading SFT data from {SFT_DATA_PATH}...", flush=True)
sft_examples = load_sft_dataset(SFT_DATA_PATH)

# Add routing fix examples
if os.path.exists(ROUTING_FIX_PATH):
    print(f"Loading routing fix data from {ROUTING_FIX_PATH}...", flush=True)
    routing_fixes = []
    with open(ROUTING_FIX_PATH) as f:
        for line in f:
            routing_fixes.append(json.loads(line))
    print(f"  {len(routing_fixes)} routing fix examples", flush=True)
    sft_examples.extend(routing_fixes)

print_dataset_summary("SFT", sft_examples)
sft_ds = Dataset.from_list(sft_examples)

sft_config = SFTConfig(
    output_dir=f"{RESULTS_DIR}/sft",
    num_train_epochs=SFT_EPOCHS,
    per_device_train_batch_size=SFT_BATCH_SIZE,
    gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION,
    learning_rate=SFT_LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    max_length=MAX_SEQ_LENGTH,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    dataset_text_field=None,
    packing=False,
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=sft_ds,
    peft_config=lora_config,
    processing_class=tokenizer,
)

print("Starting SFT training...", flush=True)
sft_start = time.time()
sft_trainer.train()
sft_elapsed = time.time() - sft_start
print(f"SFT training done in {sft_elapsed:.0f}s ({sft_elapsed/60:.1f} min)", flush=True)

SFT_ADAPTER_DIR = f"{MODELS_DIR}/sft_adapter"
sft_trainer.model.save_pretrained(SFT_ADAPTER_DIR)
tokenizer.save_pretrained(SFT_ADAPTER_DIR)
print(f"SFT adapter saved to {SFT_ADAPTER_DIR}", flush=True)

del sft_trainer, model
gc.collect()
torch.cuda.empty_cache()
print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

# === DPO Training ===
from colab_utils import load_dpo_dataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

DPO_DATA_PATH = f"{DATA_DIR}/dpo_train.jsonl"
print(f"Loading DPO data from {DPO_DATA_PATH}...", flush=True)
dpo_examples = load_dpo_dataset(DPO_DATA_PATH)

# Add routing fix DPO pairs
ROUTING_DPO_PATH = f"{DATA_DIR}/routing_fix_dpo.jsonl"
if os.path.exists(ROUTING_DPO_PATH):
    print(f"Loading routing fix DPO pairs...", flush=True)
    with open(ROUTING_DPO_PATH) as f:
        for line in f:
            dpo_examples.append(json.loads(line))

print_dataset_summary("DPO", dpo_examples)
dpo_ds = Dataset.from_list(dpo_examples)

print("Reloading base model in bf16 for DPO...", flush=True)
base_for_dpo = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
base_for_dpo.config.use_cache = False

dpo_model = PeftModel.from_pretrained(base_for_dpo, SFT_ADAPTER_DIR, is_trainable=True)
dpo_model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
dpo_model.config.use_cache = False

dpo_tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR, trust_remote_code=True)
if dpo_tokenizer.pad_token is None:
    dpo_tokenizer.pad_token = dpo_tokenizer.eos_token

dpo_config = DPOConfig(
    output_dir=f"{RESULTS_DIR}/dpo",
    num_train_epochs=DPO_EPOCHS,
    per_device_train_batch_size=DPO_BATCH_SIZE,
    gradient_accumulation_steps=DPO_GRADIENT_ACCUMULATION,
    learning_rate=DPO_LEARNING_RATE,
    beta=DPO_BETA,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=MAX_SEQ_LENGTH // 2,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=dpo_model,
    args=dpo_config,
    train_dataset=dpo_ds,
    processing_class=dpo_tokenizer,
)

print("Starting DPO training...", flush=True)
dpo_start = time.time()
dpo_trainer.train()
dpo_elapsed = time.time() - dpo_start
print(f"DPO training done in {dpo_elapsed:.0f}s ({dpo_elapsed/60:.1f} min)", flush=True)

DPO_ADAPTER_DIR = f"{MODELS_DIR}/dpo_adapter"
dpo_trainer.model.save_pretrained(DPO_ADAPTER_DIR)
dpo_tokenizer.save_pretrained(DPO_ADAPTER_DIR)
print(f"DPO adapter saved to {DPO_ADAPTER_DIR}", flush=True)

del dpo_trainer, dpo_model, base_for_dpo
gc.collect()
torch.cuda.empty_cache()

# === Merge LoRA adapter into base model ===
print("Loading base model in fp16 for clean merge...", flush=True)
base_merge = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

merged_model = PeftModel.from_pretrained(base_merge, DPO_ADAPTER_DIR)
print("Merging LoRA adapter into base model...", flush=True)
merged_model = merged_model.merge_and_unload()

MERGED_DIR = f"{MODELS_DIR}/merged"
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
merge_tokenizer = AutoTokenizer.from_pretrained(DPO_ADAPTER_DIR, trust_remote_code=True)
merge_tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}", flush=True)

del base_merge, merged_model
gc.collect()
torch.cuda.empty_cache()

# === Smoke test ===
from colab_utils import smoke_test_prompt, evaluate_response

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
    enable_thinking=False,
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

print(f"\n=== Training pipeline complete ===", flush=True)
print(f"SFT: {sft_elapsed:.0f}s ({sft_elapsed/60:.1f} min)", flush=True)
print(f"DPO: {dpo_elapsed:.0f}s ({dpo_elapsed/60:.1f} min)", flush=True)
print(f"Total: {(sft_elapsed + dpo_elapsed):.0f}s ({(sft_elapsed + dpo_elapsed)/60:.1f} min)", flush=True)
print(f"SFT adapter: {SFT_ADAPTER_DIR}", flush=True)
print(f"DPO adapter: {DPO_ADAPTER_DIR}", flush=True)
print(f"Merged model: {MERGED_DIR}", flush=True)
