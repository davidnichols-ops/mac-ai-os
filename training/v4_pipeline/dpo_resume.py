#!/usr/bin/env python3
"""DPO + Merge + Smoke test — resumes from saved SFT adapter."""
import os
import sys
import gc
import time

# === Configuration ===
BASE_MODEL = "Qwen/Qwen3.5-9B"
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 2048
DPO_EPOCHS = 1
DPO_LEARNING_RATE = 5e-5
DPO_BETA = 0.1
DPO_BATCH_SIZE = 2
DPO_GRADIENT_ACCUMULATION = 2

DATA_DIR = "/root/training/data"
RESULTS_DIR = "/root/training/results"
MODELS_DIR = "/root/training/models"
SFT_ADAPTER_DIR = f"{MODELS_DIR}/sft_adapter"
DPO_ADAPTER_DIR = f"{MODELS_DIR}/dpo_adapter"
MERGED_DIR = f"{MODELS_DIR}/merged"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === DPO Training ===
sys.path.insert(0, "/root/training")
from colab_utils import load_dpo_dataset, print_dataset_summary
from trl import DPOTrainer, DPOConfig
from peft import PeftModel
from datasets import Dataset

DPO_DATA_PATH = f"{DATA_DIR}/dpo_train.jsonl"
print(f"Loading DPO data from {DPO_DATA_PATH}...", flush=True)
dpo_examples = load_dpo_dataset(DPO_DATA_PATH)
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
dpo_model.gradient_checkpointing_enable()
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

print(f"\n=== DPO + Merge pipeline complete ===", flush=True)
print(f"DPO: {dpo_elapsed:.0f}s ({dpo_elapsed/60:.1f} min)", flush=True)
print(f"DPO adapter: {DPO_ADAPTER_DIR}", flush=True)
print(f"Merged model: {MERGED_DIR}", flush=True)
