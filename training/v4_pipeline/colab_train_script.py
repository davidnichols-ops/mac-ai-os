#!/usr/bin/env -S colab run --gpu T4 --keep --timeout 14400
"""QLoRA SFT + DPO training script for the adaptive operator v4 pipeline.

Runs on a Google Colab T4 GPU via `colab run`. Mounts Google Drive for
persistent storage, loads datasets, trains SFT then DPO, saves adapters
and the merged model back to Drive.

Usage:
    colab run --gpu T4 --keep --timeout 14400 colab_train_script.py
"""

import subprocess
import sys
import os

# === Step 0: Install packages ===
print("Installing packages...", flush=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "peft", "trl", "bitsandbytes",
    "datasets", "accelerate", "torch",
], check=True)

# === Step 1: Mount Google Drive ===
print("Mounting Google Drive...", flush=True)
from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR = '/content/drive/MyDrive/adaptive_operator_v4'
os.makedirs(f'{PROJECT_DIR}/data', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/models', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/results', exist_ok=True)

# Copy colab_utils.py from Drive to /content
import shutil
src_utils = f'{PROJECT_DIR}/colab_utils.py'
if os.path.exists(src_utils):
    shutil.copy(src_utils, '/content/colab_utils.py')
    print(f"Copied colab_utils.py from Drive", flush=True)
else:
    print(f"ERROR: colab_utils.py not found at {src_utils}", flush=True)
    sys.exit(1)

sys.path.insert(0, "/content")

# === Step 2: Configuration ===
BASE_MODEL = "Qwen/Qwen3.5-9B"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
SFT_EPOCHS = 3
SFT_LEARNING_RATE = 1e-4
SFT_BATCH_SIZE = 1
SFT_GRADIENT_ACCUMULATION = 8
MAX_SEQ_LENGTH = 2048
DPO_EPOCHS = 1
DPO_LEARNING_RATE = 5e-5
DPO_BETA = 0.1

# === Step 3: Load model in 4-bit NF4 ===
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {BASE_MODEL} in 4-bit NF4...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
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
model.config.pretraining_tp = 1

print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}", flush=True)

# === Step 4: SFT Training ===
from colab_utils import load_sft_dataset, get_lora_target_modules, print_dataset_summary
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()
model.config.use_cache = False

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=get_lora_target_modules(),
)

SFT_DATA_PATH = f"{PROJECT_DIR}/data/sft_train.jsonl"
print(f"Loading SFT data from {SFT_DATA_PATH}...", flush=True)
sft_examples = load_sft_dataset(SFT_DATA_PATH)
print_dataset_summary("SFT", sft_examples)
sft_ds = Dataset.from_list(sft_examples)

sft_config = SFTConfig(
    output_dir=f"{PROJECT_DIR}/results/sft",
    num_train_epochs=SFT_EPOCHS,
    per_device_train_batch_size=SFT_BATCH_SIZE,
    gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION,
    learning_rate=SFT_LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    max_length=MAX_SEQ_LENGTH,
    fp16=True,
    bf16=False,
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
sft_trainer.train()

SFT_ADAPTER_DIR = f"{PROJECT_DIR}/models/sft_adapter"
sft_trainer.model.save_pretrained(SFT_ADAPTER_DIR)
tokenizer.save_pretrained(SFT_ADAPTER_DIR)
print(f"SFT adapter saved to {SFT_ADAPTER_DIR}", flush=True)

# Free memory before DPO
del sft_trainer, model
import gc
gc.collect()
torch.cuda.empty_cache()

# === Step 5: DPO Training ===
from colab_utils import load_dpo_dataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

DPO_DATA_PATH = f"{PROJECT_DIR}/data/dpo_train.jsonl"
print(f"Loading DPO data from {DPO_DATA_PATH}...", flush=True)
dpo_examples = load_dpo_dataset(DPO_DATA_PATH)
print_dataset_summary("DPO", dpo_examples)
dpo_ds = Dataset.from_list(dpo_examples)

print("Reloading base model in 4-bit for DPO...", flush=True)
base_for_dpo = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_for_dpo.config.use_cache = False

dpo_model = PeftModel.from_pretrained(base_for_dpo, SFT_ADAPTER_DIR, is_trainable=True)
dpo_model.gradient_checkpointing_enable()
dpo_model.config.use_cache = False

dpo_tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR, trust_remote_code=True)
if dpo_tokenizer.pad_token is None:
    dpo_tokenizer.pad_token = dpo_tokenizer.eos_token

dpo_config = DPOConfig(
    output_dir=f"{PROJECT_DIR}/results/dpo",
    num_train_epochs=DPO_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION,
    learning_rate=DPO_LEARNING_RATE,
    beta=DPO_BETA,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    max_length=MAX_SEQ_LENGTH,
    max_prompt_length=MAX_SEQ_LENGTH // 2,
    fp16=True,
    bf16=False,
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
dpo_trainer.train()

DPO_ADAPTER_DIR = f"{PROJECT_DIR}/models/dpo_adapter"
dpo_trainer.model.save_pretrained(DPO_ADAPTER_DIR)
dpo_tokenizer.save_pretrained(DPO_ADAPTER_DIR)
print(f"DPO adapter saved to {DPO_ADAPTER_DIR}", flush=True)

del dpo_trainer, dpo_model, base_for_dpo
gc.collect()
torch.cuda.empty_cache()

# === Step 6: Merge LoRA adapter into base model ===
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

MERGED_DIR = f"{PROJECT_DIR}/models/merged"
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
merge_tokenizer = AutoTokenizer.from_pretrained(DPO_ADAPTER_DIR, trust_remote_code=True)
merge_tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}", flush=True)

del base_merge, merged_model
gc.collect()
torch.cuda.empty_cache()

# === Step 7: Smoke test ===
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
    print("\nSMOKE TEST PASSED: control token + tool-call format present.", flush=True)
else:
    print("\nSMOKE TEST ISSUES:", flush=True)
    for issue in result["issues"]:
        print(f"  - {issue}", flush=True)

print("\n=== Training pipeline complete ===", flush=True)
print(f"SFT adapter: {SFT_ADAPTER_DIR}", flush=True)
print(f"DPO adapter: {DPO_ADAPTER_DIR}", flush=True)
print(f"Merged model: {MERGED_DIR}", flush=True)
