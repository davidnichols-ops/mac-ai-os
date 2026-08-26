#!/usr/bin/env python3
"""Coding SFT training — LoRA on base Qwen3.5-9B.

Trains a separate coding LoRA on the BASE model (not our merged model),
then we'll merge both LoRAs (tool-calling + coding) to preserve both capabilities.

Raw completion format: function signature + docstring → body.
No chat template, no special tokens — just code completion.
"""

import json
import os
import torch
from datasets import Dataset
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = "/dev/shm/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DATASET_PATH = "/root/training/coding_sft_data/coding_sft_dataset.jsonl"
OUTPUT_DIR = "/dev/shm/coding_lora"
MAX_SEQ_LEN = 1024


def main():
    print("=== Coding SFT Training ===", flush=True)
    print(f"Base model: {MODEL_PATH}", flush=True)
    print(f"Dataset: {DATASET_PATH}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)

    # Load dataset
    samples = []
    with open(DATASET_PATH) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples", flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize — raw completion, no chat template
    def tokenize_fn(examples):
        # Concatenate prompt + completion with a newline
        # The model learns to complete function bodies
        texts = []
        for i in range(len(examples["text"])):
            texts.append(examples["text"][i])

        result = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
            return_tensors=None,
        )

        # For completion training, we want to mask the prompt tokens
        # and only compute loss on the completion
        # But for simplicity, we'll compute loss on the full sequence
        # This is standard for code completion training
        # Don't set labels — let the data collator handle it
        return result

    # Create dataset
    ds = Dataset.from_list([{"text": s["text"]} for s in samples])
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    print(f"Tokenized {len(tokenized)} samples", flush=True)

    # Load model
    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # LoRA config — same rank as our tool-calling LoRA for compatibility
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...", flush=True)
    trainer.train()

    # Save LoRA
    print(f"\nSaving LoRA to {OUTPUT_DIR}", flush=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!", flush=True)


if __name__ == "__main__":
    main()
