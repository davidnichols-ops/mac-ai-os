#!/usr/bin/env python3
"""Convert Qwen3.5-9B to MLX 4-bit — disk-efficient version.

Downloads model, loads weights, deletes HF cache, then quantizes and saves.
"""
import mlx.core as mx
import mlx.nn as nn
import json
import os
import shutil
from pathlib import Path
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from huggingface_hub import snapshot_download, hf_hub_download

mx.set_default_device(mx.Device(mx.cpu))

HF_REPO = os.environ.get("HF_REPO", "davidnichols-ops/adaptive-operator-v4")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN environment variable is required to download the model.")
OUT = str(Path.home() / "mac-ai-os" / "training" / "v4_pipeline" / "models" / "mlx-4bit")
os.makedirs(OUT, exist_ok=True)

# Step 1: Download config + tokenizer (small files)
print("Downloading config and tokenizer...", flush=True)
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    path = hf_hub_download(repo_id=HF_REPO, filename=fname, token=HF_TOKEN)
    shutil.copy2(path, f"{OUT}/{fname}")
    print(f"  Copied {fname}", flush=True)

with open(f"{OUT}/config.json") as f:
    config = json.load(f)

# Step 2: Download model.safetensors to a temp location
print("Downloading model.safetensors (17GB)...", flush=True)
model_file = hf_hub_download(repo_id=HF_REPO, filename="model.safetensors", token=HF_TOKEN)
print(f"Downloaded to: {model_file}", flush=True)

# Step 3: Load weights into memory
print("Loading weights into memory...", flush=True)
weights = mx.load(model_file)

# Step 4: Delete HF cache to free disk space
print("Deleting HF cache to free disk...", flush=True)
hf_cache = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
    print("  HF cache deleted", flush=True)

# Step 5: Remap weight keys
print("Remapping weight keys...", flush=True)
remapped = {}
for k, v in weights.items():
    if k.startswith("model.language_model."):
        remapped[k.replace("model.language_model.", "model.")] = v
    else:
        remapped[k] = v
del weights
print(f"  {len(remapped)} weights remapped", flush=True)

# Step 6: Build model and load weights
args = TextModelArgs.from_dict(config)
args.model_type = "qwen3_5_text"
model = TextModel(args)
model.load_weights(list(remapped.items()), strict=False)
del remapped

# Step 7: Quantize
print("Quantizing to 4-bit...", flush=True)
nn.quantize(model, bits=4, group_size=64)

# Step 8: Save
flat = dict(tree_flatten(model.parameters()))
print(f"{len(flat)} parameters", flush=True)
print("Saving model.safetensors...", flush=True)
mx.save_safetensors(f"{OUT}/model.safetensors", flat)
print("Saved!", flush=True)

# Step 9: Update config with quantization info
config["quantization"] = {"bits": 4, "group_size": 64, "quant_data_type": "uint32"}
with open(f"{OUT}/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Verify
for f in sorted(os.listdir(OUT)):
    p = f"{OUT}/{f}"
    sz = os.path.getsize(p)
    print(f"  {f}: {sz/1e9:.2f} GB", flush=True)

print("\nMLX 4-bit conversion complete!", flush=True)
