# Local Training Environment

This document defines the two training paths for Adaptive Operator models:
a **cloud path** for initial LoRA fine-tuning and a **local path** for
evaluation, inference, and MLX conversion. The split exists for a hard
reason — the M4 Mac is a daily-driver laptop, not a training rig. Training
runs on cloud GPUs; the Mac runs the resulting quantized model.

> **Hard rule:** Do NOT run local LoRA training on the M4. See the recorded
> lesson `dont_overload_local_machine_training`. Local MLX LoRA while other
> heavy processes are running crashes the machine.

## Cloud Path — Initial LoRA Fine-Tuning

| Field | Value |
|-------|-------|
| Platform | Together AI (managed LoRA) or rented H100 |
| Base model | Qwen3.5-9B (available on Together AI) |
| Method | LoRA fine-tuning |
| Cost | ~$0.48 / M tokens (Together AI); $4 minimum per job |
| Requirement | `TOGETHER_API_KEY` environment variable |
| Output | LoRA adapter (downloadable) |

### Steps

1. **Prepare dataset** — export trajectories to JSONL (see
   `docs/training-data.md` for the trajectory schema). Use the local-operator
   export command:
   ```bash
   uv run operator export-dataset --dir trajectories --output datasets/train.jsonl
   ```
2. **Upload to Together** — submit the JSONL dataset to Together AI as a
   fine-tuning training file.
3. **Launch LoRA job** — start a LoRA fine-tune against `Qwen3.5-9B`:
   ```bash
   uv run operator finetune --dataset datasets/train.jsonl
   ```
   Requires `TOGETHER_API_KEY` in the environment.
4. **Download adapter** — once the job completes, download the LoRA adapter
   weights from Together AI.
5. **Convert to MLX** — merge the adapter with the base model, then quantize
   to 4-bit (see the conversion workflow below).

### Cost notes

- Together AI charges ~$0.48 per million tokens processed during training.
- Each job has a $4 minimum. Small datasets (a few hundred trajectories) are
  cheap; scale matters more than per-job overhead.
- Renting an H100 directly is an alternative when you need full control of the
  training loop (custom optimizers, longer runs, multi-GPU). It is more
  expensive per hour but removes the managed-platform constraints.

## Local Path — Evaluation, Inference, MLX Conversion

| Field | Value |
|-------|-------|
| Platform | Apple Silicon (M4 Mac) |
| Tools | MLX framework (`mlx_lm`, `mlx_vlm`) |
| Use cases | evaluation, inference, MLX conversion |
| Target output | `Qwen3.5-9B-Adaptive-Operator-v3-MLX-4bit` |
| Limitation | No local LoRA training (daily-driver laptop, not a training rig) |

### MLX conversion

After downloading the LoRA adapter from the cloud path, convert the merged
model to MLX 4-bit quantized so it can run efficiently on Apple Silicon.

```bash
# Text-only model
python -m mlx_lm.convert \
  --hf-path <merged-model-path> \
  --quantize --q-bits 4 \
  --mlx-path ./Qwen3.5-9B-Adaptive-Operator-v3-MLX-4bit

# Vision-language model
python -m mlx_vlm.convert \
  --hf-path <merged-model-path> \
  --quantize --q-bits 4 \
  --mlx-path ./Qwen3.5-9B-Adaptive-Operator-v3-MLX-4bit
```

The resulting directory is a self-contained MLX model (weights, tokenizer,
config) that loads directly via the `mlxvlm` provider in MAOS.

## Full Workflow — Cloud Train → Local Run

This is the end-to-end pipeline from raw trajectories to a running v3 model
inside MAOS.

1. **Train on Together AI (cloud)** — prepare the dataset, upload, launch the
   LoRA fine-tune against `Qwen3.5-9B`. Wait for the job to finish.
2. **Download LoRA adapter** — pull the adapter weights from Together AI to
   local disk.
3. **Merge adapter with base model** — combine the LoRA adapter with the
   `Qwen3.5-9B` base weights to produce a full merged model (HuggingFace
   format). This is the step that bakes the fine-tune into the base weights.
4. **Convert to MLX 4-bit quantized** — run `mlx_lm.convert` (or
   `mlx_vlm.convert` for VLMs) with `--quantize --q-bits 4` against the merged
   model. Output goes to a directory named
   `Qwen3.5-9B-Adaptive-Operator-v3-MLX-4bit`.
5. **Copy to `models/adaptive-operator-v3/`** — place the converted MLX model
   under `models/adaptive-operator-v3/` in this repo (see
   `models/adaptive-operator-v3/README.md` for the target spec).
6. **Update MAOS config** — point the MAOS planner model at the new v3 path:
   ```yaml
   # configs/user.yaml
   planner:
     provider: mlxvlm
     model: models/adaptive-operator-v3/Qwen3.5-9B-Adaptive-Operator-v3-MLX-4bit
   ```
   Verify with `uv run app models` that the new model is detected.
7. **Run evaluation harness** — evaluate the v3 model on the benchmark before
   promoting it to the default planner:
   ```bash
   # Local MLX evaluation
   uv run operator eval-lora --model models/adaptive-operator-v3/Qwen3.5-9B-Adaptive-Operator-v3-MLX-4bit
   ```
   Compare against v2 baseline results stored in
   `models/adaptive-operator-v2/evaluation/`. Only promote v3 to the default
   planner if it beats v2 on the benchmark.

## Environment Variables

| Variable | Path | Purpose |
|----------|------|---------|
| `TOGETHER_API_KEY` | cloud | Required for LoRA fine-tuning via Together AI |
| `DEEPINFRA_API_KEY` | cloud | Teacher/critic API inference (local-operator pipeline) |
| `OPENROUTER_API_KEY` | cloud | Optional fallback for API inference |

Store keys in `~/.zshrc` (export) or a gitignored `.env` file. Never hardcode
secrets in config files.

## Related

- `docs/training-data.md` — trajectory schema (the atomic training unit)
- `models/adaptive-operator-v2/README.md` — current production model
- `models/adaptive-operator-v3/README.md` — next training target
- `~/Projects/local-operator/AGENTS.md` — distillation pipeline commands
