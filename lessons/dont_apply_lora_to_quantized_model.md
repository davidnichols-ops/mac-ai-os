# Lesson: Don't Apply LoRA Adapter to Quantized Base Model

**Date**: 2026-08-02
**Severity**: Methodology violation
**Recorded by**: David Nichols

## What Happened

After v3 training completed on Together AI, disk space was tight (26GB free).
Instead of downloading the merged FP16 model (15GB compressed, 19GB extracted)
and converting to MLX 4-bit the standard way, I attempted to:

1. Download a 4-bit quantized base model from HuggingFace
2. Apply the LoRA adapter directly to the 4-bit model at inference time

This is a deviation from professional technical norms.

## Why It's Wrong

1. **LoRA adapters are trained against FP16 weights.** Applying them to 4-bit
   quantized weights introduces compounding quantization error. The adapter
   was trained to modify specific weight values — those values are already
   approximated in the 4-bit model.

2. **Standard practice is: merge adapter into FP16 base → quantize the merged
   model.** This preserves the adapter's learned modifications at full precision
   before quantization rounds them.

3. **Inference quality evaluation should happen on the same platform where the
   model was trained.** Together AI hosts the merged model. Use a dedicated
   endpoint there for evaluation before any local conversion.

## Correct Workflow

1. Train on Together AI (LoRA) ✓
2. Create dedicated endpoint on Together AI for the merged model
3. Run evaluation harness against the Together AI endpoint
4. Compare v2 vs v3 results
5. If v3 wins: download merged FP16 from Together AI
6. Convert merged FP16 → MLX 4-bit (single quantization step)
7. Deploy locally to MAOS

## What I Should Have Done

When disk space was tight, the answer was:
- Use Together AI dedicated endpoint for evaluation (no local disk needed)
- Clean up disk space by removing old v2 merged FP16 (already done)
- Download the merged model only when ready to convert to MLX 4-bit
- If disk space is still insufficient, download to an external drive

Never compromise the evaluation methodology to save disk space. The evaluation
results are only valid if the inference path matches standard practice.
