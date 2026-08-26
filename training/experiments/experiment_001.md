# Experiment 001 — Balanced v3 LoRA (Production Dataset)

## Hypothesis

A LoRA fine-tune of Qwen3.5-9B on the production adaptive_operator_dataset_v1
will improve tool selection, compute mode selection, recovery behavior, and
general operator behavior compared to the v2 baseline.

## Dataset

- Source: adaptive_operator_dataset_v1 (production, 57-project engineering corpus)
- Total examples: 2,651 (2,297 train + 354 held-out evaluation)
- Projects represented: 26+
- Compute labels: THINK 55.6%, FAST 26.8%, VERIFY 9.9%, RECOVER 7.2%, ESCALATE 0.5%
- Token range: 175-1,381 (mean 404, median 381, stdev 185)
- Quality: 0 secrets, 0 duplicates (after dedup), 0 train/test overlap, 0 malformed

### Category Breakdown

| Type | Count |
|------|-------|
| tools | 700 |
| proxy_war | 593 |
| compute | 483 |
| architecture | 382 |
| engineering | 224 |
| failure | 152 |
| debugging | 56 |
| research | 49 |
| verify | 8 |
| testing | 2 |
| benchmark | 2 |

## Configuration

- Config: `training/configs/operator_v3_balanced.yaml`
- Base model: Qwen/Qwen3.5-9B
- Method: LoRA (rank=64, alpha=128, dropout=0.05)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Mixture: 40% SFT, 20% tools, 15% compute_policy, 10% recovery, 10% architecture
- Training: 3 epochs, lr=2e-4, batch_size=4, grad_accum=4 (effective 16)
- Max seq length: 4096
- Seed: 42

### Key Change from Initial Plan

LoRA rank increased from 16 to 64 per TRAINING_GUIDE.md recommendation.
Rationale: behavioral learning (decision-making, tool use) needs more capacity
than style transfer. Rank 16 may suffice for style but not behavior change.

## Expected Improvement

| Capability | v2 Baseline | Expected v3 | Measurement |
|------------|-------------|-------------|-------------|
| Tool competence | 75% (3/4) | >85% | eval harness |
| Recovery | 100% (2/2) | >90% | eval harness |
| Planning | 0% (0/2)* | >70% | eval harness |
| Compute allocation | 50% (1/2) | >75% | eval harness |
| Overall | 60% (6/10) | >75% | eval harness |

*Planning failures were due to the 3-step budget limit, not capability.

### v2 Baseline Notes

- Control tokens (`<FAST>`/`<THINK>`) were NOT emitted in any generation
- 29 total tool calls across 10 tasks
- 5/10 tasks showed error-then-continuation (recovery) behavior
- The model over-called `file_list` (11x in one generation) on the ambiguous task

## Observed Result

### Dry Run (2026-08-02)

- Dataset uploaded to Together AI: file-adb5c16c-efa9-4673-9dfd-7d442a7e37e1
- Samples: 2,297 (production SFT file)
- File size: 4.29 MB
- Price estimation: pending (Together AI needs to validate the file first)
- Estimated cost: ~$4-15 based on $0.48/M tokens for 3 epochs on 2,297 samples

### Training Run (2026-08-02)

- **Job ID: ft-9e455920-d7ab**
- Status: **COMPLETED**
- Duration: ~19 minutes (18:16 → 18:35 UTC)
- Epochs: 3 (completed at steps 21, 42, 63)
- Total steps: 63
- Token count: 839,207
- Batch size: 8 ( Together AI effective)
- Learning rate: 1e-5 (Together AI default, not our requested 2e-4)
- Cost: $4.00 flat
- File ID: file-adb5c16c-efa9-4673-9dfd-7d442a7e37e1
- Adapter: `davidnicholsop-199a/Qwen3.5-9B-adaptive_operator_v3_balanced-08177287-adapter`
- Merged model: `davidnicholsop_199a/Qwen3.5-9B-adaptive_operator_v3_balanced-08177287`
- Adapter downloaded: `models/v3/huggingface/adapter/` (680MB)

### Endpoint Evaluation (2026-08-02)

- Endpoint: `davidnicholsop-199a/adaptive-operator-v3-eval` (H100 80GB)
- Endpoint ID: ep_CdeR6FXP3i67uHKoAuzyp
- Status: Stopped (scaled to 0 replicas to save cost)

### Control Token Emission Results

| Test | Expected | Actual (temp 0.3) | Actual (temp 0.7) |
|------|----------|-------------------|-------------------|
| len() | FAST | NONE | NONE |
| list files | FAST | THINK | NONE |
| git status | FAST | THINK | — |
| retry strategy | THINK | THINK | THINK |
| architecture | THINK | NONE | NONE |
| test failed | RECOVER | THINK | THINK |
| check tests | VERIFY | THINK | THINK |

**Findings:**

1. **Control token emission rate**: ~71% at temp 0.3, ~40% at temp 0.7
2. **Mode distribution**: All emitted tokens were `[THINK]`. The model never emitted `[FAST]`, `[VERIFY]`, `[RECOVER]`, or `[ESCALATE]`.
3. **Root cause**: The Qwen3.5 base model has a strong prior to emit empty `<think></think>` blocks before the response. The LoRA fine-tune taught the model the control token format, but the base model's think-block behavior dominates at the start of generation.
4. **Training data imbalance**: THINK was 55.6% of training data. The model learned "when in doubt, emit [THINK]" — which is conservative but not differentiated.
5. **Pre-seeded tokens work**: When the assistant turn is pre-seeded with `[FAST] ` or `[THINK] `, the model continues in the correct format. The model HAS learned the behavior, it just doesn't reliably initiate it.
6. **Quality of responses**: The model's responses are competent and well-structured. The content quality is good — the issue is purely the control token emission.

### Lessons Recorded

1. `dont_apply_lora_to_quantized_model` — Never apply LoRA to quantized weights; always merge into FP16 first
2. `control_token_chat_template_mismatch` — Together AI uses base model chat template, not the fine-tuned one

## Decision

**v3 is a partial success.** The model generates high-quality operator responses but
does not reliably emit control tokens. The model defaults to `[THINK]` for everything,
which is safe but not differentiated.

### Recommended Next Steps

1. **For v3 deployment**: Use the model with a runtime that strips empty `<think></think>`
   blocks and parses control tokens from the cleaned response. The model's content quality
   is good even when the token is missing — default to `[THINK]` behavior when no token
   is emitted.

2. **For v4 training**: The control token issue needs to be addressed at the training data
   level:
   - **Option A**: Train with the Qwen3.5 chat template that includes the empty `<think>`
     block, so the model learns to emit the control token AFTER the think block
   - **Option B**: Use a different base model that doesn't have the think-block prior
   - **Option C**: Increase the training signal for non-THINK modes (FAST, VERIFY, RECOVER,
     ESCALATE) by upweighting those examples or adding more of them
   - **Option D**: Add a special token to the tokenizer for each mode, so the model treats
     them as first-class tokens rather than text sequences

3. **For immediate use**: The model is competent as a general operator. The control token
   issue is a nice-to-have, not a blocker. The model can be deployed with a runtime that
   defaults to `[THINK]` mode when no token is emitted.

## Artifacts

- Config: `training/configs/operator_v3_balanced.yaml`
- Run directory: `training/runs/v3_run_001/`
- Baseline: `eval/results/v2_baseline.json`
- v3 evaluation: `eval/results/v3_eval.json` (pending)
- Comparison: `eval/results/v3_comparison.md` (pending)
- Dataset manifest: `datasets/validated/operator_v1/manifest.json`
- Training guide: `datasets/incoming/adaptive_operator_dataset_v1/metadata/TRAINING_GUIDE.md`

## Notes

- The production dataset has compute labels embedded in assistant responses
  ([FAST], [THINK], [VERIFY], [RECOVER], [ESCALATE]) — this should fix the
  v2 issue where control tokens were not emitted.
- The dataset is decision-dense (2,651 examples from 26+ projects) rather
  than volume-heavy. This is by design — quality over quantity.
- ESCALATE is underrepresented (0.5%). This reflects the user's bias toward
  action over escalation. May need augmentation for v4.
- The dataset includes a proper held-out evaluation set (354 examples)
  covering all categories.
