# Phase 2 Decision — After Dataset Integration

## Status

Phase 1.5 is complete. The following are in place:

- [x] Stable agent loop (think-act-observe-verify)
- [x] Tool calling (6 built-in tools with schemas)
- [x] Safety gates (LOW/MEDIUM/HIGH/CRITICAL)
- [x] Trajectory capture (schema v2.1, training-quality telemetry)
- [x] Dataset importer (datasets/adaptive_operator_v1/)
- [x] Training pipeline (SFT, preference, compute policy formats)
- [x] Evaluation harness (eval/operator_v3/)
- [x] v3 training configuration (training/configs/operator_v3.yaml)
- [x] Reproducible model conversion path (Together AI → MLX 4-bit)

## Decision

**Preferred path: B — Train v3 first, then continue runtime expansion.**

```
Dataset integration
    |
    v
Small v3 LoRA
    |
    v
Evaluate
    |
    v
Continue runtime expansion (Phase 2+)
```

## Rationale

A stronger model will reveal architecture needs better than assumptions.

The current v2 model can call tools and complete simple tasks, but it has
known limitations:
- No native function calling (text-based tool call format)
- Limited recovery behavior
- No verification step
- Keyword-based routing still present

Training v3 on the v1 dataset + MAOS-generated trajectories will produce
a model that:
- Selects tools more accurately
- Recovers from errors more reliably
- Verifies its work before declaring success
- Allocates compute (FAST/THINK) more efficiently

With v3 in hand, Phase 2 (routing removal) becomes natural — the model
itself decides compute mode, not deterministic code. Phase 3 (tool ecosystem
expansion) becomes more productive — a better model uses new tools more
effectively.

## Alternative: Option A — Proceed to Phase 2 routing removal

If v3 training is delayed or produces insufficient improvement:
- Remove keyword routing from the orchestrator
- Let the model self-select compute mode via control tokens
- Expand the tool ecosystem
- Generate more trajectories with the v2 model
- Train v3 later with a richer dataset

This is viable but suboptimal. The v2 model's tool selection and recovery
are weak enough that Phase 2 changes won't be fully exercised. We'd be
building architecture for a model that can't yet take advantage of it.

## Recommendation

1. Import the v1 dataset when it arrives
2. Run the dataset importer and generate the import report
3. Prepare the SFT training data from v1 + MAOS trajectories
4. Launch a small v3 LoRA job on Together AI (~$29, ~2 hours)
5. Convert to MLX 4-bit
6. Run the evaluation harness
7. If v3 shows improvement: proceed to Phase 2
8. If v3 doesn't improve: investigate training data quality before proceeding

## Definition of Done for Phase 1.5

- [x] Stable agent loop
- [x] Tool calling
- [x] Safety gates
- [x] Trajectory capture
- [x] Dataset importer
- [x] Training pipeline
- [x] Evaluation harness
- [x] v3 training configuration
- [x] Reproducible model conversion path

## Final Directive

Do not optimize MAOS around Qwen3.5-9B v2. Build the environment where
Adaptive Operator v3 can learn, operate, fail, recover, and improve.
The runtime is the long-term asset; the model checkpoint is only one
iteration.
