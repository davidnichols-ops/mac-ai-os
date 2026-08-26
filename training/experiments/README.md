# Experiment Ledger — Adaptive Operator Training

This directory contains the scientific record of every training experiment.
Each experiment is documented in its own file with:

- Hypothesis
- Dataset
- Configuration
- Expected improvement
- Observed result
- Decision

The project accumulates scientific knowledge, not just checkpoints.

## Experiments

| # | Name | Date | Status | Result |
|---|------|------|--------|--------|
| 001 | Balanced v3 LoRA | 2026-08-02 | Pending | — |

## Naming Convention

```
experiment_NNN.md
```

Where NNN is a zero-padded sequential number.

## Decision Protocol

After each experiment, one of three decisions:

1. **Continue** — improvement observed, proceed to next experiment
2. **Adjust** — partial improvement, modify mixture/hyperparameters and retry
3. **Investigate** — no improvement, analyze dataset quality or evaluation failures

Do not immediately add complexity. A failed experiment is still valuable data.
