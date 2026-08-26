# Adaptive Operator v1 — Dataset Statistics (Production)

## Overview

| Metric | Value |
|--------|-------|
| Total incoming lines | 4,725 |
| Validated samples | 2,651 |
| Rejected (schema) | 0 |
| Rejected (secrets) | 0 |
| Duplicates removed | 2,074 |
| Malformed | 0 |
| Train split | 2,297 |
| Test split | 354 |
| Validation split (internal) | 133 |
| Task ID leakage | 0 |

The 2,074 duplicates are expected — the SFT file contains all examples, and
the category-specific files (recovery, tools, architecture, compute_policy)
are subsets of the SFT file. The importer correctly deduplicated them.

## Dataset Composition

| Type | Count | Description |
|------|-------|-------------|
| tools | 700 | Tool selection and usage |
| proxy_war | 593 | FAST vs THINK competitions |
| compute | 483 | Adaptive compute decisions |
| architecture | 382 | Design decisions and tradeoffs |
| engineering | 224 | Autonomous work trajectories |
| failure | 152 | Failure recovery scenarios |
| debugging | 56 | Debugging investigations |
| research | 49 | Research experiments |
| verify | 8 | Verification trajectories |
| testing | 2 | Test writing |
| benchmark | 2 | Benchmark execution |

## Compute Label Distribution

| Label | Count | Percentage | Description |
|-------|-------|------------|-------------|
| THINK | 1,474 | 55.6% | Architecture, debugging, multiple hypotheses |
| FAST | 711 | 26.8% | Deterministic, low risk, direct action |
| VERIFY | 263 | 9.9% | Change made, evidence required |
| RECOVER | 190 | 7.2% | Previous attempt failed, new hypothesis needed |
| ESCALATE | 13 | 0.5% | Insufficient information, external knowledge needed |

## Token Statistics

| Metric | Value |
|--------|-------|
| Min tokens | 175 |
| Max tokens | 1,381 |
| Mean tokens | 403.92 |
| Median tokens | 381 |
| Stdev | 185.29 |

The tight stdev (185) indicates a well-bounded dataset — no extreme outliers.
All samples fit within a 4096 token context window with room to spare.

## Projects Represented

26+ projects from the user's engineering corpus, including:

- **local-operator** — distillation pipeline, trajectory schema, proxy wars
- **MAOS** — AI OS, observation/cognitive/agency layers, PR governance
- **aafp** — post-quantum networking, 128 commits, 4 phases
- **AAFP-research** — 162 commits, 16-phase architecture study
- **sandbox-of-doom** — upstream submission, CI triage
- **trustcard** — MCP security, crypto, danger detection
- **coremltools** — FP16 numerical stability fixes
- **sentinel-graph** — 10M benchmark, research framework
- **X-MaC-digital-twin** — macOS diagnostics
- **gargantua-inceptyon** — SSE connections, review-driven iteration
- **nano-spawn** — process spawning, stdio validation
- **apple_quality_recognition_engine** — edge CV, OpenCV 5 crash
- **apple-photos-export** — album filter debugging
- **ultralytics** — DDP callbacks, export parity
- **tensorflow** — XLA runtime negatives fix
- **petals** — activation anomaly detection

## Security Validation

| Check | Result |
|-------|--------|
| API keys | 0 found |
| Tokens | 0 found |
| Passwords | 0 found |
| Absolute paths | Scrubbed |
| Private filesystem refs | Scrubbed |
| Email addresses | Redacted |
| IP addresses | Redacted |

## Quality Assessment

| Check | Result |
|-------|--------|
| Duplicate count | 2,074 (expected — SFT superset) |
| Empty trajectories | 0 |
| Malformed JSON | 0 |
| Missing fields | 0 |
| Train/test overlap | 0 |
| Category coverage | 11 types, all represented |
| Compute label coverage | 5 labels (FAST/THINK/VERIFY/RECOVER/ESCALATE) |

## Training Readiness

The dataset is ready for training. Key properties:

1. **Decision density over dataset size** — 2,651 carefully curated examples
2. **Phased training support** — separate files for recovery, tools, architecture, compute_policy
3. **Compute labels embedded** — assistant responses begin with [FAST]/[THINK]/[VERIFY]/[RECOVER]/[ESCALATE]
4. **Held-out evaluation** — 354 examples covering all categories
5. **No secrets, no leakage, no malformed data**

## Training Guide Summary (from metadata/TRAINING_GUIDE.md)

### Recommended LoRA Parameters
- **Rank: 64** (higher than typical — behavioral learning needs more capacity)
- **Alpha: 128** (2x rank)
- **Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj**
- **QLoRA 4-bit NF4** with double quantization

### Recommended Training Sequence
1. Recovery first (189 examples) — teach failure recovery
2. Tool use (572 examples) — teach tool selection
3. Full SFT (2,297 examples) — autonomous work
4. Architecture (330 examples) — deliberate design
5. Compute policy (983 examples) — adaptive compute routing

### Alternative: Single-phase
Mix all data: 40% engineering, 20% tools, 15% proxy wars, 10% recovery, 10% architecture, 5% research
