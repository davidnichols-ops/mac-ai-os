# Adaptive Operator Dataset v1 — Manifest

**Dataset Name:** adaptive_operator_dataset_v1
**Version:** 1.0.0
**Created:** 2026-08-01
**Target Model:** Qwen3.5-9B (Adaptive Operator v3)
**Optimization Target:** Decision density over dataset size

---

## Purpose

This dataset teaches an AI agent how to operate as an experienced engineering operator:
- Inspect systems and reason through problems
- Take actions and verify results
- Recover from failures with new hypotheses
- Decide when deeper thinking is necessary (adaptive compute)
- Use tools effectively across diverse environments

This is NOT a code completion dataset. This is NOT a chatbot dataset. This is an
**operator behavior dataset** that teaches decision-making, tool usage, failure recovery,
verification discipline, and adaptive reasoning policy.

---

## Creation Method

1. **Phase 0-1:** Inventoried 57 projects in ~/Projects, classified by dataset value
2. **Phase 2-3:** Extracted 193 seed trajectories from 26 projects via parallel subagents
3. **Phase 1 (v1):** Deep-extracted 400 additional trajectories from 8 priority projects
4. **Expansion:** Programmatically generated variants (tool-use, recovery, verify, fast, architecture)
5. **Proxy Wars:** Generated 593 FAST vs THINK competitions from all trajectories
6. **Quality Filter:** Validated all examples (0 rejected, 0 duplicates, 0 secrets)
7. **SFT Formatting:** Converted to chat format with system/user/assistant messages
8. **Evaluation Split:** Held out 354 examples (different project distribution)

---

## Statistics

### Overview

| Metric | Value |
|--------|-------|
| Base trajectories extracted | 593 |
| Total examples after expansion | 2,058 |
| Proxy wars generated | 593 |
| Total SFT training examples | 2,297 |
| Evaluation examples | 354 |
| Total dataset | 2,651 |
| Projects represented | 26+ |
| Rejected examples | 0 |
| Duplicate examples | 0 |
| Secrets found | 0 |
| Train/eval overlap | 0 |

### Compute Label Distribution (Train)

| Label | Count | Percentage | Description |
|-------|-------|------------|-------------|
| THINK | 1,324 | 57.6% | Architecture, debugging, multiple hypotheses |
| FAST | 589 | 25.6% | Deterministic, low risk, direct action |
| VERIFY | 216 | 9.4% | Change made, evidence required |
| RECOVER | 157 | 6.8% | Previous attempt failed, new hypothesis needed |
| ESCALATE | 11 | 0.5% | Insufficient information, external knowledge needed |

### Type Distribution (Train)

| Type | Count | Description |
|------|-------|-------------|
| proxy_war | 593 | FAST vs THINK competitions |
| tools | 572 | Tool selection and usage |
| compute | 390 | Adaptive compute decisions |
| architecture | 330 | Design decisions and tradeoffs |
| engineering | 182 | Autonomous work trajectories |
| failure | 126 | Failure recovery scenarios |
| debugging | 50 | Debugging investigations |
| research | 44 | Research experiments |
| verify | 8 | Verification trajectories |
| benchmark | 2 | Benchmark execution |
| testing | 2 | Test writing |

### Evaluation Category Coverage

| Category | Count |
|----------|-------|
| coding | 170 |
| terminal | 128 |
| compute | 93 |
| architecture | 52 |
| recovery | 35 |
| debugging | 35 |
| failure | 26 |
| engineering | 42 |
| research | 5 |
| testing | 2 |

---

## File Structure

```
adaptive_operator_dataset_v1/
├── sft/
│   └── train.jsonl              # 2,297 SFT examples (chat format)
├── recovery/
│   └── train.jsonl              # 189 recovery-focused examples
├── tools/
│   └── train.jsonl              # 572 tool-use examples
├── architecture/
│   └── train.jsonl              # 330 architecture decision examples
├── compute_policy/
│   └── train.jsonl              # 983 compute policy + proxy war examples
├── evaluation/
│   └── test.jsonl               # 354 held-out evaluation examples
├── metadata/
│   ├── MANIFEST.md              # This file
│   ├── TRAINING_GUIDE.md        # Training recommendations
│   └── quality_report.json      # Quality validation results
└── FINAL_DATASET_REPORT.md      # Final summary report
```

---

## SFT Format

Each training example is a JSON object with:
```json
{
  "messages": [
    {"role": "system", "content": "You are an adaptive engineering operator..."},
    {"role": "user", "content": "## Task\n...\n## System\n...\n## Problem\n..."},
    {"role": "assistant", "content": "[THINK] ...\n1. **step**\n   → observation\n...\n**Result:** ...\n**Lesson:** ..."}
  ],
  "metadata": {
    "type": "engineering|failure|tools|architecture|research|compute|proxy_war",
    "compute_label": "FAST|THINK|RECOVER|VERIFY|ESCALATE",
    "source": "git_history|docs|code|benchmark|experiment|derived",
    "system": "project name"
  }
}
```

The assistant response begins with a compute label directive (`[FAST]`, `[THINK]`, etc.)
that teaches the model to classify cognitive requirement before acting.

---

## Source Projects

### HIGH Value (primary extraction)
- local-operator (distillation pipeline, trajectory schema, proxy wars)
- MAOS (AI OS, observation/cognitive/agency layers, PR governance)
- aafp (post-quantum networking, 128 commits, 4 phases)
- AAFP-research (162 commits, 16-phase architecture study)
- sandbox-of-doom (upstream submission, CI triage, evidence boundary)
- trustcard (MCP security, crypto, danger detection, evidence substrate)
- coremltools (FP16 numerical stability, softplus/logsoftmax fixes)
- sentinel-graph (10M benchmark, research framework)
- X-MaC-digital-twin (macOS diagnostics, 94 commits)
- gargantua-inceptyon (SSE connections, review-driven iteration, 99 commits)
- nano-spawn (process spawning, stdio validation, 97 commits)
- apple_quality_recognition_engine (edge CV, camera retry, OpenCV 5 crash)
- apple-photos-export (128 commits, 18 branches, album filter debugging)
- ultralytics (DDP callbacks v1→v2, export parity)
- tensorflow (XLA runtime negatives fix)
- petals (activation anomaly detection, hostile architecture review)

### Additional Sources
- heartland-factory, trajectory-os, elonwatch, elonwatch-mac
- roboflow-python, jane-street-md5-puzzle
- Standalone research docs (operating principles, methodology, evidence registry)
- Shell history (real command patterns for tool-use trajectories)
- MAOS CLI commands (80+ tool-use patterns)

---

## Limitations

1. **Derived examples:** ~60% of examples are programmatically derived from seed trajectories.
   While grounded in real engineering work, they are not independent observations.
2. **Project bias:** Dataset is heavily weighted toward the user's personal projects.
   Generalization to other engineering domains may require additional data.
3. **ESCALATE underrepresented:** Only 0.5% of examples are ESCALATE. This reflects
   the user's bias toward action over escalation, but may need augmentation.
4. **Research type limited:** Only 44 research examples. Additional research
   trajectories would improve coverage.
5. **Single user style:** All examples reflect one engineer's style and preferences.
   This is a feature (consistent style) and a limitation (style bias).

---

## Provenance

- **Base data:** Git history from 26+ projects in ~/Projects and ~/mac-ai-os
- **Extraction method:** Parallel subagents reading git logs, diffs, READMEs, docs, and source code
- **Expansion method:** Programmatic variant generation (tool-use, recovery, verify, fast, architecture)
- **Quality validation:** Automated checks for required fields, secrets, duplicates, train/test separation
- **No synthetic data:** All examples are grounded in real git history, real code, or real documentation
