# Training Guide — Adaptive Operator Dataset v1

**Dataset:** adaptive_operator_dataset_v1
**Target Model:** Qwen3.5-9B (Adaptive Operator v3)
**Training Methods:** SFT, LoRA/QLoRA, distillation, adaptive compute policy training

---

## Recommended Training Sequence

The dataset is designed for phased training. Each phase builds on the previous.

### Phase 1: Recovery Training (teach failure recovery first)

**File:** `recovery/train.jsonl` (189 examples)
**Rationale:** The most valuable operator behavior is recovering from failures.
Teaching this first ensures the model learns to stop, re-examine evidence, and
form new hypotheses before learning to act quickly.

**What it teaches:**
- Wrong hypothesis → new evidence → correct diagnosis
- Three-strikes rule: change approach after 3 failures, don't tweak parameters
- Evidence-based debugging, not guessing
- Regression detection and rollback

### Phase 2: Tool-Use Training (teach effective tool selection)

**File:** `tools/train.jsonl` (572 examples)
**Rationale:** After learning to recover, the model needs to know which tools
to use and when. This phase teaches tool selection, command construction,
result interpretation, and next-action planning.

**What it teaches:**
- Git operations (log, diff, branch, bisect, grep)
- Build commands (cargo, npm, uv, pytest)
- Debugging tools (gdb, lldb, strace, logging)
- File inspection (read, grep, find, stat)
- Process management (ps, lsof, kill)
- CI/CD operations (gh, docker, actions)

### Phase 3: Engineering Trajectories (teach autonomous work)

**File:** `sft/train.jsonl` (2,297 examples, full dataset)
**Rationale:** With recovery and tool-use as foundation, the model learns
complete engineering trajectories: goal → investigation → action → verification → result.

**What it teaches:**
- Multi-step problem solving
- Investigation before action
- Verification after every change
- Documentation of failures and lessons

### Phase 4: Architecture Reasoning (teach deliberate design)

**File:** `architecture/train.jsonl` (330 examples)
**Rationale:** Architecture decisions require evaluating tradeoffs, considering
constraints, and making deliberate choices. This phase teaches the model to
think before acting on design problems.

**What it teaches:**
- Problem identification and constraint analysis
- Option generation and tradeoff evaluation
- Deliberate decision-making with documented reasoning
- Consequence analysis

### Phase 5: Compute Policy Training (teach adaptive reasoning)

**File:** `compute_policy/train.jsonl` (983 examples, includes proxy wars)
**Rationale:** The final phase teaches the model to classify cognitive
requirement and allocate compute appropriately. This is the foundation for
adaptive compute control in v3.

**What it teaches:**
- FAST: direct action for deterministic tasks
- THINK: deep investigation for ambiguous problems
- RECOVER: new hypothesis when previous approach fails
- VERIFY: evidence gathering after changes
- ESCALATE: recognizing when information is insufficient

---

## LoRA/QLoRA Recommendations

### LoRA Rank

**Recommended rank: 64**

Rationale:
- The dataset teaches behaviors (decision-making, tool use) not just knowledge
- Higher rank (64 vs 16) allows the model to learn new behavioral patterns, not just style
- The dataset is 2,297 examples — large enough to benefit from higher rank
- Lower rank (16-32) may suffice for style transfer but not behavior change
- The user's local-operator router experiments showed that behavioral patterns
  require more capacity than style patterns

### LoRA Alpha

**Recommended alpha: 128 (2x rank)**

Rationale: Standard practice is alpha = 2 * rank. This scaling preserves
the original model's knowledge while allowing the LoRA adapters to
meaningfully shift behavior.

### Target Modules

**Recommended:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

Rationale: Targeting all linear layers (not just attention) allows the model
to learn new behavioral patterns in both attention and feed-forward layers.
For a 9B model, this is feasible with QLoRA 4-bit quantization.

### QLoRA Quantization

**Recommended:** 4-bit (NF4) with double quantization

Rationale:
- 9B model in 4-bit fits in ~6GB VRAM (Apple Silicon unified memory)
- NF4 preserves more information than INT4
- Double quantization reduces memory further
- The user's M4 Mac has unified memory suitable for this

---

## Training Hyperparameters

### Epochs

**Recommended: 3 epochs**

Rationale:
- 1 epoch: model sees each example once — may not fully learn behavioral patterns
- 3 epochs: model sees each example 3 times — sufficient for behavior change without overfitting
- 5+ epochs: risk of overfitting to specific trajectory patterns
- The dataset is 2,297 examples — 3 epochs = 6,891 total steps at batch size 1

### Learning Rate

**Recommended: 2e-4 (with cosine schedule)**

Rationale:
- 2e-4 is standard for LoRA fine-tuning
- Cosine schedule with warmup (3% of total steps) prevents early instability
- For QLoRA, 1e-4 to 2e-4 is the typical range
- Higher learning rates (5e-4) may cause the model to forget base capabilities

### Batch Size

**Recommended: 4 (with gradient accumulation 4 for effective batch 16)**

Rationale:
- Physical batch size 4 fits in memory with 4-bit quantization
- Gradient accumulation 4 gives effective batch size 16
- Larger effective batch sizes (32+) may reduce training stability for behavioral learning

### Mixing Ratios

If training in phases, use these mixing ratios:

| Phase | Data Mix | Ratio |
|-------|----------|-------|
| 1 (Recovery) | recovery/train.jsonl only | 100% recovery |
| 2 (Tools) | 50% tools + 30% recovery + 20% engineering | Mixed |
| 3 (Full SFT) | sft/train.jsonl (full dataset) | 100% full |
| 4 (Architecture) | 40% architecture + 30% engineering + 20% tools + 10% recovery | Mixed |
| 5 (Compute) | compute_policy/train.jsonl only | 100% compute |

**Alternative: Single-phase training**
Mix all data together with this ratio:
- 40% engineering trajectories
- 20% tool use
- 15% proxy wars
- 10% recovery
- 10% architecture
- 5% research

---

## Evaluation Strategy

### Evaluation Set

**File:** `evaluation/test.jsonl` (354 examples)

This set is held out from training and covers all categories:
- coding (170), terminal (128), compute (93), architecture (52)
- recovery (35), debugging (35), failure (26), research (5)

### Evaluation Metrics

1. **Task completion rate:** Can the model complete the task in the trajectory?
2. **Tool selection accuracy:** Does the model choose the right tool?
3. **Recovery success:** Does the model form new hypotheses after failure?
4. **Compute label accuracy:** Does the model choose FAST/THINK correctly?
5. **Verification discipline:** Does the model verify changes before declaring success?

### Evaluation Protocol

1. Run the model on evaluation tasks in a controlled environment
2. Compare model output to the reference trajectory
3. Score on: correct approach, correct tools, correct outcome, correct lesson
4. Measure compute label prediction accuracy separately
5. Track recovery success rate (does model recover from induced failures?)

### Benchmark Integration

The local-operator project has a 150-task benchmark (LOCAL-OPERATOR-BENCH) with
10 task families. After training, evaluate on this benchmark:

```bash
cd ~/Projects/local-operator
uv run operator eval-lora --model mlx-community/Qwen3.5-9B-4bit
```

---

## Distillation Path

For distillation from a larger teacher model:

1. Use the SFT dataset as the student training set
2. Generate teacher trajectories on the same tasks using a 30B+ model
3. Compare student vs teacher trajectories
4. Train on the difference (where teacher outperforms student)

The local-operator project already has this infrastructure:
```bash
uv run operator generate-api --teacher-model Qwen/Qwen3-30B-A3B --mode both --limit 50
uv run operator export-dataset --dir trajectories --output datasets/train.jsonl
```

---

## Adaptive Compute Policy Training

The compute_policy/train.jsonl file contains 983 examples for training the
adaptive compute router. This teaches the model to predict FAST vs THINK
before executing a task.

### Router Training

The local-operator project has a learned router that can be trained on this data:

```bash
cd ~/Projects/local-operator
uv run operator export-router --dir trajectories/proxy_wars --output datasets/router_train.jsonl
uv run operator train-router --dataset datasets/router_train_enriched.jsonl --output models/learned_router.joblib
```

### Router Performance Baseline

From previous experiments (2100 proxy wars, 210 tasks × 10 seeds):

| Router | Accuracy (in-sample) | Group-aware CV | F1 | Regret |
|--------|---------------------|-----------------|------|--------|
| always_fast | 61.1% | — | 0.357 | 817 |
| always_think | 51.6% | — | 0.308 | 1016 |
| difficulty_based | 62.9% | — | 0.574 | 780 |
| HeuristicRouter | 63.4% | ~64% | 0.581 | 768 |
| LearnedRouter | 93.0% | 67.9% | 0.919 | 147 |

The learned router saves 34.5% tokens and 38.1% time vs always-THINK.

---

## Notes

- **Do not shuffle phases:** The recommended sequence builds on previous phases.
  Recovery before tools, tools before engineering, engineering before architecture.
- **Monitor for overfitting:** If the model starts repeating trajectory patterns
  verbatim, reduce epochs or increase dropout.
- **Preserve base capabilities:** Use LoRA (not full fine-tuning) to preserve the
  base model's general capabilities while learning operator behaviors.
- **The compute label in the assistant response is a training signal:** It teaches
  the model to explicitly classify cognitive requirement before acting. This is
  the foundation for v3 adaptive compute control.
