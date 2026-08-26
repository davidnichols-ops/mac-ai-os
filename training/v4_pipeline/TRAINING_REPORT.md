# Adaptive Operator v4 — Training Report

## Model

| Field | Value |
|-------|-------|
| Base model | Qwen/Qwen3.5-9B |
| Student target | Qwen2.5-0.5B (distillation) |
| Teacher | Qwen v3.1 (Together AI dedicated endpoint) |
| Training method | LoRA SFT + DPO |
| Precision | bf16 (no quantization) |
| LoRA rank | 64 (alpha 128) |
| Training data | 4,992 SFT examples + 5,000 DPO pairs |
| Training hardware | RTX PRO 5000 Blackwell (48GB VRAM) |

## Control Token System

The model is trained to emit one of five control tokens at the start of every response, routing the task to the appropriate compute level:

| Token | Usage | Distribution in training data |
|-------|-------|-------------------------------|
| `[FAST]` | Direct action, simple tasks | 47.4% (2,366) |
| `[THINK]` | Multi-step reasoning before acting | 41.6% (2,078) |
| `[VERIFY]` | Act then confirm result (destructive/irreversible) | 2.9% (145) |
| `[RECOVER]` | Reassess after failure, try different strategy | 7.1% (354) |
| `[ESCALATE]` | Surface for human decision (high-risk/security) | 1.0% (49) |

## Tool Call Format

The model emits structured tool calls in XML+JSON format:

```xml
<tool_call>
{"name": "shell", "arguments": {"command": "uv run pytest"}}
</tool_call>
```

Supported tools: `shell`, `file_read`, `file_write`, `file_edit`, `file_list`, `grep`, `find_file`, `git`, `web_search`, `web_fetch`, `todo_write`, `ask_user`.

---

## Pipeline Architecture

```
Prompt Generator (5K unique prompts)
    ↓
Qwen v3.1 Teacher Inference (Together AI, 32 parallel workers)
    ↓
5-Reviewer Orchestrator (pure Python heuristics)
    ↓
Data Quality Filter (dedup + n-gram decontamination)
    ↓
SFT Export + DPO Pair Generation
    ↓
LoRA SFT Training (bf16, 3 epochs)
    ↓
LoRA DPO Training (bf16, 1 epoch)
    ↓
Merge LoRA → fp16 model
    ↓
Smoke Test (control token + tool call format verification)
```

---

## Issues Encountered

### 1. Agent Tool Parser Conflict

**Problem:** The orchestrating agent's tool parser interpreted JSON-like content within Python string literals as actual tool calls, preventing direct file writing of code containing `<tool_call>` XML tags or JSON payloads.

**Fix:** Used subagents for file writing, with the main agent orchestrating. All Python files containing tool-call format strings were written by subagents that don't have the same parser conflict.

### 2. Together AI Endpoint Scaling to Zero

**Problem:** The `davidnicholsop-199a/adaptive-operator-v3-1-eval` dedicated endpoint scaled to zero after idle period, then refused connections on restart.

**Fix:** Required explicit restart via Together AI v2 API. The v1 API for dedicated endpoints is deprecated. Endpoint creation/restart requires specific `autoscaling` and `hardware` arguments.

### 3. Together AI API Domain Change

**Problem:** API domain changed from `api.together.xyz` to `api.together.ai` mid-session, causing silent failures in inference code using the old domain.

**Fix:** Updated all API URLs to `api.together.ai`. The inference client (`qwen_inference.py`) was updated to use `api-inference.together.ai` for the inference endpoint.

### 4. Together AI Billing Block

**Problem:** Fine-tuning job submission blocked due to insufficient credits ($4.00 minimum required).

**Fix:** User added credits. This triggered the strategic shift from Together AI hosted training to self-managed GPU rental.

### 5. Duplicate Responses from Resume Mode

**Problem:** The SFT generator's resume mode appended to the output file without deduplication, resulting in 9,394 lines for 4,953 written responses (4,953 duplicates).

**Fix:** Post-generation deduplication by `prompt_id`. Replaced the original file with the deduplicated version (5,000 unique entries).

### 6. Missing Control Tokens in Teacher Responses

**Problem:** 832 of 5,000 teacher responses (16.6%) had `mode=None` — the teacher model failed to emit a control token. Additionally, 99.9% of responses narrated tool use ("select tool: X", "→ command") instead of emitting structured `<tool_call>` blocks.

**Fix:** The 5-reviewer orchestrator detects missing control tokens and synthesizes the correct token based on the prompt category. Narrated tool use is converted to structured XML+JSON tool calls during the synthesis step.

### 7. Review Orchestrator Score Distribution

**Problem:** Average review scores revealed systematic teacher weaknesses:
- `tool_selection`: 2.78/10 (narration instead of structured calls)
- `format_clarity`: 2.17/10 (verbose, wrong format)
- `control_token`: 6.06/10 (16% missing tokens)
- `code_quality`: 7.34/10 (acceptable)
- `error_handling`: 6.53/10 (missing failure-mode handling)

**Fix:** The synthesis step produces clean, correctly-formatted responses that fix all flagged issues. The original teacher response becomes the DPO "rejected" example; the synthesized improved response becomes the "chosen" example.

### 8. Colab T4 Limitations

**Problem:** Google Colab T4 has several constraints:
- 16GB VRAM (requires 4-bit QLoRA)
- No bf16 support (only fp16, causing overflow/underflow risks)
- One concurrent GPU session per account (HTTP 412 on second allocation)
- Session loss after long-running jobs (websocket timeout, VM kernel reclaim)
- Module caching in kernel (patching .py files requires kernel restart)

**Fix:** Shifted to vast.ai for dedicated GPU access. Colab notebooks remain as fallback/interactive options.

### 9. Vast.ai 2FA Blocking API Operations

**Problem:** Vast.ai API requires 2FA session authentication for instance management operations (list, destroy). The API key alone returns 401.

**Fix:** Used direct SSH access (user-provided connection strings) instead of the vast.ai CLI for instance management. Instance destruction must be done manually from the dashboard.

### 10. PyTorch CUDA Version Mismatch (RTX 3090)

**Problem:** The vast.ai RTX 3090 instance had driver 570.133.07 (CUDA 12.8), but the pre-installed PyTorch 2.13.0 was built for CUDA 13.0. `torch.cuda.is_available()` returned False.

**Fix:** Installed `torch==2.11.0+cu128` from the PyTorch CUDA 12.8 wheel index. Required uninstalling the old torch and cleaning ~3.4GB of bundled NVIDIA CUDA libraries to free disk space.

### 11. fp16 Out of Memory (RTX 3090, 24GB)

**Problem:** Full fp16 LoRA training of a 9B model on 24GB VRAM caused CUDA OOM. The model weights alone consume ~18GB, leaving insufficient room for optimizer states and activations.

**Fix:** Switched to 4-bit NF4 QLoRA with bf16 compute dtype. Model footprint dropped to 7.64GB, leaving 16GB for training. However, throughput was only ~40s/step (4.3 hour ETA for SFT alone).

### 12. Flash Attention Build Failures

**Problem:** `flash-attn` failed to build from source on both the RTX 3090 and RTX PRO 4000 Blackwell instances. No prebuilt wheel available for Python 3.12 + CUDA 13.0.

**Impact:** Without flash attention, the `packing=True` option in SFTTrainer risks cross-contamination between packed sequences (samples in the same packed batch can attend to each other).

**Fix:** Set `attn_implementation="sdpa"` (PyTorch native scaled dot-product attention) and `packing=False` to avoid cross-contamination. SDPA is reasonably fast on Blackwell but slower than flash attention would be.

### 13. Disk Space Exhaustion on Vast.ai

**Problem:** Vast.ai containers have only 32GB of disk space. Installing PyTorch + transformers + peft + trl + bitsandbytes consumed ~25GB, leaving insufficient space for the 9B model download (~18GB in fp16).

**Fix:** Redirected the HuggingFace cache to `/dev/shm` (RAM-backed tmpfs, 50GB on the 3090 instance, 136GB on the Blackwell). Set `HF_HOME=/dev/shm/huggingface` environment variable.

### 14. RTX PRO 4000 Blackwell — Insufficient Throughput

**Problem:** The RTX PRO 4000 Blackwell (24GB) loaded the 4-bit model successfully but training throughput was ~40s/step — approximately 4.3 hours for SFT alone. The GPU was underutilized due to the small batch size (1) and gradient checkpointing overhead.

**Fix:** Upgraded to RTX PRO 5000 Blackwell (48GB), which allows full bf16 training (no quantization), batch size 4, and LoRA rank 64. Throughput improved to ~2.6s/step (80 min for SFT).

---

## Strategic Decisions

### 1. Together AI → Colab → Vast.ai Migration

**Decision:** Migrate training infrastructure from Together AI hosted fine-tuning to Google Colab T4 to vast.ai dedicated GPUs.

**Rationale:** Together AI required $4+ in credits for fine-tuning and offered less control over training hyperparameters. Colab T4 is free but limited (16GB, fp16 only, session instability). Vast.ai provides dedicated GPUs at $0.11-0.20/hr with full control. The RTX PRO 5000 Blackwell at 48GB enables full bf16 training without quantization — the highest quality training path.

**Trade-off:** Vast.ai requires manual setup (SSH, package installation, data upload) vs Together AI's managed API. The quality gain from bf16 + LoRA r=64 vs 4-bit QLoRA r=8 justifies the operational overhead.

### 2. Pure-Python 5-Reviewer Orchestrator (No API Calls)

**Decision:** Implement the 5-reviewer quality assessment as pure Python heuristics rather than using LLM API calls for review.

**Rationale:**
- Processing 5,000 responses through 5 LLM API reviewers would cost ~$15-30 in API calls and take hours
- The heuristics target specific, well-defined failure modes (narration vs structured tool calls, missing control tokens, fabricated results, wrong tool selection)
- Pure Python runs at 1,700 responses/sec — 5,000 responses in 3 seconds
- The synthesis step is template-based, producing deterministic, correctly-formatted outputs

**Trade-off:** Heuristics miss subtle quality issues that an LLM reviewer would catch. The improved responses are structurally correct but may not improve the underlying reasoning quality of the teacher's answers. This is acceptable because the primary goal is format compliance (control tokens + tool calls), not reasoning improvement.

### 3. DPO Pair Design: Cross-Model, Not Self-Preference

**Decision:** Generate DPO preference pairs by comparing the original Qwen v3.1 response (rejected) against the synthesized improved response (chosen), rather than using self-preference (temperature-sampled pairs from the same model).

**Rationale:**
- The teacher's responses have systematic format failures (99.9% narration, 16% missing control tokens) — these are clear, objective quality differences
- Self-preference pairs require multiple samples per prompt (higher API cost) and the quality difference between samples is often noise, not signal
- Cross-model pairs provide a stronger learning signal: the model learns "don't narrate, emit structured tool calls" and "don't fabricate results"

**Trade-off:** Cross-model pairs may over-penalize the teacher's reasoning quality (which was actually decent — 7.34/10 code quality score). The DPO signal is dominated by format differences rather than reasoning differences.

### 4. LoRA Rank Scaling with VRAM

**Decision:** Scale LoRA rank proportionally to available VRAM: r=8 (T4 16GB), r=16 (3090 24GB QLoRA), r=32 (3090 24GB QLoRA), r=64 (Blackwell 48GB bf16).

**Rationale:** Higher rank captures more of the learned behavior. The control token routing and tool call format are fundamentally new behaviors not present in the base model, requiring more adapter capacity than a simple style transfer. With 48GB VRAM and bf16, r=64 adds only ~200MB to the model footprint.

**Trade-off:** Higher rank increases the risk of overfitting on the relatively small dataset (5K examples). Mitigated by 3 epochs with cosine LR schedule and dropout 0.05.

### 5. Template-Based Synthesis Instead of LLM-Based Improvement

**Decision:** The synthesis step that produces "improved" responses uses deterministic Python templates, not an LLM call to rewrite the response.

**Rationale:**
- Deterministic templates guarantee format compliance (control token + structured tool calls)
- LLM-based rewriting would introduce the same failure modes we're trying to fix (the rewriter might narrate instead of emit tool calls)
- Templates produce consistent, reproducible training data
- No additional API cost

**Trade-off:** The improved responses are structurally correct but semantically simpler than the teacher's original reasoning. The coding answers are minimal stubs ("Read the code, identify the failing path, apply the minimal fix, then run tests") rather than actual code. This is acceptable for SFT because the model learns the format and routing behavior; the base model's existing coding ability provides the semantic content.

### 6. Data Quality Filtering

**Decision:** Apply deduplication (exact + fuzzy Jaccard), n-gram decontamination (8-gram overlap with eval prompts), and length/quality filtering before training.

**Rationale:**
- The resume-mode deduplication issue showed that duplicate data can silently inflate dataset size
- N-gram decontamination prevents training-test leakage
- Quality filtering (min score threshold) removes the worst examples

**Result:** 8 fuzzy duplicates removed (5,000 → 4,992), 0 decontamination removals (eval prompts are sufficiently different from training prompts).

### 7. bf16 Over fp16 on Blackwell

**Decision:** Use bf16 instead of fp16 for training on the Blackwell GPU.

**Rationale:**
- bf16 has the same dynamic range as fp32 (8 exponent bits) but less precision (7 mantissa bits vs 10 for fp16)
- fp16 is prone to overflow/underflow in gradient computation, especially with gradient checkpointing
- Blackwell natively supports bf16 (unlike the T4 which only supports fp16)
- bf16 eliminates the need for gradient scaling (no `GradScaler` needed)

**Trade-off:** Slightly less numerical precision than fp16, but the dynamic range advantage far outweighs this for training stability.

### 8. No Packing Without Flash Attention

**Decision:** Disable sequence packing (`packing=False`) when flash attention is not available.

**Rationale:** Packing concatenates multiple training examples into a single sequence to maximize GPU utilization. Without flash attention's variable-length attention, packed sequences allow cross-attention between examples, causing cross-contamination. SDPA (PyTorch's built-in attention) does not support variable-length attention masks efficiently.

**Trade-off:** ~20-30% throughput reduction vs packed training. Mitigated by the Blackwell's raw speed (2.6s/step unpacked is still fast).

---

## Training Configuration (Final)

```python
BASE_MODEL = "Qwen/Qwen3.5-9B"
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# SFT
SFT_EPOCHS = 3
SFT_LEARNING_RATE = 2e-4
SFT_BATCH_SIZE = 4
SFT_GRADIENT_ACCUMULATION = 2  # effective batch = 8
MAX_SEQ_LENGTH = 2048

# DPO
DPO_EPOCHS = 1
DPO_LEARNING_RATE = 5e-5
DPO_BETA = 0.1

# Infrastructure
PRECISION = "bf16"
ATTENTION = "sdpa"
PACKING = False
GRADIENT_CHECKPOINTING = True
GPU = "RTX PRO 5000 Blackwell (48GB)"
```

## Training Metrics (SFT, first 110 steps)

| Step | Loss | Token Accuracy | Learning Rate |
|------|------|----------------|---------------|
| 10 | 2.088 | 54.3% | 9.6e-06 |
| 20 | 1.294 | 69.8% | 2.0e-05 |
| 70 | 0.106 | 97.7% | 7.3e-05 |
| 80 | 0.101 | 97.8% | 8.4e-05 |
| 90 | 0.095 | 98.0% | 9.5e-05 |
| 100 | 0.085 | 98.2% | 1.1e-04 |
| 110 | 0.082 | 98.2% | 1.2e-04 |

Loss converged rapidly (2.09 → 0.08 in 110 steps), indicating the format/routing behavior is easy to learn. Token accuracy at 98.2% after 6% of training.

## Dataset Statistics

### SFT Dataset (4,992 examples)

| Category | Count | Percentage |
|----------|-------|------------|
| tool_use_file_ops | 889 | 17.8% |
| coding_write | 828 | 16.6% |
| coding_debug | 615 | 12.3% |
| tool_use_shell | 543 | 10.9% |
| tool_use_git | 435 | 8.7% |
| coding_test | 404 | 8.1% |
| coding_refactor | 376 | 7.5% |
| planning | 374 | 7.5% |
| tool_use_search | 298 | 6.0% |
| recovery | 238 | 4.8% |

### Difficulty Distribution

| Level | Count | Percentage |
|-------|-------|------------|
| 1 (simple) | 2,374 | 47.5% |
| 2 (moderate) | 2,035 | 40.8% |
| 3 (complex) | 591 | 11.8% |

### DPO Dataset (5,000 pairs)

| Field | Mean Length (chars) |
|-------|---------------------|
| Prompt | 70 |
| Chosen (improved) | 185 |
| Rejected (original) | 950 |

The rejected responses are 5x longer than chosen — the teacher was verbose with narration. The improved responses are concise and structured.

## Review Orchestrator Findings

| Metric | Value |
|--------|-------|
| Total responses processed | 5,000 |
| Mode corrections (wrong control token) | 2,418 (48.4%) |
| Narration fixed (narrated → structured) | 4,997 (99.9%) |
| Processing time | 3.08 seconds |
| Processing rate | 1,623 responses/sec |

## Infrastructure Cost Summary

| Resource | Usage | Cost |
|----------|-------|------|
| Together AI inference (5K responses) | 5.4M tokens | ~$2-3 |
| Google Drive storage | ~25MB | Free |
| Colab T4 (attempted, abandoned) | ~10 min | Free |
| Vast.ai RTX 3090 (abandoned) | ~20 min | ~$0.04 |
| Vast.ai RTX PRO 4000 (abandoned) | ~15 min | ~$0.02 |
| Vast.ai RTX PRO 5000 Blackwell | ~2 hours (est.) | ~$0.30 |
| **Total** | | **~$3.50** |

## Files

| File | Purpose |
|------|---------|
| `prompt_generator.py` | Generates 5K diverse unique prompts across 10 categories |
| `qwen_inference.py` | Together AI inference client (handles reasoning_content quirk) |
| `sft_generator.py` | Parallel teacher response generation (ThreadPoolExecutor, 32 workers) |
| `review_orchestrator.py` | 5-reviewer pure-Python quality assessment + synthesis |
| `data_quality.py` | Dedup, n-gram decontamination, length/quality filtering |
| `sft_export.py` | Convert improved responses to OpenAI chat format for SFT |
| `dpo_generator.py` | Generate cross-model DPO preference pairs |
| `dpo_export.py` | Export DPO pairs to Together AI format |
| `training_loop.py` | Self-improving loop orchestrator (generate→train→eval→improve) |
| `vast_train.py` | fp16/bf16 LoRA SFT+DPO training script for vast.ai |
| `colab_qlora_training.ipynb` | Colab T4 QLoRA training notebook |
| `colab_eval.ipynb` | Colab evaluation notebook with statistical significance testing |
| `colab_utils.py` | Shared utilities for Colab notebooks |
| `colab_eval_utils.py` | Evaluation utilities (response parsing, tool schemas, stats) |
| `eval_suite.py` | Evaluation suite with control token + tool call format checks |
| `test_cases.py` | Test cases for evaluation |
| `stats.py` | Statistical significance testing (Wilson interval, Fisher exact, Cohen's d) |
