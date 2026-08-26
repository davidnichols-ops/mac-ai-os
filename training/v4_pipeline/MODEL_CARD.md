---
library_name: mlx
base_model: Qwen/Qwen3.5-9B
tags:
  - mlx
  - mlx-lm
  - loRA
  - sft
  - dpo
  - agent
  - tool-use
  - control-tokens
  - adaptive-compute
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
model_type: qwen3_5_text
---

# Adaptive Operator v4

A Qwen3.5-9B model fine-tuned with a custom 5-token control system for adaptive compute allocation in agentic workflows. The model routes every task to one of five compute levels at inference time, emitting a control token at the start of each response before executing structured tool calls.

**This is a research artifact, not a production model.** The fine-tune successfully taught control token routing and tool call formatting, but it degraded the base model's coding ability by 47 percentage points. The lessons learned from this project are more valuable than the model itself. Read on for the full honest breakdown.

---

## Nerd TL;DR

> For engineers who want the raw facts, warts and all.

**What this is:** A LoRA SFT + DPO fine-tune of Qwen3.5-9B that teaches the model to emit one of five control tokens (`[FAST]`, `[THINK]`, `[VERIFY]`, `[RECOVER]`, `[ESCALATE]`) and make structured XML+JSON tool calls. Trained on 4,992 SFT examples + 5,000 DPO pairs synthesized from Qwen v3.1 teacher responses, corrected by a pure-Python 5-reviewer orchestrator.

**What this is NOT:** A coding model. The fine-tune **degraded** raw coding ability. HumanEval+ pass@1 dropped from 64.0% (base) to 17.1% (fine-tuned, raw completion) to 13.4% (fine-tuned, chat template). The 95.1% / 81.1% HumanEval scores in the benchmark section are the **base model** with execution-based multi-sample selection (50 samples, T=0.2+0.8), not this fine-tuned model. Those benchmarks establish the ceiling of the base model's coding ability and validate the evaluation methodology.

**Should you use this?** Probably not for coding. Use the base Qwen3.5-9B for code generation. This model is useful if you need a standalone model that emits structured tool calls without a Python routing layer. If you already have a routing layer (like MAOS, LangChain, etc.), the base model with a good system prompt is better.

**Training at a glance:**

| Stage | Steps | Time | Final Loss | Key Metric |
|-------|-------|------|------------|------------|
| SFT (LoRA r=64, bf16) | 1,887 | 105 min | 0.060 | 99.2% token accuracy |
| DPO (beta=0.1, 1 epoch) | 630 | 77 min | 3.9e-05 | 100% reward accuracy, margin ~43 |

**The DPO loss went to near-zero.** The chosen/rejected pairs were trivially separable -- the teacher narrated tools (950 chars avg) while the synthesized responses emitted structured calls (185 chars avg). The DPO signal is dominated by format differences, not reasoning quality. The model learned "don't narrate, emit structured calls" but did not learn deeper preference signals.

**The control token system works but has routing edge cases.** The model sometimes routes debugging tasks to `[RECOVER]` instead of `[THINK]` when the prompt contains failure-related keywords ("failing", "broken", "error"). Keyword sensitivity from the training data distribution.

**Coding SFT LoRA was attempted and abandoned.** A separate coding LoRA trained on MBPP + synthetic data (1,256 examples) hurt performance (57.3% vs 64% base). Overfitting on simple patterns. The base model was already strong; fine-tuning on narrow data degraded it.

**Infrastructure:** Trained on a rented RTX PRO 5000 Blackwell (48GB VRAM) via vast.ai. Total cost ~$3.50 ($2-3 Together AI inference + $0.30 GPU rental). bf16, no quantization, SDPA attention (flash-attn wouldn't build), no packing (cross-contamination risk without flash attention).

**Deployment formats:** bf16 GGUF (17GB), Q8_0 (8.9GB), Q6_K (6.9GB), Q5_K_M (6.0GB), Q5_K_S (5.9GB), Q4_K_M (5.2GB), Q4_K_S (5.0GB), Q4_0 (4.9GB), Q3_K_M (4.3GB), MLX 4-bit (4.7GB). fp8 is NOT supported by llama.cpp.

**Thinking mode gotcha:** Qwen3.5 has built-in thinking mode. You MUST pass `enable_thinking=False` in the chat template, or the model generates a thinking block before the control token. This is why the chat-template benchmark (13.4%) was worse than raw completion (17.1%) -- thinking tokens contaminated the code output.

---

## The Honest Assessment

This project succeeded at what it set out to do (teach control tokens + tool call format) and failed at what we hoped it would do (maintain coding ability while adding agentic behavior). Both outcomes are instructive.

### What we got right

- **The pipeline works.** Data generation (5K prompts -> teacher inference -> 5-reviewer orchestrator -> synthesis -> quality filter) produced clean training data in 3 seconds for 5,000 responses, at zero API cost.
- **The training converged.** SFT hit 99.2% token accuracy in 105 minutes. DPO hit 100% reward accuracy in 77 minutes. The format/routing behavior is easy to learn.
- **The cost was $3.50.** Together AI inference ($2-3) + vast.ai GPU rental ($0.30). No expensive API bills, no long training runs.
- **The benchmarking methodology is sound.** Execution-based multi-sample selection (50 samples, T=0.2+0.8) is the standard technique used by DeepSeek-Coder and CodeLlama. We validated it on the base model: 64% greedy -> 81.1% with execution selection.
- **The deployment pipeline works.** fp16 safetensors, MLX 4-bit, and 9 GGUF quantizations all produced and uploaded.

### What we got wrong

- **Fine-tuning degraded coding ability by 47 percentage points.** HumanEval+ dropped from 64.0% (base) to 17.1% (fine-tuned). The LoRA adapters optimized for format compliance at the expense of semantic code generation. This is the fundamental trade-off of fine-tuning on format data: you get format, you lose substance.
- **The DPO signal was format-only.** Cross-model DPO pairs (teacher narration vs synthesized structured calls) are trivially separable. The loss went to 3.9e-05. The model learned "don't narrate" but not "reason better." For deeper preference learning, self-preference pairs with multiple temperature samples would have been better.
- **The coding SFT LoRA overfit.** Training on 1,256 MBPP + synthetic examples hurt the base model (57.3% vs 64%). The base model's coding ability was already strong. Fine-tuning on narrow, simple data degraded generalization. Lesson: don't fine-tune a strong model on weak data.
- **The control token routing has keyword sensitivity.** The model routes debugging tasks to `[RECOVER]` instead of `[THINK]` when the prompt contains "failing", "broken", "error". The training data distribution biased the routing.
- **Chat template evaluation was misleading.** The thinking mode contamination (13.4% vs 17.1%) took time to diagnose. Should have tested raw completion first.
- **Flash attention never built.** Python 3.12 + CUDA 13.0 had no prebuilt wheel. Used SDPA with packing=False, accepting a 20-30% throughput hit.

### Lessons that matter more than the model

These lessons cost $3.50 and 2 days to learn. They're worth more than the model:

1. **Fine-tuning is a trade-off, not an upgrade.** If your base model is already good at something, fine-tuning on different data will degrade that ability. The LoRA adapters don't add capability -- they redirect capacity. Know what you're trading away.

2. **DPO is only as good as your pairs.** If your chosen/rejected pairs differ in format (not reasoning), the model learns format (not reasoning). Near-zero DPO loss is a red flag, not a success. It means the task was trivial.

3. **Execution-based benchmarking is the correct methodology.** Greedy pass@1 underestimates model ability. Multi-sample with execution selection (50 samples, T=0.2+0.8) gives the true ceiling. 28/164 problems had no passing sample at T=0.2; 15 of those were solved at T=0.8. High-temperature diversity is critical.

4. **Test the base model first.** We spent time trying to "recover" coding ability with a coding SFT LoRA, not realizing the base model was already strong (64% HumanEval+). If we'd benchmarked the base model first, we'd have known the ceiling and not wasted time on the coding LoRA.

5. **Template-based synthesis is deterministic but shallow.** The 5-reviewer orchestrator + template synthesis produced correctly formatted responses at 1,623 responses/sec for $0. But the improved responses were structurally correct and semantically simple. The base model provides the reasoning; the templates provide the format.

6. **$3.50 is enough for a 9B LoRA fine-tune.** Together AI inference ($2-3 for 5K responses) + vast.ai RTX PRO 5000 Blackwell ($0.30 for 3 hours). You don't need expensive infrastructure for experimentation.

---

## Model Details

| Field | Value |
|-------|-------|
| Base model | Qwen/Qwen3.5-9B |
| Parameters | 9B |
| Training method | LoRA SFT + DPO |
| LoRA rank | 64 (alpha 128, dropout 0.05) |
| Training precision | bf16 (no quantization during training) |
| Training data | 4,992 SFT examples + 5,000 DPO pairs |
| Training hardware | RTX PRO 5000 Blackwell (48GB VRAM) |
| Context length | 2048 tokens |
| Attention implementation | SDPA (PyTorch native) |
| Packing | Disabled (no flash attention available) |
| Total training time | ~3 hours (SFT 105 min + DPO 77 min) |
| Total cost | ~$3.50 |

## Control Token System

The model emits one of five control tokens at the start of every response:

| Token | Usage | Distribution in training data |
|-------|-------|-------------------------------|
| `[FAST]` | Direct action, simple tasks | 47.4% (2,366) |
| `[THINK]` | Multi-step reasoning before acting | 41.6% (2,078) |
| `[VERIFY]` | Act then confirm result (destructive/irreversible) | 2.9% (145) |
| `[RECOVER]` | Reassess after failure, try different strategy | 7.1% (354) |
| `[ESCALATE]` | Surface for human decision (high-risk/security) | 1.0% (49) |

## Tool Call Format

```
<tool_call>
{"name": "shell", "arguments": {"command": "uv run pytest"}}
</tool_call>```

Supported tools: `shell`, `file_read`, `file_write`, `file_edit`, `file_list`, `grep`, `find_file`, `git`, `web_search`, `web_fetch`, `todo_write`, `ask_user`.

## Usage

### MLX (Apple Silicon)

```python
import mlx_lm

model, tokenizer = mlx_lm.load("davidnichols-ops/adaptive-operator-v4-mlx-4bit")

SYSTEM = """You are an adaptive engineering operator. You inspect systems, reason through problems, take actions, verify results, recover from failures, and decide when deeper thinking is necessary. You work from evidence, not assumptions.

## Compute Modes (control tokens)
Begin every response with exactly one control token to signal how much reasoning the task needs:
  [FAST]     -- direct, low-latency action. Use for simple tool calls, file reads, status checks, and routine edits.
  [THINK]    -- extended reasoning before acting. Use for design, debugging, refactoring, multi-step planning, and ambiguous problems.
  [VERIFY]   -- act, then verify the result before declaring success. Use when a change must be confirmed (tests pass, file written, deploy succeeded).
  [RECOVER]  -- a previous attempt failed; reassess and try a different strategy. Use after errors, broken builds, or unexpected output.
  [ESCALATE] -- the task exceeds safe autonomous scope; stop and surface the situation for a human. Use for irreversible or high-risk actions.

## Tool Use
When a task requires a tool, emit a structured tool call in this exact format:
  <tool_call>
  {"name": "tool_name", "arguments": {"param": "value"}}
  </tool_call>

After each tool call you will receive the result. Use it to decide your next action. When the task is complete, give a concise final answer with no tool calls."""

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Show me the last 10 git commits."},
]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = mlx_lm.generate(model, tokenizer, prompt=chat, max_tokens=256)
print(response)
```

### HuggingFace Transformers (fp16)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("davidnichols-ops/adaptive-operator-v4", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("davidnichols-ops/adaptive-operator-v4")

# CRITICAL: disable thinking mode or control tokens won't appear at response start
chat = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
```

### Ollama (GGUF)

```bash
ollama create adaptive-operator-v4 -f Modelfile
ollama run adaptive-operator-v4 "Show me the last 10 git commits."
```

## GGUF Quantizations

All quantizations are in the [GGUF repo](https://huggingface.co/davidnichols-ops/adaptive-operator-v4-gguf), generated from the bf16 source with llama.cpp:

| Format | Size | Bits/weight | Best for |
|--------|------|-------------|----------|
| bf16 | 17 GB | 16.0 | Maximum fidelity, CPU inference |
| Q8_0 | 8.9 GB | 8.5 | Near-lossless, good speed/quality balance |
| Q6_K | 6.9 GB | 6.6 | High quality, moderate size |
| Q5_K_M | 6.0 GB | 5.8 | Good quality/size trade-off |
| Q5_K_S | 5.9 GB | 5.6 | Slightly smaller than Q5_K_M |
| Q4_K_M | 5.2 GB | 5.0 | Recommended default for 4-bit |
| Q4_K_S | 5.0 GB | 4.8 | Smaller 4-bit variant |
| Q4_0 | 4.9 GB | 4.7 | Fastest 4-bit, legacy format |
| Q3_K_M | 4.3 GB | 4.1 | Smallest viable, quality degradation |

**fp8 is not supported** by llama.cpp. The highest-fidelity 16-bit option is bf16.

## Training Pipeline

```
Prompt Generator (5K unique prompts across 10 categories)
    -> Qwen v3.1 Teacher Inference (Together AI, 32 parallel workers, ~$2-3)
    -> 5-Reviewer Orchestrator (pure Python heuristics, 1,623 responses/sec)
       - Code Quality & Correctness
       - Tool Selection & Usage
       - Control Token Routing
       - Error Handling & Edge Cases
       - Response Format & Clarity
    -> Data Quality Filter (dedup + n-gram decontamination)
    -> SFT Export + DPO Pair Generation
    -> LoRA SFT Training (bf16, 3 epochs, lr=2e-4, r=64)
    -> LoRA DPO Training (bf16, 1 epoch, lr=5e-5, beta=0.1)
    -> Merge LoRA -> fp16 model
    -> Quantization (MLX 4-bit + 9 GGUF variants)
```

## Training Metrics

### SFT (3 epochs, 1,887 steps, 105 min)

| Metric | Value |
|--------|-------|
| Final train loss | 0.060 |
| Final token accuracy | 99.2% |
| Convergence | 2.09 -> 0.08 in 110 steps (6% of training) |
| Throughput | 2.39 samples/sec, 0.30 steps/sec |
| Total tokens | 5.9M |

### DPO (1 epoch, 630 steps, 77 min)

| Metric | Value |
|--------|-------|
| Final train loss | 3.9e-05 (near-zero -- see lessons above) |
| Reward accuracy | 100% |
| Reward margin | ~43 (chosen +8.5, rejected -35.5) |
| Throughput | 1.09 samples/sec, 0.14 steps/sec |

## Benchmark Results

### HumanEval / HumanEval+ (EvalPlus) -- Base Qwen3.5-9B

These benchmarks were run on the **base model** to establish the coding ceiling and validate the evaluation methodology. The fine-tuned model's coding ability is lower (see below).

| Method | HumanEval pass@1 | HumanEval+ pass@1 |
|--------|:-:|:-:|
| Greedy (temp=0) | 70.7% | 64.0% |
| 20 samples (T=0.2), random pick | 69.9% | 62.1% |
| 20 samples, execution-selected (base tests) | 82.9% | 70.7% |
| 20 samples, execution-selected (base+plus tests) | 82.9% | 72.6% |
| **50 samples (T=0.2+0.8), execution-selected** | **95.1%** | **81.1%** |
| pass@10 (20 samples, T=0.2) | 81.5% | 74.6% |

### Fine-tuned Model -- Coding Ability Impact

| Model | HumanEval+ pass@1 (raw completion) |
|-------|:-:|
| Base Qwen3.5-9B | 64.0% |
| Fine-tuned (chat template) | 13.4% |
| Fine-tuned (raw completion) | 17.1% |
| Coding SFT LoRA merged | 57.3% (abandoned -- overfit) |

### Fine-tuned Model -- Agentic Metrics

| Metric | Value |
|--------|-------|
| SFT final loss | 0.060 |
| DPO reward accuracy | 100% |
| Control token routing accuracy | 86% FAST mode on simple tasks |
| Tool call format compliance | >99% |

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

Cross-model preference pairs: original Qwen v3.1 teacher response (rejected) vs synthesized improved response (chosen). The rejected responses are 5x longer (950 chars vs 185 chars) -- the teacher was verbose with narration. The improved responses are concise and structured.

## Repository Contents

| Path | Description |
|------|-------------|
| `model.safetensors` | Merged fp16 model (17GB) |
| `sft_adapter/` | SFT LoRA adapter (465MB) |
| `dpo_adapter/` | DPO LoRA adapter (908MB) |
| `config.json` | Model configuration |
| `tokenizer.json` | Tokenizer |
| `chat_template.jinja` | Chat template |
| `TRAINING_REPORT.md` | Full training report with 14 issues encountered |
| `PIPELINE_ARCHITECTURE.md` | Pipeline architecture reference |
| `benchmark/` | Benchmark scripts (multi-sample, execution selection, etc.) |
| `training/` | Training scripts (SFT, DPO) |

## Related Repositories

| Repository | Description |
|------------|-------------|
| [adaptive-operator-v4-mlx-4bit](https://huggingface.co/davidnichols-ops/adaptive-operator-v4-mlx-4bit) | MLX 4-bit quantized for Apple Silicon (4.7GB) |
| [adaptive-operator-v4-gguf](https://huggingface.co/davidnichols-ops/adaptive-operator-v4-gguf) | 9 GGUF quantizations (bf16, Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q4_0, Q3_K_M) |
| [adaptive-operator-v4-dataset](https://huggingface.co/datasets/davidnichols-ops/adaptive-operator-v4-dataset) | SFT + DPO training data |

## Training Cost

| Resource | Usage | Cost |
|----------|-------|------|
| Together AI inference (5K responses) | 5.4M tokens | ~$2-3 |
| Vast.ai RTX PRO 5000 Blackwell | ~3 hours | ~$0.30 |
| **Total** | | **~$3.50** |

## Important: Thinking Mode

Qwen3.5 has a built-in thinking mode that prepends internal reasoning before the response. To get control tokens at the start of the output, you **must** pass `enable_thinking=False` when applying the chat template:

```python
chat = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
```

Without this, the model generates a thinking block first, and the control token appears later in the response.

## Limitations

- **Coding ability degradation**: The fine-tune degraded raw coding ability (64% -> 17% HumanEval+). Use the base Qwen3.5-9B for coding tasks.
- **Context length**: 2048 tokens (may truncate long conversations)
- **Tool call accuracy**: The model emits correctly formatted tool calls but may select suboptimal tools for ambiguous tasks
- **Reasoning depth**: SFT responses are structurally correct but semantically simple -- the base model provides semantic content, but the synthesis templates produce minimal stubs
- **Repetition**: 4-bit quantization with greedy decoding can cause repetition loops; use temperature sampling (0.7-0.8)
- **System prompt required**: The control token system only activates when the system prompt is provided
- **Thinking mode**: Must disable `enable_thinking` in the chat template
- **THINK routing**: The model sometimes routes debugging tasks to `[RECOVER]` instead of `[THINK]` when the prompt contains failure-related keywords
- **DPO overfitting**: Near-zero DPO loss indicates format-dominated signal, not reasoning quality
- **Training data bias**: 47.5% of training data is difficulty level 1 (simple). The model may underperform on complex, multi-step tasks

## Training Configuration

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

## License

Apache 2.0 (derived from Qwen3.5-9B)

## Citation

```bibtex
@misc{adaptive-operator-v4,
  title={Adaptive Operator v4: Qwen3.5-9B with Control Token Routing},
  author={David Nichols},
  year={2025},
  url={https://huggingface.co/davidnichols-ops/adaptive-operator-v4}
}
```
