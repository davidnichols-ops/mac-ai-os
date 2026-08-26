---
language: en
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags:
  - mlx
  - mlx-lm
  - 4-bit
  - quantized
  - agent
  - tool-use
  - control-tokens
  - adaptive-compute
pipeline_tag: text-generation
---

# Adaptive Operator v4 — MLX 4-bit

This is the MLX 4-bit quantized version of [adaptive-operator-v4](https://huggingface.co/davidnichols-ops/adaptive-operator-v4), optimized for Apple Silicon inference.

| Field | Value |
|-------|-------|
| Base model | Qwen/Qwen3.5-9B |
| Quantization | 4-bit (group size 64) |
| File size | 4.7 GB |
| Load time (M2 Pro) | ~2.6s |
| Framework | mlx-lm 0.31.3+ |

## Usage

```python
import mlx_lm

model, tokenizer = mlx_lm.load("davidnichols-ops/adaptive-operator-v4-mlx-4bit")

SYSTEM = """You are an adaptive engineering operator. You inspect systems, reason through problems, take actions, verify results, recover from failures, and decide when deeper thinking is necessary. You work from evidence, not assumptions. You document failures as carefully as successes.

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
    {"role": "user", "content": "List files in the current directory."},
]

# IMPORTANT: Disable thinking mode so control tokens appear at the start
chat = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
response = mlx_lm.generate(model, tokenizer, prompt=chat, max_tokens=256)
print(response)
```

## Important: Thinking Mode

Qwen3.5 has a built-in thinking mode that prepends internal reasoning before the response. To get control tokens at the start of the output, you **must** pass `enable_thinking=False` when applying the chat template. Otherwise the model generates a thinking block first, and the control token appears later in the response.

## Control Tokens

| Token | Usage |
|-------|-------|
| `[FAST]` | Direct action, simple tasks |
| `[THINK]` | Multi-step reasoning before acting |
| `[VERIFY]` | Act then confirm result |
| `[RECOVER]` | Reassess after failure |
| `[ESCALATE]` | Surface for human decision |

## Related

- [Full fp16 model + LoRA adapters](https://huggingface.co/davidnichols-ops/adaptive-operator-v4)
- [Training datasets](https://huggingface.co/datasets/davidnichols-ops/adaptive-operator-v4-dataset)

## License

Apache 2.0 (derived from Qwen3.5-9B)
