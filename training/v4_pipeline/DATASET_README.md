---
language: en
license: apache-2.0
task_categories:
  - text-generation
  - text-classification
  - reinforcement-learning-from-human-feedback
tags:
  - agent
  - tool-use
  - control-tokens
  - sft
  - dpo
  - code
  - instruction-tuning
size_categories:
  - 1K<n<10K
---

# Adaptive Operator v4 — Training Datasets

Training data for [adaptive-operator-v4](https://huggingface.co/davidnichols-ops/adaptive-operator-v4), a Qwen3.5-9B model fine-tuned with custom control tokens for adaptive compute allocation in agentic workflows.

## Dataset Summary

| Split | Examples | Format | Size |
|-------|----------|--------|------|
| SFT | 4,992 | OpenAI chat messages | 8.8 MB |
| DPO | 5,000 | Chosen/rejected pairs | 8.3 MB |
| Raw teacher responses | 5,000 | OpenAI chat messages | 6.5 MB |
| Improved responses | 4,992 | OpenAI chat messages | 13 MB |
| Preference pairs (raw) | 5,000 | Prompt + chosen + rejected | 6.6 MB |

## Files

| File | Description |
|------|-------------|
| `sft_train.jsonl` | SFT training data in OpenAI chat format (system + user + assistant) |
| `dpo_train.jsonl` | DPO training data in TRL format (prompt + chosen + rejected) |
| `raw_teacher_responses.jsonl` | Original Qwen v3.1 teacher responses (before improvement) |
| `improved_responses.jsonl` | Synthesized improved responses (after 5-reviewer orchestrator) |
| `preference_pairs.jsonl` | Raw DPO preference pairs before TRL formatting |

## Data Format

### SFT (`sft_train.jsonl`)

```json
{
  "messages": [
    {"role": "system", "content": "You are an adaptive engineering operator..."},
    {"role": "user", "content": "Show which branches are not yet pushed to the remote"},
    {"role": "assistant", "content": "[FAST]\nDirect action.\n\n<tool_call>\n{\"name\": \"git\", \"arguments\": {\"command\": \"branch -r\"}}\n  </tool_call>"}
  ]
}
```

### DPO (`dpo_train.jsonl`)

```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "[FAST]\nDirect action.\n\n<tool_call>\n..."}],
  "rejected": [{"role": "assistant", "content": "I'll help you with that. Let me think about..."}]
}
```

## Control Token Distribution (SFT)

| Token | Count | Percentage | Usage |
|-------|-------|------------|-------|
| `[FAST]` | 2,366 | 47.4% | Direct action, simple tasks |
| `[THINK]` | 2,078 | 41.6% | Multi-step reasoning |
| `[RECOVER]` | 354 | 7.1% | Reassess after failure |
| `[VERIFY]` | 145 | 2.9% | Act then confirm |
| `[ESCALATE]` | 49 | 1.0% | Surface for human |

## Task Categories (SFT)

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

## Pipeline

1. **Prompt generation**: 5,000 unique prompts across 10 categories (file ops, coding, debugging, testing, refactoring, planning, search, git, shell, recovery)
2. **Teacher inference**: Qwen v3.1 via Together AI (32 parallel workers)
3. **5-reviewer orchestrator**: Pure Python heuristics scoring tool selection, format clarity, control tokens, code quality, error handling (1,623 responses/sec)
4. **Synthesis**: Template-based improvement fixing format issues (99.9% narration -> structured tool calls, 48.4% wrong control tokens corrected)
5. **Quality filtering**: Dedup (8 fuzzy duplicates removed), n-gram decontamination, length filtering
6. **DPO pair generation**: Cross-model pairs (teacher = rejected, synthesized = chosen)

## Known Characteristics

- **Rejected responses are 5x longer than chosen** (950 chars vs 185 chars) — the teacher was verbose with narration
- **Improved responses are structurally correct but semantically simple** — templates produce correct format, base model provides reasoning
- **DPO signal is dominated by format differences** (narration vs structured calls), not reasoning quality

## License

Apache 2.0 (training data generated from Qwen v3.1 teacher, derived from Qwen3.5-9B base model)

## Related

- [Model (fp16 + LoRA adapters)](https://huggingface.co/davidnichols-ops/adaptive-operator-v4)
- [MLX 4-bit model](https://huggingface.co/davidnichols-ops/adaptive-operator-v4-mlx-4bit)
