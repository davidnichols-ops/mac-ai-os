# Training Data Generation — Trajectories

## Overview

Every `maos do` execution produces a structured trajectory. This is the atomic
unit of training data for future Adaptive Operator models. The trajectory captures
the complete interaction: user intent, model decisions, tool calls, results,
errors, recovery, and final outcome.

## Schema

Version: `2.0.0`

```json
{
  "trajectory_id": "traj_abc123",
  "schema_version": "2.0.0",
  "task": "list the files in the current directory",
  "task_context": "...",
  "model_name": "Qwen3.5-9B-Adaptive-Operator-v2-MLX-4bit",
  "model_provider": "mlxvlm",
  "steps": [
    {
      "step_id": 0,
      "step_type": "model_output",
      "content": "I'll list the files...",
      "compute_mode": "fast",
      "generation_time_s": 33.6,
      "completion_tokens": 27,
      "tokens_per_second": 0.83
    },
    {
      "step_id": 1,
      "step_type": "tool_call",
      "tool_name": "file_list",
      "tool_args": {"path": "."},
      "risk_level": "LOW"
    },
    {
      "step_id": 2,
      "step_type": "tool_result",
      "tool_result": ".DS_Store\n.git/\n...",
      "tool_ok": true
    },
    {
      "step_id": 3,
      "step_type": "model_output",
      "content": "Here are the files...",
      "generation_time_s": 72.6,
      "completion_tokens": 152
    }
  ],
  "success": true,
  "failure_category": "none",
  "final_answer": "Here are the files...",
  "total_steps": 4,
  "total_tool_calls": 1,
  "total_tokens": 179,
  "time_to_completion_s": 111.2,
  "fast_steps": 0,
  "think_steps": 0,
  "safety_violations": 0,
  "approvals_requested": 0
}
```

## Step Types

| Type | Description |
|------|-------------|
| `observation` | Initial environment state |
| `model_output` | Model generated text (with inference metrics) |
| `tool_call` | Model requested a tool execution |
| `tool_result` | Tool execution result |
| `error` | An error occurred |
| `verification` | Outcome verification |
| `recovery` | Recovery attempt after failure |
| `termination` | Loop ended (success or failure) |

## Failure Categories

| Category | Description |
|----------|-------------|
| `none` | Didn't fail |
| `wrong_tool` | Called the wrong tool |
| `wrong_args` | Right tool, wrong arguments |
| `wrong_hypothesis` | Correct tool use, incorrect reasoning |
| `missed_evidence` | Didn't inspect available information |
| `incomplete` | Didn't finish the task |
| `premature` | Declared finish before verifying |
| `infinite_loop` | Repeated same action >3 times |
| `step_budget` | Exceeded max steps |
| `safety` | Crossed a hard safety limit |
| `env_error` | Environment itself broke |
| `model_error` | Model generation failed |

## Trajectory Classes (for filtering)

| Class | Description |
|-------|-------------|
| A | Correct and efficient (gold) |
| B | Correct but inefficient |
| C | Failed but educational (recovery training) |
| D | Incorrect (normally discard) |
| E | Unsafe (always discard) |
| F | Ambiguous (manual review) |

## Storage

Trajectories are persisted as JSON to `data/trajectories/traj_*.json`.
Each file is self-contained and can be used directly for training.

## Future: Distillation Pipeline

The local-operator project has a complete distillation pipeline that:
1. Generates trajectories (teacher model)
2. Classifies them (quality scoring)
3. Exports to JSONL (training format)
4. Fine-tunes student models (Together AI LoRA)

MAOS trajectories use the same schema philosophy and can feed this pipeline.
