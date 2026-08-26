# Adaptive Operator v3 — Evaluation Harness

Evaluation harness for the MAOS Adaptive Operator v3 (the think-act-observe-verify
agent loop). This harness runs a suite of tasks through the real `AgentLoop`
and records structured results for analysis.

## Structure

```
eval/operator_v3/
├── tasks/              # Evaluation task definitions (YAML/JSON)
│   ├── tool_competence.yaml
│   ├── recovery.yaml
│   ├── planning.yaml
│   └── compute_allocation.yaml
├── results/            # Evaluation results (JSON reports)
├── eval_harness.py     # The harness — loads tasks, runs the loop, verifies
└── run_eval.py         # CLI runner
```

## Task categories

| Category | What it measures |
| --- | --- |
| `tool_competence` | Can the operator select and call the right tool? |
| `recovery` | Can the operator recover from errors (bad path, unknown tool)? |
| `planning` | Can the operator sequence multiple steps and choose an approach? |
| `compute_allocation` | Does the operator allocate compute correctly (FAST vs THINK)? |

## Task schema

Each task file defines a `category` and a list of `tasks`. Each task:

```yaml
- name: <unique task name>
  prompt: "<the user prompt fed to the agent loop>"
  expected_behavior: "<human-readable description of correct behavior>"
  # Verification fields (all optional):
  verify_tool_used: ["file_list", "shell"]   # at least one of these must be called
  verify_min_tool_calls: 2                    # minimum total tool calls
  verify_no_tools: true                       # no tools should be called
  verify_recovery: true                       # an error must occur then the loop continues
```

## Usage

```bash
# Run all tasks with the default model
uv run python -m eval.operator_v3.run_eval

# Specify a model and output location
uv run python -m eval.operator_v3.run_eval \
    --model mlx://qwen3-14b \
    --tasks-dir eval/operator_v3/tasks \
    --output eval/operator_v3/results/report.json

# Run a single category
uv run python -m eval.operator_v3.run_eval --category tool_competence
```

Results are written as JSON to `eval/operator_v3/results/` with per-task
outcomes (success, tool calls, steps, duration, trajectory) and an aggregate
summary.
