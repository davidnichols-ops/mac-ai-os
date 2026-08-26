"""Helper functions for the QLoRA Colab training notebook.

This module is imported by ``colab_qlora_training.ipynb`` so that the data
loading, formatting, and evaluation logic can be version-controlled alongside
the rest of the v4 pipeline rather than living only inside notebook cells.

The functions here are designed to run inside a Google Colab environment
(transformers / trl / peft / bitsandbytes installed) but they have no hard
dependency on Colab itself -- they can be unit-tested locally with mocked
objects.

Data formats (matching the rest of the v4 pipeline):

  SFT  (sft_train.jsonl)   -- one JSON object per line:
      {"messages": [
          {"role": "system",    "content": "..."},
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "..."},
      ]}

  DPO  (dpo_train.jsonl)   -- one JSON object per line:
      {"input": {"messages": [
          {"role": "system", "content": "..."},
          {"role": "user",   "content": "..."},
      ]},
       "preferred_output":     [{"role": "assistant", "content": "..."}],
       "non_preferred_output": [{"role": "assistant", "content": "..."}]}

Control tokens (the v4 operator protocol):
    [FAST] [THINK] [VERIFY] [RECOVER] [ESCALATE]

Tool calls use the format:
   <tool_call>
  {"name": "...", "arguments": {...}}
   </tool_call>

After each tool call the operator receives the result and decides the next
action.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_TOKENS: tuple[str, ...] = (
    "[FAST]",
    "[THINK]",
    "[VERIFY]",
    "[RECOVER]",
    "[ESCALATE]",
)

# Matches <tool_call> ... </tool_call> blocks (non-greedy, DOTALL).
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# QLoRA target modules for Qwen3.5 -- all linear projections.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping blank/bad lines."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[colab_utils] skipping malformed JSON at {p}:{lineno}")


def count_jsonl(path: str | Path) -> int:
    """Count non-empty lines in a JSONL file."""
    n = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# SFT formatting
# ---------------------------------------------------------------------------


def format_sft_example(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize an SFT record into the {"messages": [...]} dict expected by
    ``SFTTrainer`` when using a chat template.

    Accepts records that already have a ``messages`` key, or records with
    flat ``system`` / ``user`` / ``assistant`` string fields.
    """
    if "messages" in record:
        return {"messages": record["messages"]}
    msgs: list[dict[str, str]] = []
    if record.get("system"):
        msgs.append({"role": "system", "content": record["system"]})
    msgs.append({"role": "user", "content": record["user"]})
    msgs.append({"role": "assistant", "content": record["assistant"]})
    return {"messages": msgs}


def load_sft_dataset(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load an SFT JSONL file and return a list of {"messages": [...]} dicts.

    If ``limit`` is set, only the first N examples are returned (handy for a
    quick smoke run on Colab).
    """
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        out.append(format_sft_example(rec))
    return out


# ---------------------------------------------------------------------------
# DPO formatting
# ---------------------------------------------------------------------------


def format_dpo_example(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a DPO record into the {"prompt", "chosen", "rejected"} dict
    expected by ``DPOTrainer`` when using conversational data.

    The v4 pipeline stores DPO data as:
        {"input": {"messages": [...]},
         "preferred_output": [...], "non_preferred_output": [...]}

    DPOTrainer (trl) accepts either:
      - {"prompt": [...], "chosen": [...], "rejected": [...]}  (list-of-turns)
      - {"prompt": str, "chosen": str, "rejected": str}        (raw strings)

    We return the list-of-turns (conversational) form so the chat template is
    applied consistently.
    """
    if "input" in record:
        prompt = record["input"]["messages"]
        chosen = record.get("preferred_output") or record.get("chosen") or []
        rejected = record.get("non_preferred_output") or record.get("rejected") or []
        return {
            "prompt": prompt,
            "chosen": chosen if isinstance(chosen, list) else [{"role": "assistant", "content": chosen}],
            "rejected": rejected if isinstance(rejected, list) else [{"role": "assistant", "content": rejected}],
        }
    # Already in prompt/chosen/rejected form
    return {
        "prompt": record["prompt"],
        "chosen": record["chosen"],
        "rejected": record["rejected"],
    }


def load_dpo_dataset(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load a DPO JSONL file and return a list of normalized preference dicts."""
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        out.append(format_dpo_example(rec))
    return out


# ---------------------------------------------------------------------------
# Evaluation / smoke-test helpers
# ---------------------------------------------------------------------------


def first_control_token(text: str) -> str | None:
    """Return the first control token (e.g. ``[FAST]``) found in ``text``,
    or ``None`` if the response is missing a control token."""
    for tok in CONTROL_TOKENS:
        if tok in text:
            return tok
    return None


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all <tool_call> ... </tool_call> blocks from ``text`` and
    parse their JSON payload. Returns a list of dicts; unparseable blocks are
    skipped with a warning."""
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            print(f"[colab_utils] skipping unparseable tool_call: {raw[:80]!r}")
    return calls


def evaluate_response(
    text: str,
    *,
    require_control_token: bool = True,
    require_tool_call: bool = False,
) -> dict[str, Any]:
    """Run a quick structural evaluation of a generated response.

    Returns a dict with:
      - ``control_token``: the first control token found (or None)
      - ``has_control_token``: bool
      - ``tool_calls``: list of parsed tool call dicts
      - ``has_tool_call``: bool
      - ``passed``: True if all required checks pass
      - ``issues``: list of human-readable issue strings
    """
    issues: list[str] = []
    tok = first_control_token(text)
    has_tok = tok is not None
    if require_control_token and not has_tok:
        issues.append(
            "missing control token (expected one of " + ", ".join(CONTROL_TOKENS) + ")"
        )

    calls = extract_tool_calls(text)
    has_call = len(calls) > 0
    if require_tool_call and not has_call:
        issues.append("expected at least one <tool_call> block but found none")

    return {
        "control_token": tok,
        "has_control_token": has_tok,
        "tool_calls": calls,
        "has_tool_call": has_call,
        "passed": len(issues) == 0,
        "issues": issues,
    }


def smoke_test_prompt() -> list[dict[str, str]]:
    """Return a simple system + user prompt pair for the notebook smoke test."""
    return [
        {
            "role": "system",
            "content": (
                "You are an adaptive engineering operator. You inspect systems, "
                "reason through problems, take actions, verify results, recover "
                "from failures, and decide when deeper thinking is necessary. "
                "You work from evidence, not assumptions.\n\n"
                "## Compute Modes (control tokens)\n"
                "Begin every response with exactly one control token:\n"
                "  [FAST] [THINK] [VERIFY] [RECOVER] [ESCALATE]\n\n"
                "## Tool Use\n"
                "When a task requires a tool, emit a structured tool call:\n"
                "  <tool_call>\n"
                "{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
                "  </tool_call>\n\n"
                "After each tool call you will receive the result. Use it to "
                "decide your next action. When the task is complete, give a "
                "concise final answer with no tool calls."
            ),
        },
        {
            "role": "user",
            "content": "Show the last 10 commits in the repository.",
        },
    ]


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def print_dataset_summary(name: str, examples: Iterable[dict[str, Any]]) -> None:
    """Print a short summary of a loaded dataset (count + first example keys)."""
    examples = list(examples)
    print(f"[{name}] {len(examples)} examples")
    if examples:
        print(f"[{name}] first example keys: {list(examples[0].keys())}")


def get_lora_target_modules() -> list[str]:
    """Return the list of LoRA target module names for Qwen3.5."""
    return list(LORA_TARGET_MODULES)


__all__ = [
    "CONTROL_TOKENS",
    "LORA_TARGET_MODULES",
    "count_jsonl",
    "evaluate_response",
    "extract_tool_calls",
    "first_control_token",
    "format_dpo_example",
    "format_sft_example",
    "get_lora_target_modules",
    "load_dpo_dataset",
    "load_sft_dataset",
    "print_dataset_summary",
    "read_jsonl",
    "smoke_test_prompt",
]
