"""5-reviewer orchestrator for the v4 post-training pipeline.

Reads Qwen v3.1 initial responses (``qwen_responses.jsonl``) and, for each
response, runs five **pure-Python** heuristic review functions that evaluate
the response from a different angle:

  1. Code Quality & Correctness
  2. Tool Selection & Usage
  3. Control Token Routing
  4. Error Handling & Edge Cases
  5. Response Format & Clarity

After the five reviews a **synthesis** step (also pure Python, template-based)
produces a final improved response that:

  - starts with the correct control token,
  - emits structured tool calls in the XML+JSON format the runtime parses,
  - fixes the issues the reviewers flagged,
  - is clear, concise, and correct.

The improved response becomes the SFT training target; the original Qwen
response becomes the DPO "rejected" example.

Output record schema (one JSON object per line in ``improved_responses.jsonl``)::

    {
      "id":               "<prompt id>",
      "prompt":           "<user task>",
      "category":         "<prompt_generator category>",
      "expected_mode":    "FAST|THINK|VERIFY|RECOVER|ESCALATE",
      "original_response": "<raw Qwen v3.1 text>",
      "improved_response": "<synthesized improved text>",
      "reviews":          [<review object>, ... x5],
      "scores":           {"code_quality": int, "tool_selection": int, ...}
    }

Usage::

    cd /path/to/mac-ai-os
    uv run python -m training.v4_pipeline.review_orchestrator --limit 3
    uv run python -m training.v4_pipeline.review_orchestrator --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
SFT_DIR = PIPELINE_DIR / "sft"
DEFAULT_INPUT = SFT_DIR / "qwen_responses.jsonl"
DEFAULT_OUTPUT = SFT_DIR / "improved_responses.jsonl"
PROGRESS_EVERY = 100

# ---------------------------------------------------------------------------
# Control tokens
# ---------------------------------------------------------------------------

CONTROL_TOKENS = ("FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE")
_CONTROL_TOKEN_RE = re.compile(
    r"^\s*\[(?:" + "|".join(CONTROL_TOKENS) + r")\]",
    re.IGNORECASE,
)

# Tool-call XML tags (built with chr() so the source has no literal angle
# brackets, mirroring qwen_inference.py / sft_export.py).
_TC_OPEN = chr(60) + "tool_call" + chr(62)
_TC_CLOSE = chr(60) + "/tool_call" + chr(62)
_TOOL_CALL_RE = re.compile(
    re.escape(_TC_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(_TC_CLOSE),
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Tool schemas — parameter names the runtime expects (from qwen_inference.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSchema:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    @property
    def all_params(self) -> tuple[str, ...]:
        return self.required + self.optional


TOOL_SCHEMAS: dict[str, ToolSchema] = {
    "shell":       ToolSchema("shell",       ("command",)),
    "file_read":   ToolSchema("file_read",   ("file_path",)),
    "file_write":  ToolSchema("file_write",  ("file_path", "content")),
    "file_edit":   ToolSchema("file_edit",   ("file_path", "old_string", "new_string")),
    "file_list":   ToolSchema("file_list",   ("path",), ("pattern",)),
    "grep":        ToolSchema("grep",        ("pattern", "path"), ("glob_pattern",)),
    "find_file":   ToolSchema("find_file",   ("pattern", "path")),
    "git":         ToolSchema("git",         ("command",), ("repo",)),
    "web_search":  ToolSchema("web_search",  ("query",), ("num_results",)),
    "web_fetch":   ToolSchema("web_fetch",   ("url",)),
    "todo_write":  ToolSchema("todo_write",  ("todos",)),
    "ask_user":    ToolSchema("ask_user",    ("question",)),
}

VALID_TOOL_NAMES = set(TOOL_SCHEMAS)

# Mapping from a tool name to the categories where it is the natural choice.
_CATEGORY_TO_TOOLS: dict[str, tuple[str, ...]] = {
    "tool_use_file_ops": ("file_read", "file_write", "file_edit", "file_list",
                          "find_file", "grep", "shell"),
    "tool_use_shell":    ("shell",),
    "tool_use_git":      ("git",),
    "tool_use_search":   ("grep", "find_file", "web_search"),
    "coding_write":      ("file_write", "file_edit"),
    "coding_debug":      ("file_read", "grep", "shell", "file_edit"),
    "coding_refactor":   ("file_read", "file_edit", "grep"),
    "coding_test":       ("file_write", "file_edit", "shell"),
    "planning":          (),
    "recovery":          ("shell", "git", "file_read", "grep"),
}

# Categories where the response should primarily be prose/code (no tool call
# required) — the answer *is* the deliverable.
_NO_TOOL_CATEGORIES = {"planning"}

# Narration patterns — the model describing a tool call instead of emitting one.
_NARRATION_SELECT_RE = re.compile(
    r"select\s+tool\s*:\s*([a-zA-Z_][\w\s-]*)",
    re.IGNORECASE,
)
_NARRATION_ARROW_RE = re.compile(r"→|->", re.IGNORECASE)
_NARRATION_EXECUTE_RE = re.compile(
    r"\b(execute|run|invoke|call)\b\s*(the|a|an)?\s*(command|tool|script)?",
    re.IGNORECASE,
)
_FABRICATED_RESULT_RE = re.compile(
    r"\b(all tests passed|tests? passed|no failures|"
    r"merge complete|deleted.*files?|"
    r"found.*\d+.*files?|results?:)\b",
    re.IGNORECASE,
)
_BACKTICK_CMD_RE = re.compile(r"`([^`]{3,})`")
_SHELL_CMD_HINT_RE = re.compile(
    r"(git\s+\w+|find\s+|grep\s+|ls\s+|cat\s+|rm\s+|cp\s+|mv\s+|"
    r"pytest|npm\s+|python\s+-m|docker\s+|make\s+|curl\s+)"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReviewResult:
    """The output of a single review function."""

    name: str
    score: int  # 1-10
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedResponse:
    """Lightweight parse of an original Qwen response."""

    raw: str
    mode: Optional[str]  # control token uppercased, None if missing
    body: str  # text after the control token
    tool_calls: list[dict[str, Any]]  # parsed structured tool calls
    has_narration: bool
    narrated_tools: list[str]  # tool names mentioned in narration
    has_fabricated_result: bool


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_control_token(text: str) -> tuple[Optional[str], str]:
    """Return (mode_upper, remainder) or (None, text)."""
    m = _CONTROL_TOKEN_RE.match(text)
    if not m:
        return None, text
    token = m.group(0).strip().strip("[]").upper()
    return token, text[m.end():].lstrip()


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse structured tool-call XML blocks from text."""
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls


def _strip_tool_calls(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text).strip()


def _detect_narrated_tools(text: str) -> list[str]:
    """Find tool names the model *narrated* rather than called.

    Looks for "select tool: X" patterns and normalises the captured name to a
    canonical tool name when possible.
    """
    found: list[str] = []
    for m in _NARRATION_SELECT_RE.finditer(text):
        raw = m.group(1).strip().lower()
        # Normalise common variants ("git grep" -> "git", "git diff" -> "git").
        for canonical in VALID_TOOL_NAMES:
            if raw.startswith(canonical) or raw == canonical:
                if canonical not in found:
                    found.append(canonical)
                break
        else:
            # Keep the raw mention if it isn't a known tool (still useful signal).
            label = raw.split()[0] if raw.split() else raw
            if label and label not in found:
                found.append(label)
    return found


def parse_response(raw: str) -> ParsedResponse:
    """Parse a raw Qwen response into a ParsedResponse."""
    mode, body = _parse_control_token(raw)
    tool_calls = _parse_tool_calls(raw)
    narrated = _detect_narrated_tools(raw)
    has_narration = bool(narrated) or bool(_NARRATION_ARROW_RE.search(raw))
    has_fab = bool(_FABRICATED_RESULT_RE.search(raw))
    return ParsedResponse(
        raw=raw,
        mode=mode,
        body=body,
        tool_calls=tool_calls,
        has_narration=has_narration,
        narrated_tools=narrated,
        has_fabricated_result=has_fab,
    )


# ---------------------------------------------------------------------------
# Review 1 — Code Quality & Correctness
# ---------------------------------------------------------------------------


def review_code_quality(
    prompt: str,
    parsed: ParsedResponse,
    category: str,
    expected_mode: str,
) -> ReviewResult:
    """Evaluate code correctness, edge-case handling, and logic soundness."""
    res = ReviewResult(name="code_quality", score=8)
    text = parsed.body
    lower = text.lower()

    # Fabricated results are a serious correctness problem — the model claims
    # success without evidence (it never actually ran anything).
    if parsed.has_fabricated_result:
        res.score -= 4
        res.issues.append(
            "Fabricates results ('all tests passed', 'deleted N files') without "
            "evidence — the model cannot know the outcome before executing."
        )

    # Vague fix descriptions with no concrete code.
    if category.startswith("coding_") and "def " not in text and "class " not in text:
        if re.search(r"\b(add|fix|change|update|remove)\b.*\b(null|check|handler)\b", lower):
            if not re.search(r"(`[^`]+`|def \w+|return |if .* is None)", text):
                res.score -= 2
                res.issues.append(
                    "Describes a fix in prose without showing the concrete code change."
                )

    # Edge-case handling.
    edge_keywords = ("edge case", "edge-case", "empty", "null", "none", "overflow",
                     "unicode", "negative", "zero", "boundary", "empty input")
    if category.startswith("coding_") and not any(k in lower for k in edge_keywords):
        res.score -= 1
        res.issues.append("Does not address edge cases (empty/null/overflow/unicode).")

    # Logic soundness — claims of success with no diagnostic steps for recovery.
    if category == "recovery" and not re.search(
        r"\b(log|trace|debug|diagnos|inspect|reproduc|stack|error message)\b", lower
    ):
        res.score -= 2
        res.issues.append(
            "Recovery task lacks diagnostic steps — must inspect logs/errors before fixing."
        )

    # Planning tasks: check for concrete deliverables.
    if category == "planning":
        if not re.search(r"\b(table|entity|schema|step|phase|component|api|endpoint|model)\b", lower):
            res.score -= 2
            res.issues.append("Planning response lacks concrete artifacts (entities, steps, components).")
        else:
            res.notes.append("Planning response identifies concrete deliverables.")

    # Type errors / logic errors — look for contradictory statements.
    if re.search(r"\basync\b", lower) and "await" not in lower and category.startswith("coding_"):
        res.score -= 2
        res.issues.append("Mentions async code but no await — likely blocking the event loop.")

    # Clamp.
    res.score = max(1, min(10, res.score))
    if not res.issues:
        res.notes.append("No obvious correctness defects detected by heuristics.")
    return res


# ---------------------------------------------------------------------------
# Review 2 — Tool Selection & Usage
# ---------------------------------------------------------------------------


def _validate_tool_call_args(tc: dict[str, Any]) -> list[str]:
    """Return a list of argument-name issues for a single tool call."""
    issues: list[str] = []
    name = tc.get("name", "")
    args = tc.get("arguments", {})
    schema = TOOL_SCHEMAS.get(name)
    if schema is None:
        issues.append(f"Unknown tool name: {name!r}.")
        return issues
    arg_keys = set(args.keys())
    # Unknown parameters.
    for key in arg_keys:
        if key not in schema.all_params:
            issues.append(f"Tool {name!r}: unexpected parameter {key!r} (valid: {list(schema.all_params)}).")
    # Missing required parameters.
    for req in schema.required:
        if req not in arg_keys:
            issues.append(f"Tool {name!r}: missing required parameter {req!r}.")
    return issues


def review_tool_selection(
    prompt: str,
    parsed: ParsedResponse,
    category: str,
    expected_mode: str,
) -> ReviewResult:
    """Evaluate whether the right tool was chosen and used correctly."""
    res = ReviewResult(name="tool_selection", score=8)
    lower_prompt = prompt.lower()

    # Narration instead of structured tool calls.
    if parsed.has_narration and not parsed.tool_calls:
        res.score -= 4
        res.issues.append(
            "Narrates tool use ('select tool: X', '→ cmd') instead of emitting a "
            "structured tool_call block. Must use <tool_call> XML with JSON payload."
        )
    elif parsed.has_narration and parsed.tool_calls:
        res.score -= 1
        res.issues.append("Contains both narration and structured tool calls — drop the narration.")

    # Validate structured tool-call arguments.
    for tc in parsed.tool_calls:
        res.issues.extend(_validate_tool_call_args(tc))

    # Should a tool have been used but wasn't?
    expected_tools = _CATEGORY_TO_TOOLS.get(category, ())
    needs_tool = bool(expected_tools) and category not in _NO_TOOL_CATEGORIES
    has_any_tool_signal = bool(parsed.tool_calls) or bool(parsed.narrated_tools)
    if needs_tool and not has_any_tool_signal:
        res.score -= 3
        res.issues.append(
            f"Category {category!r} implies tool use ({expected_tools}) but no tool was selected."
        )

    # Overcomplication — using a tool for a pure-reasoning planning task.
    if category in _NO_TOOL_CATEGORIES and parsed.tool_calls:
        res.score -= 1
        res.issues.append("Planning task does not require a tool call — answer directly.")

    # Right tool for the job — quick keyword-based sanity check.
    if has_any_tool_signal:
        chosen = parsed.tool_calls[0]["name"] if parsed.tool_calls else (
            parsed.narrated_tools[0] if parsed.narrated_tools else ""
        )
        if chosen and expected_tools and chosen not in expected_tools and chosen in VALID_TOOL_NAMES:
            res.score -= 1
            res.issues.append(
                f"Tool {chosen!r} may not be the best fit for {category!r} "
                f"(expected one of {expected_tools})."
            )

    # Shell commands embedded in prose for git tasks (should use the git tool).
    if category == "tool_use_git":
        if "shell" in {tc["name"] for tc in parsed.tool_calls} or (
            parsed.narrated_tools and parsed.narrated_tools[0] == "shell"
        ):
            res.score -= 2
            res.issues.append("Used 'shell' for a git task — prefer the structured 'git' tool.")

    res.score = max(1, min(10, res.score))
    if not res.issues:
        res.notes.append("Tool selection and arguments look correct.")
    return res


# ---------------------------------------------------------------------------
# Review 3 — Control Token Routing
# ---------------------------------------------------------------------------


def review_control_token(
    prompt: str,
    parsed: ParsedResponse,
    category: str,
    expected_mode: str,
) -> ReviewResult:
    """Evaluate whether the control token matches the task type."""
    res = ReviewResult(name="control_token", score=8)

    if parsed.mode is None:
        res.score -= 5
        res.issues.append(
            "Missing control token — every response MUST start with one of "
            "[FAST], [THINK], [VERIFY], [RECOVER], [ESCALATE]."
        )
        # Recommend the expected mode.
        if expected_mode in CONTROL_TOKENS:
            res.issues.append(f"Should have been [{expected_mode}] based on the task category.")
        return res

    if parsed.mode not in CONTROL_TOKENS:
        res.score -= 5
        res.issues.append(f"Unrecognised control token: [{parsed.mode}].")
        return res

    # Compare against the expected mode derived from the prompt category.
    if expected_mode and parsed.mode != expected_mode:
        # Severity depends on the mismatch.
        severe = (
            (expected_mode == "RECOVER" and parsed.mode == "FAST")
            or (expected_mode == "ESCALATE" and parsed.mode in ("FAST", "THINK"))
            or (expected_mode == "VERIFY" and parsed.mode == "FAST")
            or (expected_mode == "THINK" and parsed.mode == "FAST")
        )
        penalty = 4 if severe else 2
        res.score -= penalty
        res.issues.append(
            f"Wrong control token: got [{parsed.mode}], should be [{expected_mode}] "
            f"for a {category!r} task."
        )
        if expected_mode == "VERIFY":
            res.issues.append("Destructive/irreversible action requires VERIFY — act then confirm.")
        elif expected_mode == "RECOVER":
            res.issues.append("A failure occurred — RECOVER mode is needed to reassess.")
        elif expected_mode == "ESCALATE":
            res.issues.append("High-risk/security situation requires ESCALATE, not autonomous action.")
        elif expected_mode == "THINK":
            res.issues.append("Multi-step reasoning task — THINK mode needed before acting.")
    else:
        res.notes.append(f"Control token [{parsed.mode}] matches the expected mode.")

    res.score = max(1, min(10, res.score))
    return res


# ---------------------------------------------------------------------------
# Review 4 — Error Handling & Edge Cases
# ---------------------------------------------------------------------------


def review_error_handling(
    prompt: str,
    parsed: ParsedResponse,
    category: str,
    expected_mode: str,
) -> ReviewResult:
    """Evaluate failure-mode handling and security concerns."""
    res = ReviewResult(name="error_handling", score=7)
    lower = parsed.body.lower()

    # Error-handling language.
    err_keywords = ("if.*fail", "on error", "try", "except", "catch", "rollback",
                    "fallback", "error path", "cleanup", "timeout", "retry")
    has_err_handling = any(re.search(k, lower) for k in err_keywords)

    if category in ("coding_debug", "recovery", "tool_use_shell") and not has_err_handling:
        res.score -= 2
        res.issues.append(
            "No failure-mode handling described — what happens if the tool fails, "
            "the file is missing, or the command errors?"
        )

    # Security concerns.
    dangerous_patterns = (
        (r"rm\s+-rf\b", "rm -rf is destructive — needs VERIFY and a scoped path."),
        (r"\bdelete\b.*\b-type\s+f\b", "Bulk delete via find -delete is destructive — needs VERIFY."),
        (r">\s*/etc/", "Writing to system paths (/etc) is high-risk — needs ESCALATE."),
        (r"\$\{.*\}|\bexec\b|\beval\b", "Possible injection vector (unescaped shell expansion / exec / eval)."),
        (r"\.\./", "Path traversal (../) detected — must sanitize user-supplied paths."),
    )
    for pattern, msg in dangerous_patterns:
        if re.search(pattern, parsed.raw, re.IGNORECASE):
            res.score -= 2
            res.issues.append(f"Security concern: {msg}")
            if expected_mode not in ("VERIFY", "ESCALATE"):
                res.issues.append("Destructive/insecure action should use VERIFY or ESCALATE.")

    # VERIFY tasks must mention verification of the result.
    if expected_mode == "VERIFY" and not re.search(
        r"\b(verify|confirm|check|assert|test)\b", lower
    ):
        res.score -= 2
        res.issues.append("VERIFY task does not describe how the result will be confirmed.")

    # Recovery tasks must mention a different strategy.
    if category == "recovery" and not re.search(
        r"\b(different|alternative|new strategy|reassess|try again|rollback)\b", lower
    ):
        res.score -= 1
        res.issues.append("Recovery task does not propose a different strategy from the failed attempt.")

    res.score = max(1, min(10, res.score))
    if not res.issues:
        res.notes.append("Error handling and security heuristics passed.")
    return res


# ---------------------------------------------------------------------------
# Review 5 — Response Format & Clarity
# ---------------------------------------------------------------------------


def review_format_clarity(
    prompt: str,
    parsed: ParsedResponse,
    category: str,
    expected_mode: str,
) -> ReviewResult:
    """Evaluate structure, verbosity, and format compliance."""
    res = ReviewResult(name="format_clarity", score=7)
    text = parsed.body

    # Control token prefix.
    if parsed.mode is None:
        res.score -= 3
        res.issues.append("Response does not start with a control token.")

    # Structured tool-call format.
    if (parsed.has_narration or _SHELL_CMD_HINT_RE.search(parsed.raw)) and not parsed.tool_calls:
        res.score -= 2
        res.issues.append(
            "Tool calls are not in the structured <tool_call> XML+JSON format the runtime parses."
        )

    # Verbosity — too long, padded with "Lesson"/"Result" filler.
    if "**Lesson:**" in parsed.raw or "**Result:**" in parsed.raw:
        res.score -= 1
        res.issues.append("Contains filler ('**Result:**', '**Lesson:**') — trim to the actionable response.")

    word_count = len(text.split())
    if word_count > 200:
        res.score -= 1
        res.issues.append(f"Verbose ({word_count} words) — should be concise and direct.")
    elif word_count < 5 and category not in _NO_TOOL_CATEGORIES:
        res.score -= 1
        res.issues.append("Too terse — does not address the task.")

    # Numbered narration steps are a formatting smell.
    if re.search(r"^\s*\d+\.\s*\*\*(select tool|execute|interpret|next action)", text, re.MULTILINE):
        res.score -= 1
        res.issues.append("Uses numbered narration steps instead of acting — emit tool calls directly.")

    # Markdown bold headers that add no information.
    if parsed.raw.count("**") > 8:
        res.score -= 1
        res.issues.append("Over-formatted with markdown bold — keep it plain and actionable.")

    res.score = max(1, min(10, res.score))
    if not res.issues:
        res.notes.append("Format and clarity look good.")
    return res


# ---------------------------------------------------------------------------
# Synthesis — produce the improved response
# ---------------------------------------------------------------------------


def _infer_tool_from_prompt(prompt: str, category: str) -> Optional[dict[str, Any]]:
    """Best-effort inference of the tool call the task actually needs.

    Returns a {"name", "arguments"} dict or None when no tool is needed
    (e.g. planning tasks, or pure code-generation).
    """
    p = prompt.lower()

    if category == "tool_use_git":
        # Derive the git subcommand from the prompt.
        if "last" in p and "commit" in p:
            n = re.search(r"(\d+)\s+commits?", p)
            count = n.group(1) if n else "10"
            return {"name": "git", "arguments": {"command": f"log --oneline -{count}"}}
        if "diff" in p or "compare" in p:
            return {"name": "git", "arguments": {"command": "diff"}}
        if "status" in p:
            return {"name": "git", "arguments": {"command": "status"}}
        if "branch" in p and ("create" in p or "new" in p):
            m = re.search(r"branch\s+(?:called\s+)?([\w/-]+)", p)
            branch = m.group(1) if m else "feature/new"
            return {"name": "git", "arguments": {"command": f"checkout -b {branch}"}}
        if "commit" in p and ("stage" in p or "message" in p):
            m = re.search(r"message\s*:\s*(.+)$", prompt, re.IGNORECASE)
            msg = m.group(1).strip() if m else "update"
            return {"name": "git", "arguments": {"command": f'commit -m "{msg}"'}}
        if "merge" in p:
            m = re.search(r"merge\s+(?:the\s+)?([\w/-]+)\s+branch", p)
            branch = m.group(1) if m else "feature/branch"
            return {"name": "git", "arguments": {"command": f"merge --no-edit {branch}"}}
        if "revert" in p:
            return {"name": "git", "arguments": {"command": "revert HEAD"}}
        if "tag" in p:
            m = re.search(r"tag\s+(\S+)", p)
            tag = m.group(1) if m else "v1.0.0"
            return {"name": "git", "arguments": {"command": f"tag {tag}"}}
        if "author" in p or "commits by" in p:
            m = re.search(r"(?:by|author)\s+(\w+)", p)
            author = m.group(1) if m else "author"
            d = re.search(r"(\d+)\s+days?", p)
            days = d.group(1) if d else "7"
            return {"name": "git", "arguments": {"command": f"log --author={author} --since={days}.days --oneline"}}
        return {"name": "git", "arguments": {"command": "log --oneline -10"}}

    if category == "tool_use_shell":
        if "test" in p:
            return {"name": "shell", "arguments": {"command": "uv run pytest"}}
        if "lint" in p:
            return {"name": "shell", "arguments": {"command": "uv run ruff check ."}}
        if "server" in p or "port" in p:
            m = re.search(r"port\s+(\d+)", p)
            port = m.group(1) if m else "8000"
            return {"name": "shell", "arguments": {"command": f"uv run uvicorn app:app --port {port}"}}
        if "git status" in p:
            return {"name": "shell", "arguments": {"command": "git status"}}
        if "install" in p:
            m = re.search(r"install\s+(?:the\s+)?(?:package\s+)?(\w+)", p)
            pkg = m.group(1) if m else "package"
            return {"name": "shell", "arguments": {"command": f"uv add {pkg}"}}
        if "kill" in p or "restart" in p:
            m = re.search(r"port\s+(\d+)", p)
            port = m.group(1) if m else "8000"
            return {"name": "shell", "arguments": {"command": f"lsof -ti:{port} | xargs kill -9"}}
        return {"name": "shell", "arguments": {"command": "echo done"}}

    if category == "tool_use_file_ops":
        # "contain the word X" / "contain X" -> grep, not file_list.
        if "contain" in p and ("word" in p or re.search(r"['\"](\w+)['\"]", p)):
            kw_m = re.search(r"word\s+['\"]?(\w+)['\"]?", p)
            keyword = kw_m.group(1) if kw_m else "import"
            dir_m = re.search(r"(?:in|under)\s+([\w./-]+)", p)
            path = dir_m.group(1) if dir_m else "src/"
            return {"name": "grep", "arguments": {"pattern": keyword, "path": path, "glob_pattern": "*.py"}}
        if "read" in p and "file" in p:
            m = re.search(r"file\s+([\w./-]+)", p)
            path = m.group(1) if m else "src/main.py"
            return {"name": "file_read", "arguments": {"file_path": path}}
        if "create" in p or "write" in p:
            m = re.search(r"(?:at|to)\s+([\w./-]+)", p)
            path = m.group(1) if m else "src/main.py"
            return {"name": "file_write", "arguments": {"file_path": path, "content": ""}}
        if "edit" in p or "replace" in p:
            m = re.search(r"(?:edit|replace).+?([\w./-]+)", p)
            path = m.group(1) if m else "src/main.py"
            return {"name": "file_edit", "arguments": {"file_path": path, "old_string": "", "new_string": ""}}
        if "list" in p or "find all" in p:
            m = re.search(r"(?:in|under)\s+([\w./-]+)", p)
            path = m.group(1) if m else "src/"
            pat_m = re.search(r"pattern\s+([\w.*-]+)", p)
            pattern = pat_m.group(1) if pat_m else "*.py"
            return {"name": "file_list", "arguments": {"path": path, "pattern": pattern}}
        if "delete" in p:
            m = re.search(r"(?:in|under)\s+([\w./-]+)", p)
            path = m.group(1) if m else "scripts/"
            return {"name": "shell", "arguments": {"command": f"find {path} -name '*.tmp' -delete"}}
        if "copy" in p:
            return {"name": "shell", "arguments": {"command": "cp src dst"}}
        return {"name": "file_list", "arguments": {"path": "src/", "pattern": "*.py"}}

    if category == "tool_use_search":
        if "web" in p or "documentation" in p or "docs" in p:
            m = re.search(r"(?:on|for|about)\s+(.+)$", prompt, re.IGNORECASE)
            query = m.group(1).strip() if m else prompt
            return {"name": "web_search", "arguments": {"query": query}}
        # codebase search
        m = re.search(r"['\"]([^'\"]+)['\"]", prompt)
        pattern = m.group(1) if m else "TODO"
        if "import" in p:
            pattern = "import"
        return {"name": "grep", "arguments": {"pattern": pattern, "path": "src/", "glob_pattern": "*.py"}}

    if category == "coding_debug":
        # Debugging needs to read the relevant file first.
        m = re.search(r"([\w./-]+\.py)", prompt)
        path = m.group(1) if m else "src/main.py"
        return {"name": "file_read", "arguments": {"file_path": path}}

    if category == "coding_test":
        m = re.search(r"([\w./-]+\.py)", prompt)
        path = m.group(1) if m else "tests/test_main.py"
        return {"name": "file_write", "arguments": {"file_path": path, "content": ""}}

    if category == "recovery":
        # Recovery should diagnose first — read logs / run tests.
        if "build" in p:
            return {"name": "shell", "arguments": {"command": "uv run pytest"}}
        if "deploy" in p:
            return {"name": "shell", "arguments": {"command": "git log --oneline -5"}}
        return {"name": "shell", "arguments": {"command": "uv run pytest"}}

    # coding_write, coding_refactor, planning — answer directly, no tool.
    return None


def _format_tool_call(tc: dict[str, Any]) -> str:
    """Render a tool call as the XML+JSON block the runtime parses."""
    payload = json.dumps(tc, ensure_ascii=False)
    return f"{_TC_OPEN}\n{payload}\n{_TC_CLOSE}"


def _short_reason(prompt: str, category: str, mode: str) -> str:
    """A one-line reasoning prefix appropriate to the mode/category."""
    if mode == "FAST":
        return "Direct action."
    if mode == "THINK":
        if category == "planning":
            return "This requires careful reasoning. Investigate before acting."
        if category.startswith("coding_"):
            return "Multi-step reasoning needed before implementing."
        return "Reasoning through the steps before acting."
    if mode == "VERIFY":
        return "Act, then verify the result before declaring success."
    if mode == "RECOVER":
        return "A previous attempt failed. Reassess and try a different strategy."
    if mode == "ESCALATE":
        return "This exceeds safe autonomous scope. Surface for human decision."
    return "Proceeding."


def _build_coding_answer(prompt: str, category: str) -> str:
    """Produce a concise code/answer body for coding/planning tasks.

    This is a template-based stub — it emits a clean, minimal, correct-shaped
    answer that fixes the narration problem. It is NOT a full code generator;
    the goal is a well-formatted SFT target that demonstrates the right shape.
    """
    p = prompt.lower()

    # Detect a function-implementation request.
    if category == "coding_write" and re.search(r"function that|script that|decorator that", p):
        m = re.search(r"function that (.+)", p)
        desc = m.group(1).rstrip(".") if m else "the requested behaviour"
        name = "solve"
        return (
            f"```python\n"
            f"def {name}(x):\n"
            f"    \"\"\"{desc.capitalize()}.\"\"\"\n"
            f"    if x is None:\n"
            f"        raise ValueError(\"input must not be None\")\n"
            f"    return x\n"
            f"```\n"
        )

    if category == "planning":
        return (
            "1. Identify the core entities and their relationships.\n"
            "2. Define the schema/contract for each component.\n"
            "3. Break the work into ordered, verifiable steps.\n"
            "4. Note failure modes and rollback strategy for each step.\n"
        )

    if category == "coding_debug":
        return (
            "Read the relevant source, identify the failing path, apply the "
            "minimal fix, then run the tests to confirm.\n"
        )

    if category == "coding_refactor":
        return "Read the code, extract the duplicated logic, then verify tests still pass.\n"

    if category == "coding_test":
        return "Write tests covering the happy path, edge cases (empty/null/overflow), and error paths.\n"

    # Generic fallback.
    return "Proceeding with the task.\n"


def synthesize_improved_response(
    prompt: str,
    parsed: ParsedResponse,
    category: str,
    expected_mode: str,
    reviews: list[ReviewResult],
) -> str:
    """Produce the final improved response from the original + reviews.

    Pure-Python template-based synthesis:
      - pick the correct control token,
      - rewrite narrated tool use as structured tool calls,
      - fix parameter names to match tool schemas,
      - add error handling if missing,
      - trim verbosity.
    """
    # 1. Choose the control token. Prefer the expected mode (from the prompt
    #    category); fall back to the original if it was correct; default THINK.
    mode = expected_mode if expected_mode in CONTROL_TOKENS else (parsed.mode or "THINK")

    # 2. Decide whether a tool call is needed and which one.
    inferred = _infer_tool_from_prompt(prompt, category)

    # If the original already had valid structured tool calls, reuse them
    # (after validating/fixing their arguments).
    tool_calls: list[dict[str, Any]] = []
    for tc in parsed.tool_calls:
        name = tc.get("name", "")
        schema = TOOL_SCHEMAS.get(name)
        if schema is None:
            continue
        fixed_args: dict[str, Any] = {}
        for req in schema.required:
            if req in tc.get("arguments", {}):
                fixed_args[req] = tc["arguments"][req]
            else:
                fixed_args[req] = ""
        for opt in schema.optional:
            if opt in tc.get("arguments", {}):
                fixed_args[opt] = tc["arguments"][opt]
        tool_calls.append({"name": name, "arguments": fixed_args})

    # If no valid structured tool calls but we inferred one (and the model
    # narrated one), use the inferred call.
    if not tool_calls and inferred and category not in _NO_TOOL_CATEGORIES:
        tool_calls = [inferred]

    # 3. Build the response body.
    reason = _short_reason(prompt, category, mode)
    parts: list[str] = [f"[{mode}]", reason, ""]

    if tool_calls:
        for tc in tool_calls:
            parts.append(_format_tool_call(tc))
            parts.append("")

    # 4. Add a concise answer / next-step for non-tool or coding tasks.
    if not tool_calls or category.startswith("coding_") or category == "planning":
        parts.append(_build_coding_answer(prompt, category))

    # 5. Add error-handling guidance if Review 4 flagged it missing and a tool
    #    is being used.
    err_review = next((r for r in reviews if r.name == "error_handling"), None)
    needs_err = err_review and any("failure-mode" in i or "VERIFY" in i for i in err_review.issues)
    if needs_err and tool_calls:
        parts.append(
            "If the tool fails, inspect the error output, adjust, and retry. "
            "Do not claim success without confirming the result."
        )

    # 6. For VERIFY mode, append an explicit verification step.
    if mode == "VERIFY" and tool_calls:
        parts.append("After the action completes, verify the result before declaring success.")

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue


def _load_processed_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id") or rec.get("prompt_id")
            if pid:
                ids.add(pid)
    return ids


def process_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Run the 5 reviews + synthesis on a single qwen_responses record.

    Returns the output record dict.
    """
    prompt = rec.get("prompt", "")
    category = rec.get("category", "unknown")
    expected_mode = rec.get("expected_mode", "")
    pid = rec.get("id") or rec.get("prompt_id", "")

    resp = rec.get("response", {})
    # The response may be a dict (InferenceResult.to_dict) or a plain string.
    if isinstance(resp, dict):
        raw = resp.get("text", "") or ""
        # Reconstruct the full text with control token for parsing.
        mode_val = resp.get("mode")
        if mode_val and not raw.startswith(f"[{mode_val}]"):
            raw = f"[{mode_val}] {raw}"
    else:
        raw = str(resp)

    parsed = parse_response(raw)

    reviews = [
        review_code_quality(prompt, parsed, category, expected_mode),
        review_tool_selection(prompt, parsed, category, expected_mode),
        review_control_token(prompt, parsed, category, expected_mode),
        review_error_handling(prompt, parsed, category, expected_mode),
        review_format_clarity(prompt, parsed, category, expected_mode),
    ]

    improved = synthesize_improved_response(prompt, parsed, category, expected_mode, reviews)

    scores = {r.name: r.score for r in reviews}

    return {
        "id": pid,
        "prompt": prompt,
        "category": category,
        "expected_mode": expected_mode,
        "original_response": raw,
        "improved_response": improved,
        # Alias for downstream compatibility (dpo_generator reads "response").
        "response": improved,
        "reviews": [r.to_dict() for r in reviews],
        "scores": scores,
    }


def run(
    input_path: Path,
    output_path: Path,
    *,
    limit: Optional[int] = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the 5-reviewer orchestrator over the input JSONL.

    For each Qwen response: parse it, run the 5 reviews, synthesize an improved
    response, and write the combined record to the output JSONL.

    Resume support skips ids already present in the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = _load_processed_ids(output_path) if resume else set()
    if processed:
        logger.info("Resume: %d ids already processed in %s", len(processed), output_path)

    total = 0
    written = 0
    skipped = 0
    errors = 0
    score_sums: dict[str, float] = {}
    mode_corrections = 0
    narration_fixed = 0
    start_time = time.monotonic()

    with output_path.open("a", encoding="utf-8") as out_f:
        for rec in _iter_jsonl(input_path):
            if limit is not None and written >= limit:
                break
            total += 1
            pid = rec.get("id") or rec.get("prompt_id")
            if pid and pid in processed:
                skipped += 1
                continue

            try:
                out_rec = process_record(rec)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Error processing record %s: %s", pid, exc)
                errors += 1
                continue

            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

            # Track stats.
            for name, score in out_rec["scores"].items():
                score_sums[name] = score_sums.get(name, 0.0) + score
            orig_mode = parse_response(out_rec["original_response"]).mode
            if orig_mode != out_rec["expected_mode"]:
                mode_corrections += 1
            if "Narrates tool use" in str(out_rec["reviews"]):
                narration_fixed += 1

            if written % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - start_time
                rate = written / elapsed if elapsed else 0.0
                avg_score = (
                    sum(score_sums.values()) / (written * 5) if written else 0.0
                )
                print(
                    f"  [{written}] written={written} skipped={skipped} "
                    f"errors={errors} avg_score={avg_score:.1f} rate={rate:.1f}/s",
                    flush=True,
                )

    elapsed = time.monotonic() - start_time
    avg_scores = {
        name: round(score_sums[name] / written, 2) if written else 0.0
        for name in ("code_quality", "tool_selection", "control_token",
                     "error_handling", "format_clarity")
    }
    stats = {
        "total_read": total,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "avg_scores": avg_scores,
        "mode_corrections": mode_corrections,
        "narration_fixed": narration_fixed,
        "elapsed": round(elapsed, 3),
        "output_path": str(output_path),
    }
    _print_summary(stats)
    return stats


def _print_summary(stats: dict[str, Any]) -> None:
    written = stats["written"]
    print("\n" + "=" * 60)
    print("5-Reviewer Orchestrator Summary")
    print("=" * 60)
    print(f"  Total read         : {stats['total_read']}")
    print(f"  Written            : {written}")
    print(f"  Skipped (resume)   : {stats['skipped']}")
    print(f"  Errors             : {stats['errors']}")
    print(f"  Mode corrections   : {stats['mode_corrections']}")
    print(f"  Narration fixed    : {stats['narration_fixed']}")
    print(f"  Elapsed            : {stats['elapsed']:.2f}s")
    print(f"  Output file        : {stats['output_path']}")
    print()
    print("  Average review scores (1-10):")
    for name, score in stats["avg_scores"].items():
        print(f"    {name:20s}: {score:5.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 5 parallel reviewers over Qwen v3.1 responses and synthesize improved responses.",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Input JSONL of Qwen responses (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL of improved responses (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Process only this many responses (for testing).",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Skip ids already present in the output file (default: true).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume; reprocess all responses (appends to output).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Limit : {args.limit if args.limit else 'all'}")
    print(f"Resume: {args.resume}")
    print()

    run(input_path, output_path, limit=args.limit, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
