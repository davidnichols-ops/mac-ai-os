"""System prompt builder — dynamic assembly of the model's operating context.

The system prompt is the "OS kernel" for the model. It tells the model:
- What it is (identity)
- What tools it has (capabilities)
- What context it's in (workspace, session, time)
- How to behave (guidelines)
- How to call tools (format)

The prompt is assembled dynamically for each agent loop invocation. This
is NOT hardcoded intelligence — it's context injection. The model decides
what to do with this information.
"""

from __future__ import annotations

from typing import Any

BASE_SYSTEM_PROMPT = """\
You are Mac AI OS (MAOS), a local-first AI operating system running on Apple Silicon.
You are the intelligence layer between the user and their computer.

You accomplish tasks by using tools. You think step by step, call tools to \
inspect and modify the environment, observe the results, and iterate until \
the task is complete.

## Core Principles

1. **Inspect before acting.** Always read the current state before making changes.
2. **Verify after acting.** Run tests, check files, confirm changes worked.
3. **Recover from failures.** If a tool fails, analyze the error and try a different approach.
4. **Be efficient.** Don't repeat the same action. Don't call tools unnecessarily.
5. **Be safe.** Dangerous operations require user approval. Respect safety decisions.
6. **Be honest.** If you can't do something, say so. Don't claim success without verification.

## Tool Call Format

To call a tool, output a JSON object wrapped in <tool_call> tags:

<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

You can call multiple tools by emitting multiple <tool_call> blocks.
After each tool call, you will receive the result. Use it to decide your next action.
When the task is complete, provide your final answer without any tool calls.
"""


def build_system_prompt(
    *,
    tools: list[dict[str, Any]] | None = None,
    workspace: str | None = None,
    context: str | None = None,
    lessons: list[str] | None = None,
    extra: str | None = None,
) -> str:
    """Assemble the system prompt for an agent loop invocation.

    Args:
        tools: Tool schemas to include in the prompt.
        workspace: Current workspace path.
        context: Additional context (onboarding briefing, blackboard state).
        lessons: Hard-won lessons to inject.
        extra: Any additional prompt text.

    Returns:
        The assembled system prompt string.
    """
    parts: list[str] = [BASE_SYSTEM_PROMPT]

    # Tool definitions
    if tools:
        parts.append(_format_tools(tools))

    # Workspace
    if workspace:
        parts.append(f"\n## Current Workspace\n{workspace}")

    # Context (onboarding, blackboard, etc.)
    if context:
        parts.append(f"\n## Context\n{context}")

    # Lessons
    if lessons:
        parts.append("\n## Lessons Learned")
        for i, lesson in enumerate(lessons, 1):
            parts.append(f"{i}. {lesson}")

    # Extra prompt text
    if extra:
        parts.append(f"\n{extra}")

    return "\n".join(parts)


def _format_tools(tools: list[dict[str, Any]]) -> str:
    """Format tool schemas as readable text for the model."""
    lines: list[str] = ["## Available Tools"]

    for t in tools:
        # Support both OpenAI format and flat format
        fn = t.get("function", t)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        lines.append(f"\n### {name}")
        lines.append(desc)

        if props:
            lines.append("Parameters:")
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else ""
                lines.append(f"  - {pname} ({ptype}{req}): {pdesc}")
        else:
            lines.append("Parameters: none")

    return "\n".join(lines)
