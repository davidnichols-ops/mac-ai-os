"""Tests for app/core/system_prompt.py - dynamic system prompt assembly.

Covers:
- BASE_SYSTEM_PROMPT content
- build_system_prompt with no extras (just base)
- tool formatting (OpenAI format + flat format, params, required, none)
- workspace / context / lessons / extra sections
- section ordering and joining
"""

from __future__ import annotations

from app.core.system_prompt import BASE_SYSTEM_PROMPT, _format_tools, build_system_prompt

# --- BASE_SYSTEM_PROMPT ------------------------------------------------------


class TestBasePrompt:
    def test_contains_identity(self):
        assert "Mac AI OS" in BASE_SYSTEM_PROMPT
        assert "MAOS" in BASE_SYSTEM_PROMPT

    def test_contains_tool_call_format(self):
        assert "<tool_call>" in BASE_SYSTEM_PROMPT

    def test_contains_core_principles(self):
        for principle in [
            "Inspect before acting",
            "Verify after acting",
            "Recover from failures",
            "Be efficient",
            "Be safe",
            "Be honest",
        ]:
            assert principle in BASE_SYSTEM_PROMPT


# --- build_system_prompt -----------------------------------------------------


class TestBuildSystemPrompt:
    def test_base_only(self):
        prompt = build_system_prompt()
        assert prompt == BASE_SYSTEM_PROMPT

    def test_workspace_section(self):
        prompt = build_system_prompt(workspace="/Users/testuser/mac-ai-os")
        assert "## Current Workspace" in prompt
        assert "/Users/testuser/mac-ai-os" in prompt

    def test_context_section(self):
        prompt = build_system_prompt(context="onboarding briefing here")
        assert "## Context" in prompt
        assert "onboarding briefing here" in prompt

    def test_lessons_section(self):
        prompt = build_system_prompt(lessons=["first lesson", "second lesson"])
        assert "## Lessons Learned" in prompt
        assert "1. first lesson" in prompt
        assert "2. second lesson" in prompt

    def test_lessons_empty_list_no_section(self):
        prompt = build_system_prompt(lessons=[])
        assert "## Lessons Learned" not in prompt

    def test_extra_section(self):
        prompt = build_system_prompt(extra="extra instructions")
        assert "extra instructions" in prompt

    def test_all_sections_present(self):
        prompt = build_system_prompt(
            tools=[{"name": "shell", "description": "run a command", "parameters": {}}],
            workspace="/tmp",
            context="ctx",
            lessons=["l1"],
            extra="ex",
        )
        assert "## Available Tools" in prompt
        assert "## Current Workspace" in prompt
        assert "## Context" in prompt
        assert "## Lessons Learned" in prompt
        assert "ex" in prompt

    def test_section_order(self):
        prompt = build_system_prompt(
            workspace="/tmp",
            context="ctx",
            lessons=["l1"],
            extra="EXTRA_MARKER_XYZ",
        )
        idx_workspace = prompt.index("## Current Workspace")
        idx_context = prompt.index("## Context")
        idx_lessons = prompt.index("## Lessons Learned")
        idx_extra = prompt.index("EXTRA_MARKER_XYZ")
        assert idx_workspace < idx_context < idx_lessons < idx_extra


# --- _format_tools -----------------------------------------------------------


class TestFormatTools:
    def test_flat_format_with_params(self):
        tools = [{
            "name": "shell",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                    "cwd": {"type": "string", "description": "Working directory."},
                },
                "required": ["command"],
            },
        }]
        out = _format_tools(tools)
        assert "## Available Tools" in out
        assert "### shell" in out
        assert "Run a shell command." in out
        assert "Parameters:" in out
        assert "command (string (required))" in out
        assert "cwd (string)" in out  # not required

    def test_openai_function_format(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path."},
                    },
                    "required": ["path"],
                },
            },
        }]
        out = _format_tools(tools)
        assert "### file_read" in out
        assert "Read a file." in out
        assert "path (string (required))" in out

    def test_no_properties(self):
        tools = [{"name": "status", "description": "Get status.", "parameters": {}}]
        out = _format_tools(tools)
        assert "Parameters: none" in out

    def test_missing_name_defaults_unknown(self):
        tools = [{"description": "no name", "parameters": {}}]
        out = _format_tools(tools)
        assert "### unknown" in out

    def test_missing_description_empty(self):
        tools = [{"name": "x", "parameters": {}}]
        out = _format_tools(tools)
        assert "### x" in out

    def test_multiple_tools(self):
        tools = [
            {"name": "a", "description": "tool a", "parameters": {}},
            {"name": "b", "description": "tool b", "parameters": {}},
        ]
        out = _format_tools(tools)
        assert "### a" in out
        assert "### b" in out

    def test_param_type_defaults_any(self):
        tools = [{
            "name": "x",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {"p": {"description": "no type"}},
                "required": [],
            },
        }]
        out = _format_tools(tools)
        assert "p (any)" in out
