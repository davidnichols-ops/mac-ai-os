"""Shared fixtures for the operator runtime test suite.

Provides a ScriptedModel that returns pre-programmed ModelResponse sequences,
so we can test the full agent loop without loading the real MLX model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.core.agent_loop import AgentLoop, LoopConfig
from app.core.safety import SafetyGate
from app.core.tool_executor import ToolExecutor
from app.models.base import Model
from app.models.types import Message, ModelResponse, ToolCall
from app.tools.builtin import builtin_tools
from app.tools.registry import ToolRouter


class ScriptedModel(Model):
    """A model that returns pre-programmed responses in sequence.

    Each call to chat() pops the next response from the script.
    If the script runs out, raises RuntimeError — tests should provide
    exactly enough responses.
    """

    name = "scripted"
    provider = "test"

    def __init__(self, responses: list[ModelResponse] | None = None):
        self._responses: list[ModelResponse] = list(responses or [])
        self._call_index = 0
        self._all_calls: list[list[Message]] = []

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
        raise NotImplementedError("ScriptedModel only supports chat()")

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ModelResponse:
        self._all_calls.append(list(messages))
        if self._call_index >= len(self._responses):
            raise RuntimeError(
                f"ScriptedModel ran out of responses after {self._call_index} calls. "
                f"Provide more responses."
            )
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    @property
    def call_count(self) -> int:
        return self._call_index

    @property
    def all_calls(self) -> list[list[Message]]:
        return self._all_calls


def make_tool_call(name: str, **arguments) -> ToolCall:
    """Create a ToolCall with auto-generated id."""
    return ToolCall(id=f"call_test_{name}", name=name, arguments=dict(arguments))


def make_tool_response(
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    text: str = "",
    mode: str | None = None,
) -> ModelResponse:
    """Create a ModelResponse that calls a single tool."""
    return ModelResponse(
        text=text or f"Calling {tool_name}",
        model="scripted",
        provider="test",
        mode=mode,
        tool_calls=[make_tool_call(tool_name, **(arguments or {}))],
        finish_reason="tool_calls",
    )


def make_text_response(text: str, *, mode: str | None = None) -> ModelResponse:
    """Create a ModelResponse with plain text (no tool calls)."""
    return ModelResponse(
        text=text,
        model="scripted",
        provider="test",
        mode=mode,
        tool_calls=[],
        finish_reason="stop",
    )


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A temporary workspace directory with a sample file."""
    (tmp_path / "hello.py").write_text("print('hello world')\n")
    (tmp_path / "data.txt").write_text("line1\nline2\nline3\n")
    return tmp_path


@pytest.fixture
def tool_router() -> ToolRouter:
    """A ToolRouter with all built-in tools and permissions granted."""
    return ToolRouter(
        builtin_tools(),
        granted_permissions={"shell.exec", "file.read", "file.write", "git.exec"},
    )


@pytest.fixture
def safety_gate() -> SafetyGate:
    """A SafetyGate with no approval callback (auto-approve LOW/MEDIUM only)."""
    return SafetyGate()


@pytest.fixture
def tool_executor(tool_router: ToolRouter, safety_gate: SafetyGate) -> ToolExecutor:
    """A ToolExecutor with safety gate, no trajectory recorder."""
    return ToolExecutor(tool_router, safety=safety_gate)


@pytest.fixture
def loop_config(tmp_workspace: Path) -> LoopConfig:
    """LoopConfig for tests — no persistence, short timeout."""
    return LoopConfig(
        max_steps=10,
        max_errors=5,
        timeout_s=10.0,
        persist_trajectories=False,
        workspace=str(tmp_workspace),
    )


@pytest.fixture
def agent_loop(
    tool_executor: ToolExecutor,
    loop_config: LoopConfig,
) -> AgentLoop:
    """An AgentLoop with a placeholder model — tests inject ScriptedModel."""
    # We need a model for the constructor, but tests will replace it
    dummy = ScriptedModel([make_text_response("placeholder")])
    return AgentLoop(dummy, tool_executor, config=loop_config)
