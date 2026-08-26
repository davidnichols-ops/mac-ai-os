"""Tests for app/core/tool_executor.py - tool execution with safety gates.

Covers:
- _DEFAULT_RISK lookup and fallback
- execute() success path
- execute() safety block (returns error, records trajectory)
- execute() approval required + denied
- execute() approval required + granted
- execute() tool raises exception (caught)
- _format_result: output, truncation, error, metadata filtering, empty
- available_tools_schema delegates to router
"""

from __future__ import annotations

from typing import Any

from app.core.safety import RiskLevel, SafetyGate
from app.core.tool_executor import _DEFAULT_RISK, ToolExecutor
from app.core.trajectory import TrajectoryRecorder
from app.tools.base import Tool, ToolResult
from app.tools.registry import ToolRouter


class _FakeTool(Tool):
    name = "fake"
    description = "fake tool"
    permissions = set()

    def __init__(self, result: ToolResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def execute(self, args: dict[str, Any]) -> ToolResult:
        self.calls.append(args)
        return self._result


class _RaisingTool(Tool):
    name = "boom"
    description = "raises"
    permissions = set()

    def execute(self, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")


class _ThrowingRouter(ToolRouter):
    """A router that raises from execute() to test the executor's own handler."""

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("router exploded")


def _router(tools: list[Tool]) -> ToolRouter:
    return ToolRouter(tools=tools, granted_permissions={"*"})


# --- _DEFAULT_RISK -----------------------------------------------------------


class TestDefaultRisk:
    def test_known_tools(self):
        assert _DEFAULT_RISK["shell"] == RiskLevel.MEDIUM
        assert _DEFAULT_RISK["file_read"] == RiskLevel.LOW
        assert _DEFAULT_RISK["file_delete"] == RiskLevel.HIGH
        assert _DEFAULT_RISK["git_push"] == RiskLevel.HIGH

    def test_get_risk_level_known(self):
        ex = ToolExecutor(_router([]))
        assert ex._get_risk_level("file_read") == RiskLevel.LOW
        assert ex._get_risk_level("shell") == RiskLevel.MEDIUM

    def test_get_risk_level_unknown_defaults_medium(self):
        ex = ToolExecutor(_router([]))
        assert ex._get_risk_level("nonexistent_tool") == RiskLevel.MEDIUM


# --- execute() success -------------------------------------------------------


class TestExecuteSuccess:
    def test_success_returns_result_text(self):
        tool = _FakeTool(ToolResult(ok=True, output="hello world"))
        ex = ToolExecutor(_router([tool]))
        text, ok, error = ex.execute("fake", {"x": 1})
        assert ok is True
        assert error is None
        assert text == "hello world"
        assert tool.calls == [{"x": 1}]

    def test_success_with_error_field(self):
        tool = _FakeTool(ToolResult(ok=False, output=None, error="bad"))
        ex = ToolExecutor(_router([tool]))
        text, ok, error = ex.execute("fake", {})
        assert ok is False
        assert error == "bad"
        assert "[error] bad" in text

    def test_unknown_tool(self):
        ex = ToolExecutor(_router([]))
        text, ok, error = ex.execute("nope", {})
        assert ok is False
        assert "unknown tool" in (error or "")


# --- execute() safety block --------------------------------------------------


class TestSafetyBlock:
    def test_blocked_command_returns_error(self):
        tool = _FakeTool(ToolResult(ok=True, output="should not run"))
        ex = ToolExecutor(_router([tool]))
        text, ok, error = ex.execute("shell", {"command": "rm -rf /"})
        assert ok is False
        assert "Safety block" in (error or "")
        assert tool.calls == []  # tool never executed

    def test_blocked_records_trajectory(self):
        tool = _FakeTool(ToolResult(ok=True, output="x"))
        traj = TrajectoryRecorder(task="t")
        ex = ToolExecutor(_router([tool]), trajectory=traj)
        ex.execute("shell", {"command": "rm -rf /"})
        # a TOOL_CALL and a TOOL_RESULT step should be recorded
        types = [s.step_type.value for s in traj.trajectory.steps]
        assert "tool_call" in types
        assert "tool_result" in types


# --- execute() approval ------------------------------------------------------


class TestApproval:
    def test_approval_denied(self):
        tool = _FakeTool(ToolResult(ok=True, output="x"))
        tool.name = "file_delete"
        gate = SafetyGate(approval_callback=lambda r, t, a: False)
        ex = ToolExecutor(_router([tool]), safety=gate)
        text, ok, error = ex.execute("file_delete", {"path": "/tmp/x"})
        assert ok is False
        assert "denied" in (error or "").lower()
        assert tool.calls == []

    def test_approval_granted_executes(self):
        tool = _FakeTool(ToolResult(ok=True, output="deleted"))
        tool.name = "file_delete"
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        ex = ToolExecutor(_router([tool]), safety=gate)
        text, ok, error = ex.execute("file_delete", {"path": "/tmp/x"})
        assert ok is True
        assert text == "deleted"
        assert tool.calls == [{"path": "/tmp/x"}]

    def test_approval_granted_increments_trajectory(self):
        tool = _FakeTool(ToolResult(ok=True, output="x"))
        tool.name = "file_delete"
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        traj = TrajectoryRecorder(task="t")
        ex = ToolExecutor(_router([tool]), safety=gate, trajectory=traj)
        ex.execute("file_delete", {"path": "/tmp/x"})
        assert traj.trajectory.approvals_requested == 1


# --- execute() exception -----------------------------------------------------


class TestExecuteException:
    def test_tool_exception_caught_by_router(self):
        # The ToolRouter catches tool exceptions and returns ToolResult(ok=False).
        # The executor sees that as a normal failed result.
        ex = ToolExecutor(_router([_RaisingTool()]))
        text, ok, error = ex.execute("boom", {})
        assert ok is False
        assert error == "kaboom"
        assert "[error] kaboom" in text

    def test_router_exception_caught_by_executor(self):
        # If the router itself raises (not the tool), the executor's own
        # try/except catches it and wraps it as "Tool execution error".
        ex = ToolExecutor(_ThrowingRouter(tools=[]))
        text, ok, error = ex.execute("anything", {})
        assert ok is False
        assert "Tool execution error" in (error or "")
        assert "router exploded" in (error or "")

    def test_tool_exception_records_trajectory(self):
        traj = TrajectoryRecorder(task="t")
        ex = ToolExecutor(_router([_RaisingTool()]), trajectory=traj)
        ex.execute("boom", {})
        types = [s.step_type.value for s in traj.trajectory.steps]
        assert "tool_result" in types


# --- _format_result ----------------------------------------------------------


class TestFormatResult:
    def test_output_only(self):
        ex = ToolExecutor(_router([]))
        assert ex._format_result(ToolResult(ok=True, output="hi")) == "hi"

    def test_no_output_no_error(self):
        ex = ToolExecutor(_router([]))
        assert ex._format_result(ToolResult(ok=True)) == "(no output)"

    def test_error_appended(self):
        ex = ToolExecutor(_router([]))
        out = ex._format_result(ToolResult(ok=False, output="partial", error="oops"))
        assert "partial" in out
        assert "[error] oops" in out

    def test_truncation(self):
        ex = ToolExecutor(_router([]), max_result_chars=10)
        out = ex._format_result(ToolResult(ok=True, output="x" * 100))
        assert out.startswith("x" * 10)
        assert "truncated" in out
        assert "100 total chars" in out

    def test_no_truncation_at_limit(self):
        ex = ToolExecutor(_router([]), max_result_chars=10)
        out = ex._format_result(ToolResult(ok=True, output="x" * 10))
        assert "truncated" not in out

    def test_metadata_filtered(self):
        ex = ToolExecutor(_router([]))
        out = ex._format_result(ToolResult(
            ok=True, output="x",
            metadata={"returncode": 0, "ignored": "yes", "exit_code": 0},
        ))
        assert "[metadata]" in out
        assert "returncode" in out
        assert "ignored" not in out

    def test_metadata_empty_omitted(self):
        ex = ToolExecutor(_router([]))
        out = ex._format_result(ToolResult(ok=True, output="x", metadata={"ignored": "yes"}))
        assert "[metadata]" not in out


# --- available_tools_schema --------------------------------------------------


class TestAvailableToolsSchema:
    def test_delegates_to_router(self):
        tool = _FakeTool(ToolResult(ok=True))
        ex = ToolExecutor(_router([tool]))
        schemas = ex.available_tools_schema()
        assert any(s.get("name") == "fake" for s in schemas)
