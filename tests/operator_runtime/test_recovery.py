"""Recovery tests — model receives error, changes approach, completes task.

The key behavior: when a tool fails, the model should receive the error,
adapt its strategy, and still complete the task. This is the most important
capability for the Adaptive Operator — failure recovery.
"""

from __future__ import annotations

from app.core.agent_loop import AgentLoop
from app.core.trajectory import FailureCategory, StepType
from app.models.types import ModelResponse, ToolCall
from tests.operator_runtime.conftest import (
    ScriptedModel,
    make_text_response,
    make_tool_response,
)


class TestToolFailureRecovery:
    """Model calls a tool that fails, then recovers with a different approach."""

    def test_wrong_path_then_correct_path(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Model reads wrong path, gets error, reads correct path."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": "/nonexistent/file.py"}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Found the file. It prints hello world."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")

        assert result.success
        assert result.tool_calls == 2
        # Trajectory should show one failed tool result and one successful
        tool_results = [
            s for s in result.trajectory.steps if s.step_type == StepType.TOOL_RESULT
        ]
        assert len(tool_results) == 2
        assert tool_results[0].tool_ok is False  # first failed
        assert tool_results[1].tool_ok is True   # second succeeded

    def test_unknown_tool_then_correct_tool(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Model calls nonexistent tool, gets error, uses file_list instead."""
        model = ScriptedModel([
            ModelResponse(
                text="trying custom tool",
                model="scripted",
                provider="test",
                tool_calls=[ToolCall(id="c1", name="magic_list", arguments={})],
                finish_reason="tool_calls",
            ),
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_text_response("Recovered. Found hello.py and data.txt."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list files")

        assert result.success
        assert result.tool_calls == 2

    def test_shell_error_then_retry(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Model runs a failing shell command, then runs a correct one."""
        model = ScriptedModel([
            make_tool_response("shell", arguments={"command": "ls /nonexistent_dir_xyz"}),
            make_tool_response("shell", arguments={"command": f"ls {tmp_workspace}"}),
            make_text_response("Successfully listed the directory contents."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list directory via shell")

        assert result.success
        assert result.tool_calls == 2

    def test_multiple_failures_then_success(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Model fails twice, then succeeds on third attempt."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": "/wrong1.py"}),
            make_tool_response("file_read", arguments={"path": "/wrong2.py"}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Finally found the right file."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")

        assert result.success
        assert result.tool_calls == 3


class TestRecoveryTrajectoryRecording:
    """Verify recovery actions are properly recorded in the trajectory."""

    def test_error_steps_recorded(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Failed tool results should be recorded with tool_ok=False."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": "/nonexistent.py"}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Done"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")

        traj = result.trajectory
        failed_results = [
            s for s in traj.steps
            if s.step_type == StepType.TOOL_RESULT and s.tool_ok is False
        ]
        assert len(failed_results) == 1
        assert failed_results[0].error is not None

    def test_successful_recovery_trajectory(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """A trajectory with recovery should still be marked successful."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": "/bad.py"}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Done after recovery"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")

        assert result.success
        assert result.trajectory.success
        assert result.trajectory.failure_category == FailureCategory.NONE

    def test_recovery_does_not_count_as_infinite_loop(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Recovery with different args should not trigger infinite loop detection."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": "/bad1.py"}),
            make_tool_response("file_read", arguments={"path": "/bad2.py"}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Done"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")

        # Should succeed, not be flagged as infinite loop
        assert result.success
        assert result.failure_category != FailureCategory.INFINITE_LOOP
