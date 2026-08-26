"""Agent loop tests — verify the think-act-observe-verify-done cycle."""

from __future__ import annotations

from app.core.agent_loop import AgentLoop, LoopConfig
from app.core.trajectory import FailureCategory
from app.models.types import ModelResponse, ToolCall
from tests.operator_runtime.conftest import (
    ScriptedModel,
    make_text_response,
    make_tool_response,
)


class TestThinkActObserveVerify:
    """Verify the full think-act-observe-verify-done cycle."""

    def test_single_tool_then_answer(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """THINK → ACT (file_list) → OBSERVE → VERIFY → DONE."""
        model = ScriptedModel([
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_text_response("Done. Found hello.py and data.txt."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list files in workspace")

        assert result.success
        assert result.steps == 2  # one tool call step, one final answer step
        assert result.tool_calls == 1
        assert result.failure_category == FailureCategory.NONE

    def test_multi_tool_chain(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """THINK → ACT → OBSERVE → THINK → ACT → OBSERVE → DONE."""
        model = ScriptedModel([
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Read the file. It prints hello world."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list then read hello.py")

        assert result.success
        assert result.tool_calls == 2
        assert result.steps == 3

    def test_no_tool_direct_answer(
        self,
        tool_executor,
        loop_config,
    ):
        """Model gives a direct answer without any tool calls."""
        model = ScriptedModel([
            make_text_response("The answer is 42."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("what is 6 * 7?")

        assert result.success
        assert result.tool_calls == 0
        assert result.steps == 1
        assert "42" in result.answer

    def test_trajectory_recorded_with_correct_steps(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Trajectory should have model_output, tool_call, tool_result steps."""
        model = ScriptedModel([
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_text_response("Done"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list files")

        traj = result.trajectory
        step_types = [s.step_type.value for s in traj.steps]
        # Should have: model_output, tool_call, tool_result, model_output
        assert "model_output" in step_types
        assert "tool_call" in step_types
        assert "tool_result" in step_types
        assert traj.success
        assert traj.total_tool_calls == 1


class TestStepBudget:
    """Verify step budget enforcement."""

    def test_step_budget_exceeded(
        self,
        tool_executor,
        tmp_workspace,
    ):
        """Model keeps calling tools — should hit step budget."""
        # Model that always calls a tool with DIFFERENT args each time
        # (to avoid infinite loop detection), never gives a final answer
        responses = [
            make_tool_response("file_list", arguments={"path": f"{tmp_workspace}/dir{i}"})
            for i in range(20)
        ]
        model = ScriptedModel(responses)
        config = LoopConfig(
            max_steps=3,
            max_errors=10,
            timeout_s=10.0,
            persist_trajectories=False,
            workspace=str(tmp_workspace),
        )
        loop = AgentLoop(model, tool_executor, config=config)
        result = loop.run("keep listing files")

        assert not result.success
        assert result.failure_category == FailureCategory.STEP_BUDGET_EXCEEDED

    def test_step_budget_exact_limit(
        self,
        tool_executor,
        tmp_workspace,
    ):
        """Model completes on the last allowed step."""
        config = LoopConfig(
            max_steps=3,
            max_errors=10,
            timeout_s=10.0,
            persist_trajectories=False,
            workspace=str(tmp_workspace),
        )
        model = ScriptedModel([
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Done on the last step"),
        ])
        loop = AgentLoop(model, tool_executor, config=config)
        result = loop.run("list then read")

        assert result.success
        assert result.steps == 3


class TestInfiniteLoopDetection:
    """Verify infinite loop detection — same tool+args 3x = stop."""

    def test_repeated_same_call_detected(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Model calls file_list with same args 3 times — should be stopped."""
        same_args = {"path": str(tmp_workspace)}
        model = ScriptedModel([
            make_tool_response("file_list", arguments=same_args),
            make_tool_response("file_list", arguments=same_args),
            make_tool_response("file_list", arguments=same_args),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("keep listing same dir")

        assert not result.success
        assert result.failure_category == FailureCategory.INFINITE_LOOP

    def test_different_args_not_flagged(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Same tool but different args should NOT trigger infinite loop."""
        model = ScriptedModel([
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_tool_response("file_list", arguments={"path": str(tmp_workspace / "subdir")}),
            make_tool_response("file_list", arguments={"path": "/tmp"}),
            make_text_response("Listed three different dirs"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list multiple dirs")

        assert result.success
        assert result.tool_calls == 3


class TestErrorBudget:
    """Verify error budget enforcement."""

    def test_error_budget_exceeded(
        self,
        tool_executor,
        tmp_workspace,
    ):
        """Model calls unknown tools repeatedly — should hit error budget."""
        # Use different tool names to avoid infinite loop detection
        bad_calls = [
            ModelResponse(
                text=f"trying bad tool {i}",
                model="scripted",
                provider="test",
                tool_calls=[ToolCall(id=f"c{i}", name=f"nonexistent_{i}", arguments={"x": i})],
                finish_reason="tool_calls",
            )
            for i in range(10)
        ]
        config = LoopConfig(
            max_steps=20,
            max_errors=3,
            timeout_s=10.0,
            persist_trajectories=False,
            workspace=str(tmp_workspace),
        )
        model = ScriptedModel(bad_calls)
        loop = AgentLoop(model, tool_executor, config=config)
        result = loop.run("keep calling bad tools")

        assert not result.success
        assert result.failure_category == FailureCategory.ENVIRONMENT_ERROR
