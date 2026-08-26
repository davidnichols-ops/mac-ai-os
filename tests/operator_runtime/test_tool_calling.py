"""Tool calling tests — model discovers tools, calls correct tool,
invalid tools rejected, malformed arguments handled."""

from __future__ import annotations

from app.core.agent_loop import AgentLoop
from app.core.tool_executor import ToolExecutor
from app.models.types import ModelResponse, ToolCall
from app.tools.registry import ToolRouter
from tests.operator_runtime.conftest import (
    ScriptedModel,
    make_text_response,
    make_tool_response,
)


class TestToolDiscovery:
    """Verify the model receives tool definitions and can call them."""

    def test_tools_injected_into_system_prompt(self, tool_router: ToolRouter):
        """Tool schemas should be available to inject into the prompt."""
        schemas = tool_router.schemas()
        assert len(schemas) >= 6  # shell, file_read, file_write, file_edit, file_list, git
        names = {s["name"] for s in schemas}
        assert "shell" in names
        assert "file_read" in names
        assert "file_list" in names

    def test_model_calls_file_list_and_completes(
        self,
        tool_executor: ToolExecutor,
        loop_config,
        tmp_workspace,
    ):
        """Model calls file_list, gets result, then gives final answer."""
        model = ScriptedModel([
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_text_response("Found hello.py and data.txt"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list files")
        assert result.success
        assert result.tool_calls == 1
        assert "hello.py" in result.answer or "Found" in result.answer

    def test_model_calls_file_read_and_completes(
        self,
        tool_executor: ToolExecutor,
        loop_config,
        tmp_workspace,
    ):
        """Model reads a file and reports its contents."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("The file contains print('hello world')"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")
        assert result.success
        assert result.tool_calls == 1


class TestInvalidToolRejection:
    """Verify invalid tool names are rejected gracefully."""

    def test_unknown_tool_returns_error(
        self,
        tool_executor: ToolExecutor,
        loop_config,
    ):
        """Model calls a non-existent tool — should get error, not crash."""
        model = ScriptedModel([
            ModelResponse(
                text="calling unknown tool",
                model="scripted",
                provider="test",
                tool_calls=[ToolCall(id="c1", name="nonexistent_tool", arguments={})],
                finish_reason="tool_calls",
            ),
            make_text_response("Tool not found, but I'll answer anyway"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("use unknown tool")
        assert result.success  # loop continues after error, model gives final answer
        assert result.tool_calls == 1
        # The trajectory should record the error
        tool_result_steps = [
            s for s in result.trajectory.steps
            if s.step_type.value == "tool_result" and not s.tool_ok
        ]
        assert len(tool_result_steps) >= 1

    def test_model_recovers_from_unknown_tool(
        self,
        tool_executor: ToolExecutor,
        loop_config,
        tmp_workspace,
    ):
        """Model calls unknown tool, gets error, then calls correct tool."""
        model = ScriptedModel([
            ModelResponse(
                text="trying something",
                model="scripted",
                provider="test",
                tool_calls=[ToolCall(id="c1", name="bad_tool", arguments={})],
                finish_reason="tool_calls",
            ),
            make_tool_response("file_list", arguments={"path": str(tmp_workspace)}),
            make_text_response("Found the files after switching tools"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list files")
        assert result.success
        assert result.tool_calls == 2  # one failed, one succeeded


class TestMalformedArguments:
    """Verify malformed arguments are handled gracefully."""

    def test_missing_required_argument(
        self,
        tool_executor: ToolExecutor,
        loop_config,
        tmp_workspace,
    ):
        """file_read without path argument should return error."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={}),  # missing 'path'
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Got the file content after fixing args"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")
        assert result.success
        assert result.tool_calls == 2  # first failed, second succeeded

    def test_wrong_argument_type(
        self,
        tool_executor: ToolExecutor,
        loop_config,
        tmp_workspace,
    ):
        """file_read with non-string path should not crash."""
        model = ScriptedModel([
            make_tool_response("file_read", arguments={"path": 12345}),
            make_tool_response("file_read", arguments={"path": str(tmp_workspace / "hello.py")}),
            make_text_response("Fixed the path type"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("read hello.py")
        assert result.success

    def test_empty_arguments_dict(
        self,
        tool_executor: ToolExecutor,
        loop_config,
    ):
        """Calling a tool with empty args dict should not crash."""
        model = ScriptedModel([
            make_tool_response("file_list", arguments={}),  # path defaults to "."
            make_text_response("Listed the current directory"),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("list files")
        assert result.success
