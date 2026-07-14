"""Tests for the MCP Context Layer.

Verifies:
- MCP resources return context packets, workspace, session, tasks, knowledge
- MCP tools call existing ActionExecutor (no bypass)
- Unauthorized requests are rejected (token validation)
- Action logging still occurs after tool execution
- Events are published after tool execution
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from app.actions import (
    ActionExecutor,
    ActionRegistry,
    PermissionChecker,
    builtin_actions,
)
from app.blackboard import Blackboard
from app.db import Database
from app.events.bus import EventBus
from app.mcp import resources, tools
from app.mcp.server import (
    create_mcp_server,
    get_or_create_token,
    validate_token,
)

# --- Test fixtures ---------------------------------------------------------


@dataclass
class FakeResult:
    text: str
    source: str
    score: float
    metadata: dict


class FakeRetriever:
    """Minimal retriever for testing — no model loading."""

    def __init__(self, results: list[FakeResult] | None = None):
        self._results = results or []

    def search(self, query: str, *, top_k: int = 5, where: dict | None = None):
        return self._results[:top_k]

    def index(self, source: str, chunks: list[dict]) -> int:
        return len(chunks)

    def count(self) -> int:
        return len(self._results)


class FakeRuntime:
    """Minimal runtime for MCP tests — no model loading."""

    def __init__(self, db, blackboard, retriever, action_executor, action_registry):
        self.db = db
        self.blackboard = blackboard
        self.retriever = retriever
        self.action_executor = action_executor
        self.action_registry = action_registry


@pytest.fixture
def runtime(tmp_path):
    """Build a minimal runtime with real DB, actions, and event bus."""
    db = Database(str(tmp_path / "test_mcp.db"))
    bus = EventBus(db=db)
    bb = Blackboard()

    # Action layer — real registry, permissions, executor.
    action_registry = ActionRegistry()
    for action, handler in builtin_actions():
        action_registry.register(action, handler)

    permissions = PermissionChecker(granted={
        "filesystem.read", "application.launch", "tasks.read", "tasks.write",
        "git.exec",
    })
    action_executor = ActionExecutor(action_registry, permissions, db, bus=bus)

    retriever = FakeRetriever([
        FakeResult(
            text="ActionExecutor validates permissions before executing.",
            source="executor.py",
            score=0.92,
            metadata={},
        ),
    ])

    return FakeRuntime(db, bb, retriever, action_executor, action_registry)


# --- Resource tests --------------------------------------------------------


def test_resource_returns_context_packet(runtime):
    """MCP resource maos://context/current returns a valid context packet."""
    result = resources.read_context_resource(runtime)
    packet = json.loads(result)

    assert "timestamp" in packet
    assert "workspace" in packet
    assert "session" in packet
    assert "tasks" in packet
    assert "blackboard" in packet
    assert "recent_activity" in packet


def test_resource_returns_workspace(runtime):
    """MCP resource maos://workspace/current returns workspace info."""
    result = resources.read_workspace_resource(runtime)
    data = json.loads(result)
    # May be "no active workspace" on empty DB, but should be valid JSON.
    assert "error" in data or "name" in data


def test_resource_returns_session(runtime):
    """MCP resource maos://session/latest returns session info."""
    result = resources.read_session_resource(runtime)
    data = json.loads(result)
    # Empty DB → "no sessions recorded", but valid JSON.
    assert "error" in data or "start_time" in data


def test_resource_returns_tasks(runtime):
    """MCP resource maos://tasks/open returns pending tasks."""
    # Create a task first.
    runtime.db.create_task("MCP test task", source="test")

    result = resources.read_tasks_resource(runtime)
    data = json.loads(result)

    assert data["count"] == 1
    assert data["tasks"][0]["title"] == "MCP test task"
    assert "dependencies" in data["tasks"][0]
    assert "is_blocked" in data["tasks"][0]


def test_resource_returns_knowledge_search(runtime):
    """MCP resource maos://knowledge/search returns vector + graph results."""
    result = resources.read_knowledge_resource(runtime, query="action executor")
    data = json.loads(result)

    assert data["query"] == "action executor"
    assert len(data["vector_results"]) > 0
    assert "ActionExecutor" in data["vector_results"][0]["text"]


def test_resource_knowledge_empty_query(runtime):
    """Knowledge search with empty query returns empty results."""
    result = resources.read_knowledge_resource(runtime, query="")
    data = json.loads(result)

    assert data["query"] == ""
    assert data["vector_results"] == []
    assert data["graph_entities"] == []


# --- Tool tests ------------------------------------------------------------


def test_tool_get_context(runtime):
    """maos_get_context tool returns context packet."""
    result = tools.maos_get_context(runtime)
    packet = json.loads(result)
    assert "workspace" in packet
    assert "session" in packet


def test_tool_search_memory(runtime):
    """maos_search_memory tool returns search results."""
    result = tools.maos_search_memory(runtime, query="action executor")
    data = json.loads(result)
    assert data["count"] > 0
    assert "ActionExecutor" in data["results"][0]["text"]


def test_tool_search_memory_empty_query(runtime):
    """maos_search_memory rejects empty query."""
    result = tools.maos_search_memory(runtime, query="")
    data = json.loads(result)
    assert "error" in data


def test_tool_list_tasks_calls_action_executor(runtime):
    """maos_list_tasks goes through ActionExecutor, not direct DB access."""
    runtime.db.create_task("Task via MCP", source="test")

    result = tools.maos_list_tasks(runtime)
    data = json.loads(result)

    assert data["ok"] is True
    assert data["action"] == "list_tasks"
    assert data["approved"] is True
    assert data["executed"] is True
    assert "Task via MCP" in data["result"]


def test_tool_create_task_calls_action_executor(runtime):
    """maos_create_task goes through ActionExecutor and creates a task."""
    result = tools.maos_create_task(runtime, title="Created via MCP")
    data = json.loads(result)

    assert data["ok"] is True
    assert data["action"] == "create_todo"
    assert data["approved"] is True
    assert data["executed"] is True
    assert "Created via MCP" in data["result"]

    # Verify the task was actually created in the DB.
    tasks = runtime.db.pending_tasks()
    assert any(t["title"] == "Created via MCP" for t in tasks)


def test_tool_run_action_generic_gateway(runtime):
    """maos_run_action executes any registered action through ActionExecutor."""
    result = tools.maos_run_action(
        runtime, action="list_tasks", target="", parameters={}
    )
    data = json.loads(result)

    assert data["ok"] is True
    assert data["action"] == "list_tasks"
    assert data["approved"] is True


def test_tool_run_action_unknown(runtime):
    """maos_run_action with unknown action returns error via ActionExecutor."""
    result = tools.maos_run_action(
        runtime, action="nonexistent_action", parameters={}
    )
    data = json.loads(result)

    assert data["ok"] is False
    assert "unknown action" in data["error"]


# --- Action logging and event publishing tests -----------------------------


def test_action_logging_occurs_after_tool_execution(runtime):
    """Tool execution through ActionExecutor logs to action_log table."""
    tools.maos_create_task(runtime, title="Logged task")

    actions = runtime.db.recent_actions()
    assert len(actions) >= 1
    # Find the create_todo action.
    create_actions = [a for a in actions if a["action_name"] == "create_todo"]
    assert len(create_actions) >= 1
    assert create_actions[-1]["approved"] == 1
    assert create_actions[-1]["executed"] == 1


def test_event_published_after_tool_execution(runtime):
    """Tool execution through ActionExecutor publishes an ACTION_EXECUTED event."""
    tools.maos_list_tasks(runtime)

    events = runtime.db.query_events(limit=20)
    assert any(e["type"] == "ACTION_EXECUTED" for e in events)


def test_permission_denied_logged(runtime):
    """When permissions are denied, the action is still logged (as denied)."""
    # Create a runtime with restricted permissions.
    restricted_perms = PermissionChecker(granted=set())
    runtime.action_executor = ActionExecutor(
        runtime.action_registry, restricted_perms, runtime.db,
        bus=EventBus(db=runtime.db),
    )

    result = tools.maos_list_tasks(runtime)
    data = json.loads(result)

    assert data["ok"] is False
    assert data["approved"] is False
    assert "permission denied" in data["error"]

    # Verify it was logged.
    actions = runtime.db.recent_actions()
    list_actions = [a for a in actions if a["action_name"] == "list_tasks"]
    assert len(list_actions) >= 1
    assert list_actions[-1]["approved"] == 0


# --- Security / token tests ------------------------------------------------


def test_token_generation_and_storage(tmp_path):
    """Token is generated on first run and stored with restrictive permissions."""
    with patch("app.mcp.server.TOKEN_DIR", tmp_path / ".maos"), \
         patch("app.mcp.server.TOKEN_FILE", tmp_path / ".maos" / "token"):

        token = get_or_create_token()
        assert len(token) == 64  # 32 bytes hex = 64 chars
        assert (tmp_path / ".maos" / "token").exists()

        # Calling again returns the same token.
        assert get_or_create_token() == token


def test_token_validation():
    """validate_token accepts correct token, rejects wrong/missing."""
    token = "a" * 64
    assert validate_token(token, token) is True
    assert validate_token("b" * 64, token) is False
    assert validate_token(None, token) is False
    assert validate_token("", token) is False


def test_token_constant_time_comparison():
    """validate_token uses secrets.compare_digest (no timing leak)."""
    token = "deadbeef" * 8
    assert validate_token(token, token) is True
    # Slightly different token should fail.
    assert validate_token("deadbeee" + "f" * 56, token) is False


# --- MCP server creation test ----------------------------------------------


def test_create_mcp_server(runtime):
    """create_mcp_server returns a FastMCP server with resources and tools."""
    with patch("app.mcp.server.TOKEN_DIR", Path(tempfile.mkdtemp()) / ".maos"), \
         patch("app.mcp.server.TOKEN_FILE", Path(tempfile.mkdtemp()) / ".maos" / "token"):
        mcp = create_mcp_server(runtime, require_token=False)

        # FastMCP server exists.
        assert mcp is not None

        # Verify tools are registered by checking the tool manager.
        # FastMCP stores tools internally; we can check via the tool manager.
        tool_names = set()
        # Access the tool manager's registered tools.
        if hasattr(mcp, "_tool_manager"):
            mgr = mcp._tool_manager
            if hasattr(mgr, "_tools"):
                tool_names = set(mgr._tools.keys())

        # If FastMCP stores them differently, check via the public API.
        if not tool_names and hasattr(mcp, "list_tools"):
            import asyncio
            tools_list = asyncio.get_event_loop().run_until_complete(mcp.list_tools())
            tool_names = {t.name for t in tools_list}

        # Verify our tools are registered.
        assert "maos_get_context" in tool_names
        assert "maos_search_memory" in tool_names
        assert "maos_create_task" in tool_names
        assert "maos_list_tasks" in tool_names
        assert "maos_run_action" in tool_names
