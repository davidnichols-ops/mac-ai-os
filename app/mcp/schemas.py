"""MCP schemas — type definitions for MCP resources and tools.

These are plain dataclasses that describe the shape of what MAOS exposes
through MCP. The actual data assembly happens in the Context API
(app/server/context.py) and the ActionExecutor (app/actions/executor.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPResourceSpec:
    """Specification of an MCP resource exposed by MAOS."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


@dataclass
class MCPToolSpec:
    """Specification of an MCP tool exposed by MAOS."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# --- Resource specifications -----------------------------------------------

CONTEXT_RESOURCE = MCPResourceSpec(
    uri="maos://context/current",
    name="current_context",
    description=(
        "Universal context packet: active workspace, current session, "
        "blackboard state, open tasks, recent activity, recent decisions, "
        "relevant knowledge, and telemetry summary. This is the primary "
        "resource — most queries start here."
    ),
)

WORKSPACE_RESOURCE = MCPResourceSpec(
    uri="maos://workspace/current",
    name="current_workspace",
    description=(
        "Active workspace: name, path, git branch, modified files, "
        "and git status."
    ),
)

SESSION_RESOURCE = MCPResourceSpec(
    uri="maos://session/latest",
    name="latest_session",
    description=(
        "Most recent work session: summary, files changed, commands run, "
        "and any unfinished work."
    ),
)

TASKS_RESOURCE = MCPResourceSpec(
    uri="maos://tasks/open",
    name="open_tasks",
    description=(
        "All pending tasks with their dependencies, priority, and "
        "associated workspace."
    ),
)

KNOWLEDGE_RESOURCE = MCPResourceSpec(
    uri="maos://knowledge/search",
    name="knowledge_search",
    description=(
        "Semantic search over indexed knowledge (vector store + knowledge "
        "graph). Pass a query parameter to search."
    ),
)

ALL_RESOURCES: list[MCPResourceSpec] = [
    CONTEXT_RESOURCE,
    WORKSPACE_RESOURCE,
    SESSION_RESOURCE,
    TASKS_RESOURCE,
    KNOWLEDGE_RESOURCE,
]


# --- Tool specifications ---------------------------------------------------

GET_CONTEXT_TOOL = MCPToolSpec(
    name="maos_get_context",
    description=(
        "Get the full MAOS context packet. Returns workspace, session, "
        "tasks, decisions, knowledge, telemetry, and blackboard state. "
        "This is the equivalent of reading maos://context/current but "
        "as a tool call."
    ),
    parameters={},
)

SEARCH_MEMORY_TOOL = MCPToolSpec(
    name="maos_search_memory",
    description=(
        "Search MAOS memory (vector store) for content matching the query. "
        "Returns ranked results with text, source, and relevance score."
    ),
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "top_k": {
            "type": "integer",
            "description": "Number of results (default 5)",
            "default": 5,
        },
    },
)

WORKSPACE_SUMMARY_TOOL = MCPToolSpec(
    name="maos_workspace_summary",
    description=(
        "Generate a summary of a workspace's current state: git branch, "
        "modified files, pending tasks. Maps to the workspace_summary "
        "action through ActionExecutor."
    ),
    parameters={
        "workspace": {
            "type": "string",
            "description": "Workspace name or path",
        },
    },
)

CREATE_TASK_TOOL = MCPToolSpec(
    name="maos_create_task",
    description=(
        "Create a new TODO task. Maps to the create_todo action through "
        "ActionExecutor (requires tasks.write permission)."
    ),
    parameters={
        "title": {"type": "string", "description": "Task title"},
        "workspace": {
            "type": "string",
            "description": "Associated workspace (optional)",
            "default": "",
        },
        "priority": {
            "type": "integer",
            "description": "Priority (default 0)",
            "default": 0,
        },
    },
)

LIST_TASKS_TOOL = MCPToolSpec(
    name="maos_list_tasks",
    description=(
        "List pending tasks. Maps to the list_tasks action through "
        "ActionExecutor (requires tasks.read permission)."
    ),
    parameters={
        "workspace": {
            "type": "string",
            "description": "Filter by workspace (optional)",
            "default": "",
        },
    },
)

RUN_ACTION_TOOL = MCPToolSpec(
    name="maos_run_action",
    description=(
        "Execute any registered MAOS action by name. This is the generic "
        "gateway to the ActionExecutor. All permission checks, "
        "reversibility checks, action logging, and event publishing "
        "are preserved."
    ),
    parameters={
        "action": {"type": "string", "description": "Action name"},
        "target": {
            "type": "string",
            "description": "Action target (optional)",
            "default": "",
        },
        "parameters": {
            "type": "object",
            "description": "Action parameters as key-value pairs",
            "default": {},
        },
    },
)

ALL_TOOLS: list[MCPToolSpec] = [
    GET_CONTEXT_TOOL,
    SEARCH_MEMORY_TOOL,
    WORKSPACE_SUMMARY_TOOL,
    CREATE_TASK_TOOL,
    LIST_TASKS_TOOL,
    RUN_ACTION_TOOL,
]
