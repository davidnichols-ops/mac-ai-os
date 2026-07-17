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

ONBOARD_TOOL = MCPToolSpec(
    name="maos_onboard",
    description=(
        "Get the onboarding briefing for a new AI session. Returns user "
        "preferences, hard-won lessons, recent session, active workspace, "
        "open tasks, and known failure patterns. This is the FIRST thing "
        "a new session should call before exploring any codebase. Pass "
        "format='json' for structured output."
    ),
    parameters={
        "format": {
            "type": "string",
            "description": "Output format: 'text' (default, readable) or 'json'",
            "default": "text",
        },
    },
)

RECORD_LESSON_TOOL = MCPToolSpec(
    name="maos_record_lesson",
    description=(
        "Record a hard-won lesson in the user model. Lessons are surfaced "
        "in every future onboarding briefing. Use this when something goes "
        "wrong and the session learns a rule that should never be violated "
        "again."
    ),
    parameters={
        "slug": {
            "type": "string",
            "description": "Short snake_case identifier (e.g. 'verify_before_pr')",
        },
        "text": {
            "type": "string",
            "description": "The lesson — what to do or what not to do",
        },
        "confidence": {
            "type": "number",
            "description": "How strongly to hold this (0-1, default 0.9)",
            "default": 0.9,
        },
        "source": {
            "type": "string",
            "description": "Origin: 'incident', 'observed', or 'manual'",
            "default": "incident",
        },
    },
)

LIST_LESSONS_TOOL = MCPToolSpec(
    name="maos_list_lessons",
    description=(
        "List all recorded lessons, sorted by confidence (highest first). "
        "Read-only."
    ),
    parameters={},
)

SEND_EMAIL_TOOL = MCPToolSpec(
    name="maos_send_email",
    description=(
        "Send an email via Gmail (from david.nichols.ops@gmail.com). "
        "Tokens are refreshed automatically. Default recipient is david."
    ),
    parameters={
        "subject": {"type": "string", "description": "Email subject line"},
        "body": {"type": "string", "description": "Plain text body"},
        "to": {
            "type": "string",
            "description": "Recipient (defaults to david.nichols.ops@gmail.com)",
            "default": "",
        },
    },
)

ESCALATE_TOOL = MCPToolSpec(
    name="maos_escalate",
    description=(
        "Send an escalation email — 'this needs human attention'. This is "
        "the anti-vibe-overload channel. When something isn't working, when "
        "a verification fails, when context is getting diluted with false "
        "confidence, escalate instead of glossing over it. Sends to "
        "david.nichols.ops@gmail.com with [MAOS] prefix."
    ),
    parameters={
        "subject": {
            "type": "string",
            "description": "Short summary of what needs attention",
        },
        "body": {
            "type": "string",
            "description": "The detailed explanation of what happened",
        },
        "context": {
            "type": "object",
            "description": "Optional structured context to append",
            "default": {},
        },
        "lesson_slug": {
            "type": "string",
            "description": "Optional lesson slug if related to a recorded lesson",
            "default": "",
        },
    },
)

SCAN_VIBES_TOOL = MCPToolSpec(
    name="maos_scan_vibes",
    description=(
        "Scan text for vibe language — unsupported success claims like "
        "'everything is working', 'verified' without evidence, 'all tests "
        "pass' without output, 'looks good'. Returns PASS/WARN/BLOCK with "
        "flagged phrases. Better a false positive than a false negative."
    ),
    parameters={
        "text": {
            "type": "string",
            "description": "The text to scan (agent output, PR description, commit message)",
        },
    },
)

CHECK_GUARDS_TOOL = MCPToolSpec(
    name="maos_check_guards",
    description=(
        "Run all guards against a text + action context. This is the "
        "comprehensive pre-action check. Run it before opening a PR, "
        "committing a fix, or declaring success. If any guard returns "
        "BLOCK, do NOT proceed. Guards: vibe language, PR pre-flight, "
        "lesson compliance, sub-agent proof check."
    ),
    parameters={
        "text": {
            "type": "string",
            "description": "The text to scan for vibes",
        },
        "action_description": {
            "type": "string",
            "description": "What the agent is about to do (for lesson compliance)",
            "default": "",
        },
        "is_pr": {
            "type": "boolean",
            "description": "Whether this is a PR-related action (triggers pre-flight)",
            "default": False,
        },
        "is_bug_fix": {
            "type": "boolean",
            "description": "Whether this is a bug fix (requires manual verification)",
            "default": False,
        },
        "has_test_output": {
            "type": "boolean",
            "description": "Whether test output is provided",
            "default": False,
        },
        "has_build_output": {
            "type": "boolean",
            "description": "Whether build output is provided",
            "default": False,
        },
        "has_manual_verification": {
            "type": "boolean",
            "description": "Whether manual verification against real data was done",
            "default": False,
        },
        "claims_subagent_agreement": {
            "type": "boolean",
            "description": "Whether sub-agent agreement is claimed as evidence",
            "default": False,
        },
        "has_empirical_evidence": {
            "type": "boolean",
            "description": "Whether empirical evidence (command output, test results) is provided",
            "default": False,
        },
    },
)

DIAGNOSTICS_TOOL = MCPToolSpec(
    name="maos_diagnostics",
    description=(
        "Check functionality and implementation status of all MAOS MCP "
        "tools. Actually calls each read-only tool and verifies it returns "
        "valid JSON. Side-effect tools are checked for handler existence. "
        "Returns per-tool status: working, degraded, broken, skipped. "
        "Use this to verify the MAOS system is healthy before relying on it."
    ),
    parameters={
        "format": {
            "type": "string",
            "description": "Output format: 'text' (default, readable) or 'json'",
            "default": "text",
        },
    },
)

TIME_TOOL = MCPToolSpec(
    name="maos_time",
    description=(
        "Get time awareness information for the current AI session. AI "
        "agents have no native concept of time — this tool gives the agent "
        "current wall-clock time, session elapsed time (how long this "
        "Devin session has been running), session start time, tool call "
        "count and frequency, idle time, and relative time hints. A "
        "'session' is a contiguous sequence of MCP tool calls with no gap "
        "longer than 10 minutes. Call this to know how long you've been "
        "working on a task."
    ),
    parameters={
        "format": {
            "type": "string",
            "description": "Output format: 'text' (default, readable) or 'json'",
            "default": "text",
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
    ONBOARD_TOOL,
    RECORD_LESSON_TOOL,
    LIST_LESSONS_TOOL,
    SEND_EMAIL_TOOL,
    ESCALATE_TOOL,
    SCAN_VIBES_TOOL,
    CHECK_GUARDS_TOOL,
    DIAGNOSTICS_TOOL,
    TIME_TOOL,
]
