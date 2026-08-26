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

TIME_RESOURCE = MCPResourceSpec(
    uri="maos://time/current",
    name="current_time",
    description=(
        "Time awareness for the current AI session: current wall-clock "
        "time, session elapsed time, tool call count, idle time, and "
        "relative time hints."
    ),
)

ALL_RESOURCES: list[MCPResourceSpec] = [
    CONTEXT_RESOURCE,
    WORKSPACE_RESOURCE,
    SESSION_RESOURCE,
    TASKS_RESOURCE,
    KNOWLEDGE_RESOURCE,
    TIME_RESOURCE,
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

RECORD_DECISION_TOOL = MCPToolSpec(
    name="maos_record_decision",
    description=(
        "Record a structured agent decision to the hash-chained audit log. "
        "Each decision is linked to the previous one via a SHA-256 digest "
        "chain, making the log tamper-evident. Use this to make agent "
        "reasoning auditable and reproducible."
    ),
    parameters={
        "agent": {
            "type": "string",
            "description": "Identifier of the agent making the decision (e.g. 'devin', 'claude-3').",
        },
        "action": {
            "type": "string",
            "description": "The action being decided on (e.g. 'commit', 'deploy', 'merge').",
        },
        "rationale": {
            "type": "string",
            "description": "Why the decision was made. This is the auditable reasoning trail.",
        },
        "kind": {
            "type": "string",
            "description": "Decision category: 'decision', 'observation', 'escalation', or 'veto'.",
            "default": "decision",
        },
        "tags": {
            "type": "string",
            "description": "Comma-separated free-form tags for grouping decisions.",
            "default": "",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence level 0-1 (default 0).",
            "default": 0,
        },
    },
)

QUERY_DECISIONS_TOOL = MCPToolSpec(
    name="maos_query_decisions",
    description=(
        "Query the decision audit log. Read-only and filtered. Returns "
        "matching decisions with digests and chain pointers. Does not "
        "include rationale (data minimization)."
    ),
    parameters={
        "agent": {
            "type": "string",
            "description": "Filter to decisions by this agent only.",
            "default": "",
        },
        "kind": {
            "type": "string",
            "description": "Filter to decisions of this kind only.",
            "default": "",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of decisions to return (default 100, max 1000).",
            "default": 100,
        },
    },
)

VERIFY_DECISIONS_TOOL = MCPToolSpec(
    name="maos_verify_decisions",
    description=(
        "Verify the integrity of the decision hash chain. Checks that "
        "every decision's digest matches its recomputed value and that "
        "chain links are intact. Returns valid=True if tamper-free."
    ),
    parameters={},
)

SEND_EMAIL_TOOL = MCPToolSpec(
    name="maos_send_email",
    description=(
        "Send an email via Gmail. "
        "Tokens are refreshed automatically. "
        "Default recipient is read from MAOS_EMAIL_RECIPIENT or the 'to' argument."
    ),
    parameters={
        "subject": {"type": "string", "description": "Email subject line"},
        "body": {"type": "string", "description": "Plain text body"},
        "to": {
            "type": "string",
            "description": "Recipient email address",
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
        "confidence, escalate instead of glossing over it. Sends to the "
        "configured MAOS email recipient with a [MAOS] prefix."
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

# --- Google Drive tools -----------------------------------------------------

DRIVE_LIST_TOOL = MCPToolSpec(
    name="maos_drive_list",
    description=(
        "List files in a Google Drive folder. "
        "Returns file id, name, mimeType, size, modifiedTime for each file. "
        "Use folder_id='root' or omit for the root folder."
    ),
    parameters={
        "folder_id": {
            "type": "string",
            "description": "Folder ID to list (default: root)",
            "default": "",
        },
        "page_size": {
            "type": "integer",
            "description": "Max files to return (default 50, max 200)",
            "default": 50,
        },
        "query": {
            "type": "string",
            "description": "Optional Drive query string (e.g. \"name contains 'report'\")",
            "default": "",
        },
    },
)

DRIVE_SEARCH_TOOL = MCPToolSpec(
    name="maos_drive_search",
    description=(
        "Search Google Drive by name or content query. "
        "Returns matching files with id, name, mimeType, size."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Search query (e.g. \"name contains 'report'\" or full-text search terms)",
        },
        "page_size": {
            "type": "integer",
            "description": "Max results (default 20)",
            "default": 20,
        },
    },
)

DRIVE_UPLOAD_TOOL = MCPToolSpec(
    name="maos_drive_upload",
    description=(
        "Upload a file to Google Drive (5TB Workspace account). "
        "Creates or overwrites a file in the specified folder. "
        "Returns the file id and metadata."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Remote file name on Drive",
        },
        "content_b64": {
            "type": "string",
            "description": "File content as base64-encoded string",
        },
        "mime_type": {
            "type": "string",
            "description": "MIME type (default: application/octet-stream)",
            "default": "application/octet-stream",
        },
        "parent_folder_id": {
            "type": "string",
            "description": "Folder ID to upload into (default: root)",
            "default": "",
        },
    },
)

DRIVE_DOWNLOAD_TOOL = MCPToolSpec(
    name="maos_drive_download",
    description=(
        "Download a file from Google Drive by its file ID. "
        "Returns the file content as base64-encoded string."
    ),
    parameters={
        "file_id": {
            "type": "string",
            "description": "Google Drive file ID",
        },
    },
)

DRIVE_MKDIR_TOOL = MCPToolSpec(
    name="maos_drive_mkdir",
    description=(
        "Create a folder on Google Drive (or get existing if it already exists). "
        "Returns the folder id and metadata."
    ),
    parameters={
        "name": {
            "type": "string",
            "description": "Folder name to create",
        },
        "parent_folder_id": {
            "type": "string",
            "description": "Parent folder ID (default: root)",
            "default": "",
        },
    },
)

DRIVE_QUOTA_TOOL = MCPToolSpec(
    name="maos_drive_quota",
    description=(
        "Get Google Drive storage quota info for the configured account. "
        "Returns usage, limit, and user email."
    ),
    parameters={},
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

READ_FILE_TOOL = MCPToolSpec(
    name="maos_read_file",
    description=(
        "Read a file with time awareness and reading behavior tracking. "
        "Returns file content PLUS a _time_awareness section showing "
        "session elapsed, tool call count, files read, re-read ratio, "
        "and a behavioral assessment. If the re-read ratio is >50%, a "
        "_warning is added. Reading is the #1 repeated action for AI "
        "agents — by routing reads through this tool, every read "
        "becomes a time check and behavioral checkpoint. Use this "
        "instead of the built-in read tool to stay aware of how long "
        "you've been working and whether you're going in circles."
    ),
    parameters={
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the file to read.",
        },
        "max_lines": {
            "type": "integer",
            "description": "Maximum lines to read (default 2000).",
            "default": 2000,
        },
    },
)

VERIFY_COMMAND_TOOL = MCPToolSpec(
    name="maos_verify_command",
    description=(
        "Execute a verification command and capture evidence (exit code, "
        "stdout, stderr, git commit, timestamp). Evidence-PRODUCING "
        "verification — the system runs the command, not the agent. The "
        "agent cannot fake the exit code, stdout, or commit hash. Use this "
        "instead of self-attested booleans like has_test_output=True. "
        "SECURITY: Only verification commands are allowed (test runners, "
        "linters, type checkers, build commands). Allowed prefixes: uv run "
        "pytest, uv run ruff, uv run mypy, pytest, ruff, mypy, npm test, "
        "npm run lint, cargo test, cargo clippy, go test, make test, git "
        "status. Capture test evidence with command='uv run pytest', lint "
        "evidence with command='uv run ruff check app tests'. Pass the "
        "returned JSON to maos_pr_readiness."
    ),
    parameters={
        "command": {
            "type": "string",
            "description": "The verification command to execute (must be in the allowlist).",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory (defaults to current directory).",
            "default": "",
        },
        "timeout": {
            "type": "integer",
            "description": "Maximum execution time in seconds.",
            "default": 300,
        },
    },
)

INSPECT_DIFF_TOOL = MCPToolSpec(
    name="maos_inspect_diff",
    description=(
        "Inspect the git diff for risk patterns. BLOCK for secrets/credentials. "
        "WARN for debug statements, dependency changes, binary files, scope "
        "violations, large diffs. REQUIRES_REVIEW for MCP tool/schema changes "
        "and security-sensitive paths. Run this before opening a PR."
    ),
    parameters={
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (defaults to cwd).",
            "default": "",
        },
        "base": {
            "type": "string",
            "description": "Base ref to diff against.",
            "default": "HEAD",
        },
        "stated_scope": {
            "type": "string",
            "description": "The stated scope of the PR (for unrelated-file detection).",
            "default": "",
        },
    },
)

PR_READINESS_TOOL = MCPToolSpec(
    name="maos_pr_readiness",
    description=(
        "Assess PR readiness via state machine: DRAFT -> UNDERSTOOD -> "
        "VERIFIED -> PR_READY. The comprehensive readiness check. Requires "
        "evidence objects (from maos_verify_command), a structured PR "
        "description, and inspects the diff. If state is NOT PR_READY, do "
        "NOT open the PR. PR description must have: Summary, Why, What "
        "Changed, Scope. Bug fixes also need: Root Cause, Regression Test, "
        "Before, After."
    ),
    parameters={
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (defaults to cwd).",
            "default": "",
        },
        "pr_description": {
            "type": "string",
            "description": "The PR description text (markdown with required sections).",
            "default": "",
        },
        "is_bug_fix": {
            "type": "boolean",
            "description": "Whether this is a bug fix (requires manual verification).",
            "default": False,
        },
        "test_evidence_json": {
            "type": "string",
            "description": "JSON evidence from maos_verify_command (tests).",
            "default": "",
        },
        "lint_evidence_json": {
            "type": "string",
            "description": "JSON evidence from maos_verify_command (lint).",
            "default": "",
        },
        "build_evidence_json": {
            "type": "string",
            "description": "JSON evidence from maos_verify_command (build).",
            "default": "",
        },
        "manual_evidence_json": {
            "type": "string",
            "description": "JSON manual verification evidence.",
            "default": "",
        },
        "diff_base": {
            "type": "string",
            "description": "Base ref for diff inspection.",
            "default": "HEAD",
        },
        "skip_diff": {
            "type": "boolean",
            "description": "Skip diff inspection.",
            "default": False,
        },
    },
)

# --- Storage management tool specifications ---------------------------------

STORAGE_SCAN_TOOL = MCPToolSpec(
    name="maos_storage_scan",
    description=(
        "Run a comprehensive storage analysis on the Mac. Decomposes the "
        "mysterious macOS 'System Data' category into identifiable, "
        "reclaimable components. Combines native macOS introspection "
        "(tmutil, diskutil, /private/var/vm, unified log archive, Spotlight, "
        "APFS snapshots), X-MaC disk + clean engines, NeoDisk scan cache "
        "detection, and user-managed cache scan."
    ),
    parameters={
        "mode": {
            "type": "string",
            "description": "Scan mode: full (default), fast, system, or caches.",
            "default": "full",
        },
        "format": {
            "type": "string",
            "description": "Output format: text or json.",
            "default": "text",
        },
    },
)

STORAGE_SYSTEM_DATA_TOOL = MCPToolSpec(
    name="maos_storage_system_data",
    description=(
        "Decompose macOS 'System Data' into identifiable components. The "
        "focused tool for solving the '120 GB of system data' mystery. "
        "Checks sleepimage, swapfiles, unified log archive, system logs, "
        "Spotlight index, asset cache, Time Machine snapshots, APFS "
        "snapshots, per-user temp files, and system databases."
    ),
    parameters={
        "format": {
            "type": "string",
            "description": "Output format: text or json.",
            "default": "text",
        },
    },
)

STORAGE_FIX_SCRIPT_TOOL = MCPToolSpec(
    name="maos_storage_fix_script",
    description=(
        "Generate a reviewable remediation shell script for storage cleanup. "
        "ALL destructive commands are COMMENTED OUT. The user reviews, "
        "uncomments what they want, and runs it. NEVER executes destructive "
        "commands — only produces the script."
    ),
    parameters={
        "output_path": {
            "type": "string",
            "description": "Where to write the fix script.",
            "default": "./storage-fixes.sh",
        },
        "include_caches": {
            "type": "boolean",
            "description": "Include user cache cleanup commands.",
            "default": True,
        },
        "include_system": {
            "type": "boolean",
            "description": "Include system data cleanup commands.",
            "default": True,
        },
        "include_tm": {
            "type": "boolean",
            "description": "Include Time Machine snapshot deletion commands.",
            "default": True,
        },
    },
)

STORAGE_MAINTAIN_TOOL = MCPToolSpec(
    name="maos_storage_maintain",
    description=(
        "Run safe macOS maintenance tasks via X-MaC: flush DNS, reindex "
        "Spotlight, rebuild LaunchServices, run periodic scripts, purge RAM, "
        "clear Quick Look. Safe tasks run automatically; sudo tasks emitted "
        "as findings."
    ),
    parameters={},
)

STORAGE_DELETE_TM_SNAPSHOT_TOOL = MCPToolSpec(
    name="maos_storage_delete_tm_snapshot",
    description=(
        "Delete a Time Machine local snapshot by date. The #1 way to "
        "reclaim 'System Data' space. Local snapshots pin purgeable space "
        "that only clears on the next backup. Use maos_storage_scan first "
        "to list available snapshots."
    ),
    parameters={
        "date": {
            "type": "string",
            "description": "Snapshot date string, e.g. '2026-07-24-152803'.",
        },
    },
)

STORAGE_HIDDEN_TOOL = MCPToolSpec(
    name="maos_storage_hidden",
    description=(
        "Scan for hidden data consumers that macOS categorizes as 'System "
        "Data'. Goes beyond standard system data decomposition and user "
        "caches. Finds hidden consumers in non-obvious locations: Devin CLI "
        "session database (3GB+), AI/ML tool data, developer tool data, app "
        "support directories, cloud storage sync, SDKs, app containers, and "
        "system-level data."
    ),
    parameters={
        "format": {
            "type": "string",
            "description": "Output format: text or json.",
            "default": "text",
        },
    },
)


# --- Coding Agent Tools (Devin CLI parity) ----------------------------------
# These mirror Devin CLI's built-in tools so MAOS can act as a full coding
# agent when accessed via MCP by external clients (Grok web, etc.).


WRITE_FILE_TOOL = MCPToolSpec(
    name="maos_write_file",
    description=(
        "Write content to a file, creating it if needed. Creates parent "
        "directories if they don't exist. Overwrites existing files. "
        "Equivalent to Devin CLI's write tool."
    ),
    parameters={
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the file to write.",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file.",
        },
    },
)

EDIT_FILE_TOOL = MCPToolSpec(
    name="maos_edit_file",
    description=(
        "Edit a file by replacing old_string with new_string. Performs "
        "exact string replacement. The old_string must be unique in the "
        "file unless replace_all=True. Equivalent to Devin CLI's edit tool."
    ),
    parameters={
        "path": {
            "type": "string",
            "description": "Absolute or relative path to the file to edit.",
        },
        "old_string": {
            "type": "string",
            "description": "The exact text to replace.",
        },
        "new_string": {
            "type": "string",
            "description": "The text to replace it with.",
        },
        "replace_all": {
            "type": "boolean",
            "description": "If True, replace all occurrences. Default False.",
            "default": False,
        },
    },
)

GREP_TOOL = MCPToolSpec(
    name="maos_grep",
    description=(
        "Search file contents using ripgrep. Returns matching lines with "
        "file paths and line numbers. Equivalent to Devin CLI's grep tool."
    ),
    parameters={
        "pattern": {
            "type": "string",
            "description": "Regular expression to search for.",
        },
        "path": {
            "type": "string",
            "description": "Directory or file to search in (default current directory).",
            "default": ".",
        },
        "glob_pattern": {
            "type": "string",
            "description": "File pattern filter (e.g. '*.py').",
            "default": "",
        },
        "case_insensitive": {
            "type": "boolean",
            "description": "Case-insensitive matching (default False).",
            "default": False,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 100).",
            "default": 100,
        },
        "context_lines": {
            "type": "integer",
            "description": "Lines of context before/after each match (default 0).",
            "default": 0,
        },
    },
)

GLOB_TOOL = MCPToolSpec(
    name="maos_glob",
    description=(
        "Find files by glob pattern. Supports brace expansion (e.g. "
        "'**/*.{ts,tsx}'). Equivalent to Devin CLI's glob tool."
    ),
    parameters={
        "pattern": {
            "type": "string",
            "description": "Glob pattern (e.g. '**/*.py', '*.json').",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (default current directory).",
            "default": ".",
        },
    },
)

FIND_FILE_TOOL = MCPToolSpec(
    name="maos_find_file",
    description=(
        "Find files by name pattern using recursive glob matching. "
        "Equivalent to Devin CLI's find_file_by_name tool."
    ),
    parameters={
        "pattern": {
            "type": "string",
            "description": "File name pattern (e.g. '*.py', 'test_*').",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in (default current directory).",
            "default": ".",
        },
    },
)

RUN_COMMAND_TOOL = MCPToolSpec(
    name="maos_run_command",
    description=(
        "Execute a shell command and return stdout, stderr, and exit code. "
        "Blocked commands include rm -rf /, mkfs, fork bombs, chmod -R 777 /. "
        "Restricted commands (sudo, kill -9, shutdown, reboot, launchctl "
        "unload) are rejected. Equivalent to Devin CLI's exec tool."
    ),
    parameters={
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory (default current directory).",
            "default": "",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 120).",
            "default": 120,
        },
        "env": {
            "type": "object",
            "description": "Additional environment variables as key-value pairs.",
            "default": {},
        },
    },
)

WEB_SEARCH_TOOL = MCPToolSpec(
    name="maos_web_search",
    description=(
        "Search the web for information. Returns relevant web page titles, "
        "URLs, and summaries. Equivalent to Devin CLI's web_search tool."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Search query string.",
        },
        "num_results": {
            "type": "integer",
            "description": "Maximum results to return (default 5).",
            "default": 5,
        },
    },
)

FETCH_URL_TOOL = MCPToolSpec(
    name="maos_fetch_url",
    description=(
        "Fetch a web page and return its content as readable text. "
        "Follows redirects. Equivalent to Devin CLI's webfetch tool."
    ),
    parameters={
        "url": {
            "type": "string",
            "description": "The URL to fetch.",
        },
    },
)

NOTEBOOK_READ_TOOL = MCPToolSpec(
    name="maos_notebook_read",
    description=(
        "Read a Jupyter notebook (.ipynb) and extract all cells with "
        "outputs. Returns cell content, type, and execution outputs in "
        "a structured format. Equivalent to Devin CLI's notebook_read tool."
    ),
    parameters={
        "path": {
            "type": "string",
            "description": "Path to the .ipynb file.",
        },
    },
)

NOTEBOOK_EDIT_TOOL = MCPToolSpec(
    name="maos_notebook_edit",
    description=(
        "Edit a cell in a Jupyter notebook. Supports replace, insert, and "
        "delete modes. Equivalent to Devin CLI's notebook_edit tool."
    ),
    parameters={
        "path": {
            "type": "string",
            "description": "Path to the .ipynb file.",
        },
        "cell_number": {
            "type": "integer",
            "description": "0-based index of the cell to edit.",
        },
        "new_source": {
            "type": "string",
            "description": "New source content for the cell.",
        },
        "cell_type": {
            "type": "string",
            "description": "Cell type: 'code', 'markdown', or 'raw' (for insert).",
            "default": "code",
        },
        "edit_mode": {
            "type": "string",
            "description": "Edit mode: 'replace' (default), 'insert', or 'delete'.",
            "default": "replace",
        },
    },
)

GIT_STATUS_TOOL = MCPToolSpec(
    name="maos_git_status",
    description=(
        "Get git status for a repository. Returns branch, staged changes, "
        "unstaged changes, and untracked files."
    ),
    parameters={
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (default current directory).",
            "default": "",
        },
    },
)

GIT_DIFF_TOOL = MCPToolSpec(
    name="maos_git_diff",
    description=(
        "Get git diff for a repository. Can show staged or unstaged "
        "changes, or compare between refs."
    ),
    parameters={
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (default current directory).",
            "default": "",
        },
        "staged": {
            "type": "boolean",
            "description": "If True, show staged changes (default False).",
            "default": False,
        },
        "base": {
            "type": "string",
            "description": "Base ref for comparison (e.g. 'main', 'HEAD~1').",
            "default": "",
        },
        "target": {
            "type": "string",
            "description": "Target ref for comparison. If base is set, shows base...target.",
            "default": "",
        },
    },
)

GIT_COMMIT_TOOL = MCPToolSpec(
    name="maos_git_commit",
    description=(
        "Stage files and create a git commit. Optionally stages all "
        "changes (git add -A) before committing."
    ),
    parameters={
        "message": {
            "type": "string",
            "description": "Commit message.",
        },
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (default current directory).",
            "default": "",
        },
        "add_all": {
            "type": "boolean",
            "description": "If True, stage all changes (git add -A) before committing.",
            "default": False,
        },
    },
)

GIT_LOG_TOOL = MCPToolSpec(
    name="maos_git_log",
    description=(
        "Get git log for a repository. Returns recent commits in "
        "oneline or detailed format."
    ),
    parameters={
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (default current directory).",
            "default": "",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of commits to return (default 20).",
            "default": 20,
        },
        "oneline": {
            "type": "boolean",
            "description": "If True, use --oneline format (default True).",
            "default": True,
        },
    },
)

GIT_BRANCH_TOOL = MCPToolSpec(
    name="maos_git_branch",
    description=(
        "Manage git branches. Actions: list (default), create, switch, "
        "delete."
    ),
    parameters={
        "repo_dir": {
            "type": "string",
            "description": "Path to the git repository (default current directory).",
            "default": "",
        },
        "action": {
            "type": "string",
            "description": "Branch action: 'list', 'create', 'switch', or 'delete'.",
            "default": "list",
        },
        "branch_name": {
            "type": "string",
            "description": "Branch name for create/switch/delete actions.",
            "default": "",
        },
    },
)

RUN_SUBAGENT_TOOL = MCPToolSpec(
    name="maos_run_subagent",
    description=(
        "Spawn a subagent to handle a task autonomously. The spec flag is "
        "MANDATORY — it determines which AI system to route through. "
        "Supported specs: devin (GLM-5.2, deepest MCP), grokhack (Grok, "
        "security research), claude-code (OpenRouter, model-agnostic), "
        "local (Phi-3.5-mini, private), openrouter (any model)."
    ),
    parameters={
        "task": {
            "type": "string",
            "description": "The task or prompt to give the subagent.",
        },
        "spec": {
            "type": "string",
            "description": (
                "MANDATORY. Which system to route to: 'devin' (Devin+GLM-5.2), "
                "'grokhack' (Grok+security personas), 'claude-code' (Claude Code "
                "via OpenRouter, model-agnostic), 'local' (MAOS local models, "
                "private), 'openrouter' (any OpenRouter model)."
            ),
        },
        "title": {
            "type": "string",
            "description": "Short title for the subagent (optional).",
            "default": "",
        },
        "profile": {
            "type": "string",
            "description": "Subagent profile: 'general' (default) or 'explore' (read-only).",
            "default": "general",
        },
        "is_background": {
            "type": "boolean",
            "description": "If True, run in background (default False).",
            "default": False,
        },
    },
)

TODO_WRITE_TOOL = MCPToolSpec(
    name="maos_todo_write",
    description=(
        "Create or update a structured todo list for the current session. "
        "Each todo has a content description and status (pending, "
        "in_progress, completed). Calling this replaces the entire todo "
        "list. Equivalent to Devin CLI's todo_write tool."
    ),
    parameters={
        "todos": {
            "type": "array",
            "description": "List of todo items, each with 'content' and 'status' keys.",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                },
            },
            "default": [],
        },
    },
)

ASK_USER_TOOL = MCPToolSpec(
    name="maos_ask_user",
    description=(
        "Present questions to the user and collect their answers. For "
        "HTTP/non-interactive sessions, returns a prompt that the calling "
        "client should present to the user. Equivalent to Devin CLI's "
        "ask_user_question tool."
    ),
    parameters={
        "questions": {
            "type": "array",
            "description": "List of question objects with 'question', 'header', and optional 'options'.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "header": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "multi_select": {"type": "boolean"},
                },
            },
            "default": [],
        },
    },
)

# --- Slack tools -------------------------------------------------------------
# These let any MCP-connected agent (local or cloud) read and send Slack
# messages, list channels, search history, and trigger incremental syncs.
# Tokens are loaded from the MAOS SlackConfig (env vars), never hardcoded.


SLACK_SEND_TOOL = MCPToolSpec(
    name="maos_slack_send",
    description=(
        "Send a message to a Slack channel or DM. Uses the bot token "
        "(chat.postMessage). The channel can be a channel ID (e.g. "
        "C0BMSQGTCJV), a channel name (e.g. #maos-ops), or a user ID for "
        "a DM. Optionally reply in a thread by providing thread_ts."
    ),
    parameters={
        "channel": {
            "type": "string",
            "description": "Channel ID, channel name (e.g. #general), or user ID for DM.",
        },
        "message": {
            "type": "string",
            "description": "The message text to send (Slack mrkdwn supported).",
        },
        "thread_ts": {
            "type": "string",
            "description": "Optional timestamp of the parent message to reply in a thread.",
            "default": "",
        },
    },
)

SLACK_READ_TOOL = MCPToolSpec(
    name="maos_slack_read",
    description=(
        "Read recent messages from a Slack channel. Uses the bot token "
        "(conversations.history). Returns messages as JSON with text, "
        "user, ts, and thread info. Use maos_slack_channels to find "
        "channel IDs."
    ),
    parameters={
        "channel": {
            "type": "string",
            "description": "Channel ID or name to read from.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum messages to return (default 50, max 200).",
            "default": 50,
        },
        "oldest": {
            "type": "string",
            "description": "Optional Unix timestamp — only return messages after this time.",
            "default": "",
        },
    },
)

SLACK_CHANNELS_TOOL = MCPToolSpec(
    name="maos_slack_channels",
    description=(
        "List all visible Slack channels. Uses the bot token "
        "(conversations.list). Returns channel names, IDs, and type "
        "(public, private, im, mpim). Use this to find channel IDs for "
        "maos_slack_send and maos_slack_read."
    ),
    parameters={
        "types": {
            "type": "string",
            "description": (
                "Comma-separated channel types to list: public_channel, "
                "private_channel, im, mpim (default: "
                "public_channel,private_channel)."
            ),
            "default": "public_channel,private_channel",
        },
    },
)

SLACK_SEARCH_TOOL = MCPToolSpec(
    name="maos_slack_search",
    description=(
        "Search Slack messages across the workspace. Uses the user token "
        "(search.messages) — requires SLACK_USER_TOKEN. Returns matching "
        "messages with channel, author, timestamp, and permalink."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Search query (Slack search syntax supported).",
        },
        "count": {
            "type": "integer",
            "description": "Maximum results to return (default 20, max 100).",
            "default": 20,
        },
    },
)

SLACK_SYNC_TOOL = MCPToolSpec(
    name="maos_slack_sync",
    description=(
        "Trigger an incremental Slack history sync into MAOS. Pulls "
        "recent messages from visible channels into the local SQLite "
        "store and refreshes the RAG index so messages become searchable "
        "via maos_search_memory. Uses the user token. Returns sync "
        "counts (channels, messages, users)."
    ),
    parameters={
        "days": {
            "type": "integer",
            "description": "How many days of history to sync (default 30).",
            "default": 30,
        },
    },
)

# --- Project sync tools ------------------------------------------------------
# These let cloud agents manage 2-way project sync state (local ~/Projects/
# <-> GDrive with blockchain evidence) via the project-sync CLI.


PROJECT_SYNC_STATUS_TOOL = MCPToolSpec(
    name="maos_project_sync_status",
    description=(
        "Show sync state for all projects. Calls the project-sync CLI "
        "to display which projects are in sync, which have local changes "
        "to push, and which have pending forks from cloud AI agents."
    ),
    parameters={},
)

PROJECT_SYNC_PUSH_TOOL = MCPToolSpec(
    name="maos_project_sync_push",
    description=(
        "Push a project to GDrive and record blockchain evidence. "
        "Uploads the current local state of the project to Google Drive "
        "and creates a hash-chained evidence entry. Use this after making "
        "local changes that should be propagated to the cloud."
    ),
    parameters={
        "project": {
            "type": "string",
            "description": "Project name in ~/Projects/.",
        },
        "agent": {
            "type": "string",
            "description": "Agent name performing the push (default 'cloud').",
            "default": "cloud",
        },
    },
)

PROJECT_SYNC_PULL_TOOL = MCPToolSpec(
    name="maos_project_sync_pull",
    description=(
        "Pull a project from GDrive. Downloads the latest cloud state "
        "of the project from Google Drive and updates the local copy. "
        "Use this to sync local with the latest cloud version."
    ),
    parameters={
        "project": {
            "type": "string",
            "description": "Project name in ~/Projects/.",
        },
        "agent": {
            "type": "string",
            "description": "Agent name performing the pull (default 'cloud').",
            "default": "cloud",
        },
    },
)

PROJECT_SYNC_FORKS_TOOL = MCPToolSpec(
    name="maos_project_sync_forks",
    description=(
        "List pending forks — cloud AI changes not yet pulled locally. "
        "A fork is registered when a cloud agent modifies a project. "
        "Use this to check if any cloud agents have made changes that "
        "need to be resolved."
    ),
    parameters={},
)

PROJECT_SYNC_FORK_TOOL = MCPToolSpec(
    name="maos_project_sync_fork",
    description=(
        "Register a cloud AI fork. Call this when a cloud agent has "
        "modified a project, recording the project hash after the "
        "changes. This creates a pending fork that must later be "
        "resolved by pulling from GDrive."
    ),
    parameters={
        "project": {
            "type": "string",
            "description": "Project name.",
        },
        "agent": {
            "type": "string",
            "description": "Agent name (e.g. 'devin', 'grokhack').",
        },
        "hash": {
            "type": "string",
            "description": "Project hash after cloud AI changes.",
        },
    },
)

PROJECT_SYNC_RESOLVE_TOOL = MCPToolSpec(
    name="maos_project_sync_resolve",
    description=(
        "Resolve a fork by pulling from GDrive. After a cloud AI fork "
        "has been registered, this pulls the latest state from Google "
        "Drive to reconcile the local copy and clears the pending fork."
    ),
    parameters={
        "project": {
            "type": "string",
            "description": "Project name to resolve.",
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
    RECORD_DECISION_TOOL,
    QUERY_DECISIONS_TOOL,
    VERIFY_DECISIONS_TOOL,
    SEND_EMAIL_TOOL,
    ESCALATE_TOOL,
    DRIVE_LIST_TOOL,
    DRIVE_SEARCH_TOOL,
    DRIVE_UPLOAD_TOOL,
    DRIVE_DOWNLOAD_TOOL,
    DRIVE_MKDIR_TOOL,
    DRIVE_QUOTA_TOOL,
    SCAN_VIBES_TOOL,
    CHECK_GUARDS_TOOL,
    VERIFY_COMMAND_TOOL,
    INSPECT_DIFF_TOOL,
    PR_READINESS_TOOL,
    DIAGNOSTICS_TOOL,
    TIME_TOOL,
    READ_FILE_TOOL,
    STORAGE_SCAN_TOOL,
    STORAGE_SYSTEM_DATA_TOOL,
    STORAGE_FIX_SCRIPT_TOOL,
    STORAGE_MAINTAIN_TOOL,
    STORAGE_DELETE_TM_SNAPSHOT_TOOL,
    STORAGE_HIDDEN_TOOL,
    # Coding Agent Tools (Devin CLI parity)
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
    GREP_TOOL,
    GLOB_TOOL,
    FIND_FILE_TOOL,
    RUN_COMMAND_TOOL,
    WEB_SEARCH_TOOL,
    FETCH_URL_TOOL,
    NOTEBOOK_READ_TOOL,
    NOTEBOOK_EDIT_TOOL,
    GIT_STATUS_TOOL,
    GIT_DIFF_TOOL,
    GIT_COMMIT_TOOL,
    GIT_LOG_TOOL,
    GIT_BRANCH_TOOL,
    RUN_SUBAGENT_TOOL,
    TODO_WRITE_TOOL,
    ASK_USER_TOOL,
    # Slack tools
    SLACK_SEND_TOOL,
    SLACK_READ_TOOL,
    SLACK_CHANNELS_TOOL,
    SLACK_SEARCH_TOOL,
    SLACK_SYNC_TOOL,
    # Project sync tools
    PROJECT_SYNC_STATUS_TOOL,
    PROJECT_SYNC_PUSH_TOOL,
    PROJECT_SYNC_PULL_TOOL,
    PROJECT_SYNC_FORKS_TOOL,
    PROJECT_SYNC_FORK_TOOL,
    PROJECT_SYNC_RESOLVE_TOOL,
]
