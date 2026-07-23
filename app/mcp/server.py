"""MCP server — FastMCP adapter over the MAOS Context API.

Creates a FastMCP server that exposes MAOS resources and tools. The
server is a thin adapter: all logic lives in app/mcp/resources.py,
app/mcp/tools.py, and the existing Context API / ActionExecutor.

Security:
    A localhost capability token is generated on first run and stored
    at ~/.maos/token. For HTTP transport, the token must be provided
    via the Authorization header. For stdio transport (default), the
    token check is skipped (stdio is local-only by definition).

    The server binds to 127.0.0.1 by default. No external network
    exposure.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from app.log import get_logger
from app.runtime import Runtime

log = get_logger(__name__)

# Token storage location.
TOKEN_DIR = Path.home() / ".maos"
TOKEN_FILE = TOKEN_DIR / "token"


def get_or_create_token() -> str:
    """Get the localhost capability token, creating it on first run.

    The token is a 32-byte hex string stored at ~/.maos/token with
    0600 permissions. It persists across restarts.
    """
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    TOKEN_FILE.write_text(token)
    # Restrict to owner only.
    os.chmod(TOKEN_FILE, 0o600)
    log.info("mcp.token.created", path=str(TOKEN_FILE))
    return token


def validate_token(provided: str | None, expected: str) -> bool:
    """Validate a capability token using constant-time comparison."""
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def create_mcp_server(runtime: Runtime, *, require_token: bool = True) -> Any:
    """Create a FastMCP server wired to the MAOS runtime.

    Args:
        runtime: The MAOS Runtime with db, retriever, blackboard, actions.
        require_token: If True, HTTP requests must include the capability
            token. stdio transport always skips token checks.

    Returns:
        A FastMCP server instance ready to run.
    """
    from mcp.server.fastmcp import FastMCP

    from app.mcp import resources, tools

    mcp = FastMCP("MAOS Context Layer")
    token = get_or_create_token() if require_token else None

    # --- Resources ---------------------------------------------------------

    @mcp.resource("maos://context/current")
    def current_context() -> str:
        """Universal context packet: workspace, session, tasks, decisions, knowledge, telemetry, blackboard."""
        return resources.read_context_resource(runtime)

    @mcp.resource("maos://workspace/current")
    def current_workspace() -> str:
        """Active workspace with git branch, modified files, and status."""
        return resources.read_workspace_resource(runtime)

    @mcp.resource("maos://session/latest")
    def latest_session() -> str:
        """Most recent work session: summary, files changed, commands, unfinished work."""
        return resources.read_session_resource(runtime)

    @mcp.resource("maos://tasks/open")
    def open_tasks() -> str:
        """Pending tasks with dependencies, priority, and workspace."""
        return resources.read_tasks_resource(runtime)

    @mcp.resource("maos://knowledge/search{query}")
    def knowledge_search(query: str = "") -> str:
        """Semantic search over indexed knowledge (vectors + graph)."""
        return resources.read_knowledge_resource(runtime, query=query)

    @mcp.resource("maos://time/current")
    def current_time() -> str:
        """Time awareness: current time, session elapsed, tool call stats, idle time."""
        return resources.read_time_resource(runtime)

    # --- Tools (instrumented with MCP telemetry) ----------------------------

    @mcp.tool()
    def maos_get_context() -> str:
        """Get the full MAOS context packet (workspace, session, tasks, decisions, knowledge, telemetry, blackboard).

        This is the primary tool for understanding what the user is currently
        working on. Start here before asking specific questions.
        """
        return tools._instrument(runtime, "maos_get_context", tools.maos_get_context)

    @mcp.tool()
    def maos_search_memory(query: str, top_k: int = 5) -> str:
        """Search MAOS memory (vector store) for content matching the query.

        Returns ranked results with text, source path, and relevance score.
        Use this to find indexed documents, code, and commit history.
        """
        return tools._instrument(
            runtime, "maos_search_memory",
            tools.maos_search_memory, query=query, top_k=top_k,
        )

    @mcp.tool()
    def maos_workspace_summary(workspace: str = "") -> str:
        """Generate a summary of a workspace's current state.

        Includes git branch, modified files, and pending tasks for the
        specified workspace. If no workspace is given, uses the active one.
        """
        return tools._instrument(
            runtime, "maos_workspace_summary",
            tools.maos_workspace_summary, workspace=workspace,
        )

    @mcp.tool()
    def maos_create_task(title: str, workspace: str = "", priority: int = 0) -> str:
        """Create a new TODO task.

        The task is created through the MAOS ActionExecutor, which enforces
        permission checks and logs the action.
        """
        return tools._instrument(
            runtime, "maos_create_task",
            tools.maos_create_task, title=title, workspace=workspace, priority=priority,
        )

    @mcp.tool()
    def maos_list_tasks(workspace: str = "") -> str:
        """List pending tasks, optionally filtered by workspace.

        Returns tasks through the MAOS ActionExecutor (read permission required).
        """
        return tools._instrument(
            runtime, "maos_list_tasks",
            tools.maos_list_tasks, workspace=workspace,
        )

    @mcp.tool()
    def maos_run_action(
        action: str, target: str = "", parameters: dict[str, Any] | None = None,
    ) -> str:
        """Execute any registered MAOS action by name.

        This is the generic gateway to the MAOS ActionExecutor. All permission
        checks, reversibility checks, action logging, and event publishing
        are preserved. Available actions can be listed via the /actions endpoint
        of the Context API.

        Common actions:
        - open_workspace: Open a project directory in Finder
        - create_todo: Create a task
        - complete_todo: Mark a task as done
        - workspace_summary: Get workspace state summary
        - list_tasks: List pending tasks
        """
        return tools._instrument(
            runtime, "maos_run_action",
            tools.maos_run_action, action=action, target=target, parameters=parameters,
        )

    # --- Onboarding tools ---

    @mcp.tool()
    def maos_onboard(format: str = "text") -> str:
        """Get the onboarding briefing for a new AI session.

        Returns user preferences, hard-won lessons, recent session, active
        workspace, open tasks, and known failure patterns. This is the FIRST
        thing a new session should call before exploring any codebase.

        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_onboard",
            tools.maos_onboard, format=format,
        )

    @mcp.tool()
    def maos_record_lesson(
        slug: str, text: str, confidence: float = 0.9, source: str = "incident",
    ) -> str:
        """Record a hard-won lesson in the user model.

        Lessons are surfaced in every future onboarding briefing. Use this
        when something goes wrong and the session learns a rule that should
        never be violated again.

        Args:
            slug: Short snake_case identifier (e.g. 'verify_before_pr')
            text: The lesson — what to do or what not to do
            confidence: How strongly to hold this (0-1, default 0.9)
            source: Origin — 'incident', 'observed', or 'manual'
        """
        return tools._instrument(
            runtime, "maos_record_lesson",
            tools.maos_record_lesson,
            slug=slug, text=text, confidence=confidence, source=source,
        )

    @mcp.tool()
    def maos_list_lessons() -> str:
        """List all recorded lessons, sorted by confidence (highest first).

        Read-only.
        """
        return tools._instrument(
            runtime, "maos_list_lessons",
            tools.maos_list_lessons,
        )

    # --- Email tools ---

    @mcp.tool()
    def maos_send_email(subject: str, body: str, to: str = "") -> str:
        """Send an email via Gmail (from david.nichols.ops@gmail.com).

        Tokens are refreshed automatically. Default recipient is david.
        Use this when the agent needs to communicate something to the user
        outside of the current session.
        """
        return tools._instrument(
            runtime, "maos_send_email",
            tools.maos_send_email, subject=subject, body=body, to=to,
        )

    @mcp.tool()
    def maos_escalate(
        subject: str,
        body: str,
        context: dict[str, Any] | None = None,
        lesson_slug: str = "",
    ) -> str:
        """Send an escalation email — 'this needs human attention'.

        This is the anti-vibe-overload channel. When something isn't working,
        when a verification fails, when context is getting diluted with false
        confidence, escalate instead of glossing over it. Sends to
        david.nichols.ops@gmail.com with [MAOS] prefix.

        Args:
            subject: Short summary of what needs attention.
            body: The detailed explanation of what happened.
            context: Optional structured context dict to append.
            lesson_slug: Optional lesson slug if related to a recorded lesson.
        """
        return tools._instrument(
            runtime, "maos_escalate",
            tools.maos_escalate,
            subject=subject, body=body,
            context=context, lesson_slug=lesson_slug,
        )

    # --- Guard tools ---

    @mcp.tool()
    def maos_scan_vibes(text: str) -> str:
        """Scan text for vibe language — unsupported success claims.

        Catches phrases like 'everything is working', 'verified' without
        evidence, 'all tests pass' without output, 'looks good', and
        sub-agent agreement used as proof. Returns PASS/WARN/BLOCK.

        Better a false positive than a false negative.
        """
        return tools._instrument(
            runtime, "maos_scan_vibes",
            tools.maos_scan_vibes, text=text,
        )

    @mcp.tool()
    def maos_check_guards(
        text: str,
        action_description: str = "",
        is_pr: bool = False,
        is_bug_fix: bool = False,
        has_test_output: bool = False,
        has_build_output: bool = False,
        has_manual_verification: bool = False,
        claims_subagent_agreement: bool = False,
        has_empirical_evidence: bool = False,
    ) -> str:
        """Run all guards against a text + action context.

        This is the comprehensive pre-action check. Run it before opening
        a PR, committing a fix, or declaring success. If any guard returns
        BLOCK, do NOT proceed — fix the issues and re-run.

        Guards: vibe language, PR pre-flight, lesson compliance, sub-agent proof.
        """
        return tools._instrument(
            runtime, "maos_check_guards",
            tools.maos_check_guards,
            text=text,
            action_description=action_description,
            is_pr=is_pr,
            is_bug_fix=is_bug_fix,
            has_test_output=has_test_output,
            has_build_output=has_build_output,
            has_manual_verification=has_manual_verification,
            claims_subagent_agreement=claims_subagent_agreement,
            has_empirical_evidence=has_empirical_evidence,
        )

    # --- Evidence-producing guard tools (PR governance) ---

    @mcp.tool()
    def maos_verify_command(
        command: str,
        cwd: str = "",
        timeout: int = 300,
    ) -> str:
        """Execute a verification command and capture evidence.

        Evidence-PRODUCING verification. The system runs the command, not the
        agent. The agent cannot fake the exit code, stdout, or commit hash.
        Use this instead of self-attested booleans like has_test_output=True.

        SECURITY: Only verification commands are allowed (test runners, linters,
        type checkers, build commands). Allowed prefixes: uv run pytest, uv run
        ruff, uv run mypy, pytest, ruff, mypy, npm test, npm run lint, cargo
        test, cargo clippy, go test, make test, git status, etc.

        Capture test evidence: maos_verify_command(command="uv run pytest")
        Capture lint evidence: maos_verify_command(command="uv run ruff check app tests")

        Returns JSON with: exit_code, stdout, stderr, duration_ms, timestamp,
        working_dir, git_commit, git_dirty.
        """
        return tools._instrument(
            runtime, "maos_verify_command",
            tools.maos_verify_command,
            command=command, cwd=cwd, timeout=timeout,
        )

    @mcp.tool()
    def maos_inspect_diff(
        repo_dir: str = "",
        base: str = "HEAD",
        stated_scope: str = "",
    ) -> str:
        """Inspect the git diff for risk patterns.

        BLOCK for secrets/credentials. WARN for debug statements, dependency
        changes, binary files, scope violations, large diffs. REQUIRES_REVIEW
        for MCP tool/schema changes and security-sensitive paths.

        Run this before opening a PR to catch risky changes.
        """
        return tools._instrument(
            runtime, "maos_inspect_diff",
            tools.maos_inspect_diff,
            repo_dir=repo_dir, base=base, stated_scope=stated_scope,
        )

    @mcp.tool()
    def maos_pr_readiness(
        repo_dir: str = "",
        pr_description: str = "",
        is_bug_fix: bool = False,
        test_evidence_json: str = "",
        lint_evidence_json: str = "",
        build_evidence_json: str = "",
        manual_evidence_json: str = "",
        diff_base: str = "HEAD",
        skip_diff: bool = False,
    ) -> str:
        """Assess PR readiness via state machine: DRAFT -> UNDERSTOOD -> VERIFIED -> PR_READY.

        The comprehensive readiness check. Requires evidence objects (from
        maos_verify_command), a structured PR description, and inspects the
        diff. If state is NOT PR_READY, do NOT open the PR.

        PR description must have sections: Summary, Why, What Changed, Scope.
        Bug fixes also need: Root Cause, Regression Test, Before, After.

        Pass evidence JSON strings from maos_verify_command to test_evidence_json,
        lint_evidence_json, etc.
        """
        return tools._instrument(
            runtime, "maos_pr_readiness",
            tools.maos_pr_readiness,
            repo_dir=repo_dir,
            pr_description=pr_description,
            is_bug_fix=is_bug_fix,
            test_evidence_json=test_evidence_json,
            lint_evidence_json=lint_evidence_json,
            build_evidence_json=build_evidence_json,
            manual_evidence_json=manual_evidence_json,
            diff_base=diff_base,
            skip_diff=skip_diff,
        )

    # --- Diagnostic tools ---

    @mcp.tool()
    def maos_diagnostics(format: str = "text") -> str:
        """Check functionality and implementation status of all MAOS tools.

        Actually calls each read-only tool and verifies it returns valid
        JSON. Side-effect tools are checked for handler existence but not
        called. Returns per-tool status: working, degraded, broken, skipped.

        Use this to verify the MAOS system is healthy before relying on it.
        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_diagnostics",
            tools.maos_diagnostics, format=format,
        )

    # --- Time awareness tools ---

    @mcp.tool()
    def maos_time(format: str = "text") -> str:
        """Get time awareness information for the current AI session.

        AI agents have no native concept of time. This tool gives the agent:
        - Current wall-clock time (with timezone and human-readable format)
        - Session elapsed time (how long this Devin session has been running)
        - Session start time (when the first tool call in this session happened)
        - Tool call count and frequency (calls per minute)
        - Idle time (how long since the last tool call)
        - Reading profile (files read, re-read ratio, reading velocity)
        - Relative time hints ("3 minutes since last tool call")

        A "session" is a contiguous sequence of MCP tool calls with no gap
        longer than 10 minutes. Call this to know how long you've been
        working on a task.

        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_time",
            tools.maos_time, format=format,
        )

    @mcp.tool()
    def maos_read_file(path: str, max_lines: int = 2000) -> str:
        """Read a file with time awareness and reading behavior tracking.

        This is the time-aware version of file reading. It returns file
        content PLUS a _time_awareness section showing session elapsed,
        tool call count, files read, re-read ratio, and a behavioral
        assessment. If the re-read ratio is >50%, a _warning is added.

        Reading is the #1 repeated action for AI agents. By routing reads
        through this tool, every read becomes a time check and behavioral
        checkpoint. Use this instead of the built-in read tool when you
        want to stay aware of how long you've been working.

        Args:
            path: Absolute or relative path to the file.
            max_lines: Maximum lines to read (default 2000).
        """
        return tools._instrument(
            runtime, "maos_read_file",
            tools.maos_read_file, path=path, max_lines=max_lines,
        )

    # Store token for HTTP transport middleware.
    mcp._maos_token = token  # type: ignore[attr-defined]
    mcp._maos_runtime = runtime  # type: ignore[attr-defined]

    return mcp


def run_mcp_server(
    runtime: Runtime,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 7778,
    require_token: bool = True,
) -> None:
    """Create and run the MCP server.

    Args:
        transport: "stdio" (default, for CLI integration) or "streamable-http".
        host: Bind address for HTTP transport (default 127.0.0.1).
        port: Port for HTTP transport (default 7778).
        require_token: Whether to require the capability token for HTTP.
    """
    mcp = create_mcp_server(runtime, require_token=require_token)

    if transport == "stdio":
        log.info("mcp.serving", transport="stdio")
        mcp.run(transport="stdio")

    elif transport in ("http", "streamable-http"):
        token = mcp._maos_token  # type: ignore[attr-defined]
        log.info("mcp.serving", transport="http", host=host, port=port)

        if token:
            print(f"\n  MAOS MCP Server — http://{host}:{port}/mcp")
            print(f"  Token: {token}")
            print(f"  Token file: {TOKEN_FILE}")
            print("\n  Clients must send: Authorization: Bearer {token}")
            print("\n  Press Ctrl+C to stop.\n")
        else:
            print(f"\n  MAOS MCP Server — http://{host}:{port}/mcp")
            print("  Token validation disabled.\n")

        # FastMCP streamable-http accepts host and port via settings.
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")

    else:
        raise ValueError(f"unsupported transport: {transport}")
