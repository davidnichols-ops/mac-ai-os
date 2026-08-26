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
        category: str = "", keywords: str = "", reminder: str = "",
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
            category: Lesson category — 'tooling', 'workflow', 'environment',
                'api', or 'architecture'. Groups lessons in the briefing.
            keywords: Comma-separated trigger words. When these appear in a
                tool call's context, the reminder is injected into the response.
            reminder: One-line reminder shown on keyword match. Keep it short.
        """
        return tools._instrument(
            runtime, "maos_record_lesson",
            tools.maos_record_lesson,
            slug=slug, text=text, confidence=confidence, source=source,
            category=category, keywords=keywords, reminder=reminder,
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

    # --- Decision audit log tools ---

    @mcp.tool()
    def maos_record_decision(
        agent: str,
        action: str,
        rationale: str,
        kind: str = "decision",
        tags: str = "",
        confidence: float = 0.0,
    ) -> str:
        """Record a structured agent decision to the hash-chained audit log.

        Each decision is linked to the previous one via a SHA-256 digest chain,
        making the log tamper-evident. Use this to make agent reasoning auditable
        and reproducible. Decisions appear in the context packet.

        Args:
            agent: Identifier of the agent making the decision (e.g. 'devin',
                'claude-3', 'human:david').
            action: The action being decided on (e.g. 'commit', 'deploy',
                'merge', 'escalate').
            rationale: Why the decision was made. This is the auditable reasoning trail.
            kind: Decision category. 'decision' = chose an action (default),
                'observation' = noted a fact, 'escalation' = sent to human,
                'veto' = blocked an action.
            tags: Comma-separated free-form tags for grouping decisions.
            confidence: Confidence level 0-1 (default 0).
        """
        return tools._instrument(
            runtime, "maos_record_decision",
            tools.maos_record_decision,
            agent=agent, action=action, rationale=rationale,
            kind=kind, tags=tags, confidence=confidence,
        )

    @mcp.tool()
    def maos_query_decisions(
        agent: str = "",
        kind: str = "",
        limit: int = 100,
    ) -> str:
        """Query the decision audit log. Read-only and filtered.

        Returns matching decisions with their digests and chain pointers. Does
        not include rationale (data minimization) — the digest proves the
        rationale was recorded.

        Args:
            agent: Filter to decisions by this agent only.
            kind: Filter to decisions of this kind only.
            limit: Maximum number of decisions to return (default 100, max 1000).
        """
        return tools._instrument(
            runtime, "maos_query_decisions",
            tools.maos_query_decisions,
            agent=agent, kind=kind, limit=limit,
        )

    @mcp.tool()
    def maos_verify_decisions() -> str:
        """Verify the integrity of the decision hash chain.

        Checks that every decision's digest matches its recomputed value and
        that the chain links are intact. Returns valid=True if the chain is
        tamper-free. Use this to detect tampering with the audit log.
        """
        return tools._instrument(
            runtime, "maos_verify_decisions",
            tools.maos_verify_decisions,
        )

    # --- Email tools ---

    @mcp.tool()
    def maos_send_email(subject: str, body: str, to: str = "") -> str:
        """Send an email via Gmail.

        Tokens are refreshed automatically. The default recipient is the
        MAOS_EMAIL_RECIPIENT environment variable or the 'to' argument.
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
        confidence, escalate instead of glossing over it. Sends to the
        configured MAOS email recipient with a [MAOS] prefix.

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

    # --- Google Drive tools ---

    @mcp.tool()
    def maos_drive_list(folder_id: str = "", page_size: int = 50, query: str = "") -> str:
        """List files in a Google Drive folder.

        Returns file id, name, mimeType, size, modifiedTime for each file.
        Use folder_id='root' or omit for the root folder.
        """
        return tools._instrument(
            runtime, "maos_drive_list",
            tools.maos_drive_list,
            folder_id=folder_id, page_size=page_size, query=query,
        )

    @mcp.tool()
    def maos_drive_search(query: str, page_size: int = 20) -> str:
        """Search Google Drive by name or content query.

        Returns matching files with id, name, mimeType, size.
        """
        return tools._instrument(
            runtime, "maos_drive_search",
            tools.maos_drive_search,
            query=query, page_size=page_size,
        )

    @mcp.tool()
    def maos_drive_upload(
        name: str,
        content_b64: str,
        mime_type: str = "application/octet-stream",
        parent_folder_id: str = "",
    ) -> str:
        """Upload a file to Google Drive (5TB Workspace account).

        Creates or overwrites a file in the specified folder.
        Content must be base64-encoded. Returns the file id and metadata.
        """
        return tools._instrument(
            runtime, "maos_drive_upload",
            tools.maos_drive_upload,
            name=name, content_b64=content_b64,
            mime_type=mime_type, parent_folder_id=parent_folder_id,
        )

    @mcp.tool()
    def maos_drive_download(file_id: str) -> str:
        """Download a file from Google Drive by its file ID.

        Returns the file content as base64-encoded string.
        """
        return tools._instrument(
            runtime, "maos_drive_download",
            tools.maos_drive_download,
            file_id=file_id,
        )

    @mcp.tool()
    def maos_drive_mkdir(name: str, parent_folder_id: str = "") -> str:
        """Create a folder on Google Drive (or get existing if it already exists).

        Returns the folder id and metadata.
        """
        return tools._instrument(
            runtime, "maos_drive_mkdir",
            tools.maos_drive_mkdir,
            name=name, parent_folder_id=parent_folder_id,
        )

    @mcp.tool()
    def maos_drive_quota() -> str:
        """Get Google Drive storage quota info for the configured account.

        Returns usage, limit, and user email.
        """
        return tools._instrument(
            runtime, "maos_drive_quota",
            tools.maos_drive_quota,
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

    # --- Network intelligence tools ---

    @mcp.tool()
    def maos_network_status(format: str = "text") -> str:
        """Get current network health and bandwidth status.

        Returns whether the network is healthy enough for cloud LLM calls,
        the active interface (wifi/ethernet), current RX/TX bandwidth,
        latency (TCP RTT probe), packet loss estimate, and a 0-100 quality
        score. The network collector must be running for fresh data.

        Use this before deciding whether to use cloud vs local models,
        before large uploads, or to explain slow responses.

        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_network_status",
            tools.maos_network_status, format=format,
        )

    @mcp.tool()
    def maos_network_history(
        hours: int = 24, resolution: str = "hourly",
    ) -> str:
        """Get historical network bandwidth and latency data.

        Returns aggregated history: per-bucket RX/TX bytes, average rates,
        latency, and any detected events (interruptions, degradation,
        interface transitions).

        Args:
            hours: Lookback window in hours (default 24).
            resolution: 'hourly' (default, indefinite retention) or 'minute'
                (30-day retention, more detail).
        """
        return tools._instrument(
            runtime, "maos_network_history",
            tools.maos_network_history,
            hours=hours, resolution=resolution,
        )

    @mcp.resource("maos://network/current")
    def current_network() -> str:
        """Current network health: interface, bandwidth, latency, quality score."""
        return resources.read_network_resource(runtime)

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

    # --- Storage management tools ---

    @mcp.tool()
    def maos_storage_scan(mode: str = "full", format: str = "text") -> str:
        """Run a comprehensive storage analysis on the Mac.

        Decomposes the mysterious macOS "System Data" category into
        identifiable, reclaimable components. Combines native macOS
        introspection (tmutil, diskutil, /private/var/vm, unified log
        archive, Spotlight, APFS snapshots), X-MaC disk + clean engines,
        NeoDisk scan cache detection, and user-managed cache scan.

        Modes:
          - "full": Everything (default). Includes xmac disk + clean.
          - "fast": Native macOS only — no filesystem walk.
          - "system": Only system data decomposition.
          - "caches": Only user cache scan.

        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_storage_scan",
            tools.maos_storage_scan, mode=mode, format=format,
        )

    @mcp.tool()
    def maos_storage_system_data(format: str = "text") -> str:
        """Decompose macOS "System Data" into identifiable components.

        The focused tool for solving the "120 GB of system data" mystery.
        Checks every known location that macOS categorizes as "System
        Data" or "Other": sleepimage, swapfiles, unified log archive,
        system logs, Spotlight index, asset cache, Time Machine local
        snapshots, APFS OS update snapshots, per-user temp files, and
        system databases.

        Each entry reports: actual size, whether it's reclaimable, whether
        it's safe to delete, and the exact command to reclaim it.

        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_storage_system_data",
            tools.maos_storage_system_data, format=format,
        )

    @mcp.tool()
    def maos_storage_fix_script(
        output_path: str = "./storage-fixes.sh",
        include_caches: bool = True,
        include_system: bool = True,
        include_tm: bool = True,
    ) -> str:
        """Generate a reviewable remediation shell script for storage cleanup.

        ALL destructive commands are COMMENTED OUT in the generated script.
        The user reviews, uncomments what they want, and runs it. This tool
        NEVER executes destructive commands — it only produces the script.

        Includes system data cleanup, Time Machine snapshot deletion, and
        user cache cleanup commands.
        """
        return tools._instrument(
            runtime, "maos_storage_fix_script",
            tools.maos_storage_fix_script,
            output_path=output_path,
            include_caches=include_caches,
            include_system=include_system,
            include_tm=include_tm,
        )

    @mcp.tool()
    def maos_storage_maintain() -> str:
        """Run safe macOS maintenance tasks via X-MaC.

        Delegates to xmac maintain: flush DNS, reindex Spotlight, rebuild
        LaunchServices, run periodic scripts, purge RAM, clear Quick Look.
        Safe tasks run automatically; sudo tasks emitted as findings.
        """
        return tools._instrument(
            runtime, "maos_storage_maintain",
            tools.maos_storage_maintain,
        )

    @mcp.tool()
    def maos_storage_delete_tm_snapshot(date: str) -> str:
        """Delete a Time Machine local snapshot by date.

        The #1 way to reclaim "System Data" space. Local snapshots pin
        purgeable space that only clears on the next backup. Deleting
        them reclaims the space immediately.

        Use maos_storage_scan first to list available snapshots and
        their dates.

        Args:
            date: Snapshot date string, e.g. "2026-07-24-152803".
        """
        return tools._instrument(
            runtime, "maos_storage_delete_tm_snapshot",
            tools.maos_storage_delete_tm_snapshot, date=date,
        )

    @mcp.tool()
    def maos_storage_hidden(format: str = "text") -> str:
        """Scan for hidden data consumers that macOS categorizes as 'System Data'.

        Goes beyond standard system data decomposition and user caches.
        Finds hidden consumers in non-obvious locations: Devin CLI session
        database (3GB+), AI/ML tool data, developer tool data, app support
        directories, cloud storage sync, SDKs, app containers, and
        system-level data (/Library/Updates, Xcode CLT, system assets).

        Each entry reports: actual size, safe to clear, regenerates, and
        the exact command to reclaim it.

        Pass format='json' for structured output.
        """
        return tools._instrument(
            runtime, "maos_storage_hidden",
            tools.maos_storage_hidden, format=format,
        )

    # --- Coding Agent Tools (Devin CLI parity) ---

    @mcp.tool()
    def maos_write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if needed.

        Creates parent directories. Overwrites existing files.
        """
        return tools._instrument(runtime, "maos_write_file", tools.maos_write_file, path=path, content=content)

    @mcp.tool()
    def maos_edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Edit a file by replacing old_string with new_string.

        The old_string must be unique unless replace_all=True.
        """
        return tools._instrument(runtime, "maos_edit_file", tools.maos_edit_file, path=path, old_string=old_string, new_string=new_string, replace_all=replace_all)

    @mcp.tool()
    def maos_grep(
        pattern: str, path: str = ".", glob_pattern: str = "",
        case_insensitive: bool = False, max_results: int = 100, context_lines: int = 0,
    ) -> str:
        """Search file contents using ripgrep.

        Returns matching lines with file paths and line numbers.
        """
        return tools._instrument(
            runtime, "maos_grep", tools.maos_grep,
            pattern=pattern, path=path, glob_pattern=glob_pattern,
            case_insensitive=case_insensitive, max_results=max_results, context_lines=context_lines,
        )

    @mcp.tool()
    def maos_glob(pattern: str, path: str = ".") -> str:
        """Find files by glob pattern.

        Supports brace expansion (e.g. '**/*.{ts,tsx}').
        """
        return tools._instrument(runtime, "maos_glob", tools.maos_glob, pattern=pattern, path=path)

    @mcp.tool()
    def maos_find_file(pattern: str, path: str = ".") -> str:
        """Find files by name pattern using recursive glob matching."""
        return tools._instrument(runtime, "maos_find_file", tools.maos_find_file, pattern=pattern, path=path)

    @mcp.tool()
    def maos_run_command(command: str, cwd: str = "", timeout: int = 120, env: dict[str, str] | None = None) -> str:
        """Execute a shell command and return stdout, stderr, and exit code.

        Blocked: rm -rf /, mkfs, fork bombs. Restricted: sudo, kill -9, shutdown.
        """
        return tools._instrument(runtime, "maos_run_command", tools.maos_run_command, command=command, cwd=cwd, timeout=timeout, env=env)

    @mcp.tool()
    def maos_web_search(query: str, num_results: int = 5) -> str:
        """Search the web for information.

        Returns relevant web page titles, URLs, and summaries.
        """
        return tools._instrument(runtime, "maos_web_search", tools.maos_web_search, query=query, num_results=num_results)

    @mcp.tool()
    def maos_fetch_url(url: str) -> str:
        """Fetch a web page and return its content as readable text."""
        return tools._instrument(runtime, "maos_fetch_url", tools.maos_fetch_url, url=url)

    @mcp.tool()
    def maos_notebook_read(path: str) -> str:
        """Read a Jupyter notebook (.ipynb) and extract all cells with outputs."""
        return tools._instrument(runtime, "maos_notebook_read", tools.maos_notebook_read, path=path)

    @mcp.tool()
    def maos_notebook_edit(
        path: str, cell_number: int, new_source: str,
        cell_type: str = "code", edit_mode: str = "replace",
    ) -> str:
        """Edit a cell in a Jupyter notebook.

        edit_mode: 'replace' (default), 'insert', or 'delete'.
        """
        return tools._instrument(
            runtime, "maos_notebook_edit", tools.maos_notebook_edit,
            path=path, cell_number=cell_number, new_source=new_source,
            cell_type=cell_type, edit_mode=edit_mode,
        )

    @mcp.tool()
    def maos_git_status(repo_dir: str = "") -> str:
        """Get git status for a repository."""
        return tools._instrument(runtime, "maos_git_status", tools.maos_git_status, repo_dir=repo_dir)

    @mcp.tool()
    def maos_git_diff(repo_dir: str = "", staged: bool = False, base: str = "", target: str = "") -> str:
        """Get git diff for a repository."""
        return tools._instrument(runtime, "maos_git_diff", tools.maos_git_diff, repo_dir=repo_dir, staged=staged, base=base, target=target)

    @mcp.tool()
    def maos_git_commit(message: str, repo_dir: str = "", add_all: bool = False) -> str:
        """Stage files and create a git commit."""
        return tools._instrument(runtime, "maos_git_commit", tools.maos_git_commit, message=message, repo_dir=repo_dir, add_all=add_all)

    @mcp.tool()
    def maos_git_log(repo_dir: str = "", limit: int = 20, oneline: bool = True) -> str:
        """Get git log for a repository."""
        return tools._instrument(runtime, "maos_git_log", tools.maos_git_log, repo_dir=repo_dir, limit=limit, oneline=oneline)

    @mcp.tool()
    def maos_git_branch(repo_dir: str = "", action: str = "list", branch_name: str = "") -> str:
        """Manage git branches. Actions: list, create, switch, delete."""
        return tools._instrument(runtime, "maos_git_branch", tools.maos_git_branch, repo_dir=repo_dir, action=action, branch_name=branch_name)

    @mcp.tool()
    def maos_run_subagent(task: str, spec: str, title: str = "", profile: str = "general", is_background: bool = False) -> str:
        """Spawn a subagent to handle a task autonomously.

        The spec flag is MANDATORY — it determines which AI system to
        route through: devin, grokhack, claude-code, local, or openrouter.
        """
        return tools._instrument(runtime, "maos_run_subagent", tools.maos_run_subagent, task=task, spec=spec, title=title, profile=profile, is_background=is_background)

    @mcp.tool()
    def maos_todo_write(todos: list[dict[str, str]] | None = None) -> str:
        """Create or update a structured todo list for the current session.

        Each todo has 'content' and 'status' (pending, in_progress, completed).
        """
        return tools._instrument(runtime, "maos_todo_write", tools.maos_todo_write, todos=todos)

    @mcp.tool()
    def maos_ask_user(questions: list[dict[str, Any]] | None = None) -> str:
        """Present questions to the user and collect their answers.

        For HTTP sessions, returns a prompt for the client to present.
        """
        return tools._instrument(runtime, "maos_ask_user", tools.maos_ask_user, questions=questions)

    # --- Slack tools ---

    @mcp.tool()
    def maos_slack_send(channel: str, message: str, thread_ts: str = "") -> str:
        """Send a message to a Slack channel or DM.

        Uses the bot token (chat.postMessage). The channel can be a channel
        ID (e.g. C0BMSQGTCJV), a channel name (e.g. #maos-ops), or a user
        ID for a DM. Optionally reply in a thread by providing thread_ts.

        Args:
            channel: Channel ID, channel name, or user ID for DM.
            message: The message text to send (Slack mrkdwn supported).
            thread_ts: Optional timestamp of the parent message to reply in a thread.
        """
        return tools._instrument(
            runtime, "maos_slack_send",
            tools.maos_slack_send,
            channel=channel, message=message, thread_ts=thread_ts,
        )

    @mcp.tool()
    def maos_slack_read(channel: str, limit: int = 50, oldest: str = "") -> str:
        """Read recent messages from a Slack channel.

        Uses the bot token (conversations.history). Returns messages as JSON
        with text, user, ts, and thread info. Use maos_slack_channels to find
        channel IDs.

        Args:
            channel: Channel ID or name to read from.
            limit: Maximum messages to return (default 50, max 200).
            oldest: Optional Unix timestamp — only return messages after this.
        """
        return tools._instrument(
            runtime, "maos_slack_read",
            tools.maos_slack_read,
            channel=channel, limit=limit, oldest=oldest,
        )

    @mcp.tool()
    def maos_slack_channels(types: str = "public_channel,private_channel") -> str:
        """List all visible Slack channels.

        Uses the bot token (conversations.list). Returns channel names, IDs,
        and type (public, private, im, mpim). Use this to find channel IDs
        for maos_slack_send and maos_slack_read.

        Args:
            types: Comma-separated channel types to list (default:
                public_channel,private_channel).
        """
        return tools._instrument(
            runtime, "maos_slack_channels",
            tools.maos_slack_channels,
            types=types,
        )

    @mcp.tool()
    def maos_slack_search(query: str, count: int = 20) -> str:
        """Search Slack messages across the workspace.

        Uses the user token (search.messages) — requires SLACK_USER_TOKEN.
        Returns matching messages with channel, author, timestamp, and
        permalink.

        Args:
            query: Search query (Slack search syntax supported).
            count: Maximum results to return (default 20, max 100).
        """
        return tools._instrument(
            runtime, "maos_slack_search",
            tools.maos_slack_search,
            query=query, count=count,
        )

    @mcp.tool()
    def maos_slack_sync(days: int = 30) -> str:
        """Trigger an incremental Slack history sync into MAOS.

        Pulls recent messages from visible channels into the local SQLite
        store and refreshes the RAG index so messages become searchable via
        maos_search_memory. Uses the user token. Returns sync counts
        (channels, messages, users).

        Args:
            days: How many days of history to sync (default 30).
        """
        return tools._instrument(
            runtime, "maos_slack_sync",
            tools.maos_slack_sync,
            days=days,
        )

    # --- Project sync tools ---

    @mcp.tool()
    def maos_project_sync_status() -> str:
        """Show sync state for all projects.

        Displays which projects are in sync, which have local changes to
        push, and which have pending forks from cloud AI agents. Calls
        the project-sync CLI.
        """
        return tools._instrument(
            runtime, "maos_project_sync_status",
            tools.maos_project_sync_status,
        )

    @mcp.tool()
    def maos_project_sync_push(project: str, agent: str = "cloud") -> str:
        """Push a project to GDrive and record blockchain evidence.

        Uploads the current local state to Google Drive and creates a
        hash-chained evidence entry. Use after making local changes that
        should be propagated to the cloud.

        Args:
            project: Project name in ~/Projects/.
            agent: Agent name performing the push (default 'cloud').
        """
        return tools._instrument(
            runtime, "maos_project_sync_push",
            tools.maos_project_sync_push,
            project=project, agent=agent,
        )

    @mcp.tool()
    def maos_project_sync_pull(project: str, agent: str = "cloud") -> str:
        """Pull a project from GDrive.

        Downloads the latest cloud state from Google Drive and updates
        the local copy.

        Args:
            project: Project name in ~/Projects/.
            agent: Agent name performing the pull (default 'cloud').
        """
        return tools._instrument(
            runtime, "maos_project_sync_pull",
            tools.maos_project_sync_pull,
            project=project, agent=agent,
        )

    @mcp.tool()
    def maos_project_sync_forks() -> str:
        """List pending forks — cloud AI changes not yet pulled locally.

        A fork is registered when a cloud agent modifies a project. Use
        this to check if any cloud agents have made changes that need to
        be resolved.
        """
        return tools._instrument(
            runtime, "maos_project_sync_forks",
            tools.maos_project_sync_forks,
        )

    @mcp.tool()
    def maos_project_sync_fork(project: str, agent: str, hash: str) -> str:
        """Register a cloud AI fork.

        Call this when a cloud agent has modified a project, recording
        the project hash after the changes. This creates a pending fork
        that must later be resolved by pulling from GDrive.

        Args:
            project: Project name.
            agent: Agent name (e.g. 'devin', 'grokhack').
            hash: Project hash after cloud AI changes.
        """
        return tools._instrument(
            runtime, "maos_project_sync_fork",
            tools.maos_project_sync_fork,
            project=project, agent=agent, hash=hash,
        )

    @mcp.tool()
    def maos_project_sync_resolve(project: str) -> str:
        """Resolve a fork by pulling from GDrive.

        After a cloud AI fork has been registered, this pulls the latest
        state from Google Drive to reconcile the local copy and clears
        the pending fork.

        Args:
            project: Project name to resolve.
        """
        return tools._instrument(
            runtime, "maos_project_sync_resolve",
            tools.maos_project_sync_resolve,
            project=project,
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
        # Allow ngrok/tunnel hosts so external clients (e.g. Gemini Spark)
        # can reach the server through a public tunnel.
        mcp.settings.transport_security = None
        mcp.run(transport="streamable-http")

    else:
        raise ValueError(f"unsupported transport: {transport}")
