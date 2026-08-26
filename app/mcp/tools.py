"""MCP tools — thin adapters over ActionExecutor and the Context API.

Every tool that performs an action goes through ActionExecutor, which
preserves:
  - permission checks
  - reversibility checks
  - action logging (action_log table)
  - event publishing (EventBus)

No tool may bypass ActionExecutor. Read-only tools (maos_get_context,
maos_search_memory) call the Context API directly since they have no
side effects.

All tool calls are instrumented with MCP telemetry: each call is logged
to the mcp_requests table and an MCP_TOOL_CALLED event is published.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from app.actions.models import ActionRequest
from app.diagnostics import format_diagnostic_report, run_diagnostics
from app.events.models import MCP_TOOL_CALLED, Event
from app.gdrive import DriveError
from app.gdrive import (
    download_file as _drive_download,
)
from app.gdrive import (
    get_or_create_folder as _drive_mkdir,
)
from app.gdrive import (
    get_storage_info as _drive_quota,
)
from app.gdrive import (
    list_files as _drive_list,
)
from app.gdrive import (
    search_files as _drive_search,
)
from app.gdrive import (
    upload_file as _drive_upload,
)
from app.gmail import EmailError
from app.gmail import escalate as _escalate
from app.gmail import send_email as _send_email
from app.guards import (
    format_guard_report,
    run_all_guards,
    scan_vibes,
    should_block,
)
from app.log import get_logger
from app.onboarding import (
    build_onboarding_packet,
    format_onboarding_briefing,
    list_lessons,
    record_lesson,
)
from app.runtime import Runtime
from app.server.context import build_context_packet

log = get_logger(__name__)


def _instrument(
    runtime: Runtime, name: str, fn: callable, *args: Any, **kwargs: Any
) -> str:
    """Wrap a tool call with telemetry logging and event publishing.

    Records to mcp_requests table and publishes MCP_TOOL_CALLED event.
    Also appends an ambient time footer to every tool response so the
    agent always knows how long it's been working — even deep in a
    coding rabbit hole.

    Finally, scans the tool call arguments + result for lesson keywords
    and injects matching one-line reminders. This is contextual lesson
    activation — the agent sees the reminder at the moment it's relevant,
    not buried in a briefing it read 30 minutes ago.
    """
    start = time.time()
    success = True
    error = ""
    try:
        result = fn(runtime, *args, **kwargs)
        # Append ambient time footer to every tool response.
        result = _append_time_footer(runtime, name, result)
        # Inject keyword-triggered lesson reminders.
        result = _append_lesson_reminders(runtime, name, result, args, kwargs)
        return result
    except Exception as e:
        success = False
        error = str(e)
        return json.dumps({"error": error})
    finally:
        latency_ms = int((time.time() - start) * 1000)
        try:
            runtime.db.log_mcp_request(
                client_name="mcp",
                kind="tool",
                name=name,
                success=success,
                latency_ms=latency_ms,
                error=error,
            )
            if runtime.bus:
                runtime.bus.publish(Event(MCP_TOOL_CALLED, {
                    "name": name, "success": success,
                    "latency_ms": latency_ms,
                }))
        except Exception:
            pass  # telemetry must never break the tool call


def _append_time_footer(runtime: Runtime, tool_name: str, result: str) -> str:
    """Append a compact time awareness footer to a tool response.

    Format: [⏱ 28m · 17 calls · idle 2m · Fri 5:31 PM CDT]

    This is injected into EVERY tool response so the agent always has
    ambient time awareness without needing to explicitly call maos_time.
    The footer is kept short to avoid bloating the response.

    For JSON responses, the time info is added as a top-level
    "_time_awareness" key. For text responses, it's appended as a
    bracketed footer line.

    Skip for maos_time itself (it already returns time info).
    """
    if tool_name == "maos_time":
        return result

    try:
        from app.time import detect_session_start

        session = detect_session_start(runtime.db)
        elapsed = session["elapsed_compact"]
        calls = session["tool_call_count"]
        idle = session.get("idle_compact", "0s")

        # Try to parse as JSON and inject as a top-level key.
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                parsed["_time_awareness"] = {
                    "elapsed": elapsed,
                    "tool_calls": calls,
                    "idle": idle,
                    "active": session["is_active"],
                }
                return json.dumps(parsed, default=str, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass

        # For non-JSON responses, append a text footer.
        status = "active" if session["is_active"] else "idle"
        footer = f"\n[⏱ {elapsed} · {calls} calls · {status} {idle}]"
        return result + footer
    except Exception:
        return result


# Cache lesson keywords to avoid DB queries on every tool call.
_lesson_keyword_cache: dict[str, Any] = {"lessons": [], "cached_at": 0.0}
_LESSON_CACHE_TTL = 300.0  # 5 minutes


def _get_keyword_lessons(runtime: Runtime) -> list[dict[str, Any]]:
    """Get lessons that have keyword triggers, with a 5-minute cache."""
    now = time.time()
    if now - _lesson_keyword_cache["cached_at"] < _LESSON_CACHE_TTL:
        return _lesson_keyword_cache["lessons"]
    try:
        from app.onboarding import list_lessons
        all_lessons = list_lessons(runtime.db)
        kw_lessons = [lesson for lesson in all_lessons if lesson.get("keywords")]
        _lesson_keyword_cache["lessons"] = kw_lessons
        _lesson_keyword_cache["cached_at"] = now
        return kw_lessons
    except Exception:
        return []


def _append_lesson_reminders(
    runtime: Runtime, tool_name: str, result: str,
    args: tuple, kwargs: dict,
) -> str:
    """Inject one-line lesson reminders when keywords match tool context.

    Scans the tool call's arguments and result text for keywords registered
    on lessons. When a keyword matches, the lesson's one-line reminder is
    injected into the response. This is contextual activation — the agent
    sees the reminder at the moment it's relevant.

    Format for JSON responses: adds a "_lesson_reminders" array.
    Format for text responses: appends "[⚠ lesson: ...]" lines.

    To avoid spam, each reminder is shown at most once per tool call, and
    only the top 3 matches are injected (sorted by lesson confidence).
    """
    # Don't inject into lesson management tools (would be recursive/noisy).
    if tool_name in ("maos_record_lesson", "maos_list_lessons", "maos_onboard"):
        return result

    try:
        kw_lessons = _get_keyword_lessons(runtime)
        if not kw_lessons:
            return result

        # Build the text corpus to scan: tool name + args + kwargs + result.
        scan_parts = [tool_name]
        for a in args:
            scan_parts.append(str(a))
        for k, v in kwargs.items():
            scan_parts.append(f"{k}={v}")
        # Include result text but cap to avoid scanning huge responses.
        scan_parts.append(result[:4000])
        scan_text = " ".join(scan_parts).lower()

        # Find matching lessons.
        matches: list[tuple[float, str, str]] = []  # (confidence, slug, reminder)
        for lesson in kw_lessons:
            keywords = lesson.get("keywords", [])
            if not keywords:
                continue
            if any(kw.lower() in scan_text for kw in keywords):
                reminder = lesson.get("reminder", "")
                if reminder:
                    matches.append((lesson["confidence"], lesson["slug"], reminder))

        if not matches:
            return result

        # Sort by confidence, take top 3.
        matches.sort(key=lambda m: -m[0])
        matches = matches[:3]

        # Inject into response.
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                parsed["_lesson_reminders"] = [
                    {"slug": slug, "reminder": reminder}
                    for _, slug, reminder in matches
                ]
                return json.dumps(parsed, default=str, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass

        # Text response: append reminder lines.
        reminder_lines = [f"\n[⚠ lesson: {slug}] {reminder}" for _, slug, reminder in matches]
        return result + "".join(reminder_lines)
    except Exception:
        return result


def maos_get_context(runtime: Runtime) -> str:
    """Get the full MAOS context packet.

    Read-only — calls the Context API directly (no side effects).
    """
    packet = build_context_packet(
        runtime.db, runtime.blackboard, retriever=runtime.retriever
    )
    return json.dumps(packet, default=str, indent=2)


def maos_search_memory(runtime: Runtime, query: str, top_k: int = 5) -> str:
    """Search MAOS memory (vector store) for content matching the query.

    Read-only — calls the retriever directly (no side effects).
    """
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        results = runtime.retriever.search(query, top_k=top_k)
        return json.dumps(
            {
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "text": r.text[:1000],
                        "source": r.source,
                        "score": round(r.score, 3),
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def maos_workspace_summary(runtime: Runtime, workspace: str = "") -> str:
    """Generate a workspace summary.

    Goes through ActionExecutor → workspace_summary action.
    Preserves permissions, logging, and event publishing.
    """
    return _run_action(
        runtime,
        action="workspace_summary",
        target=workspace,
        parameters={"workspace": workspace},
    )


def maos_create_task(
    runtime: Runtime,
    title: str,
    workspace: str = "",
    priority: int = 0,
) -> str:
    """Create a new TODO task.

    Goes through ActionExecutor → create_todo action.
    Preserves permissions, logging, and event publishing.
    """
    return _run_action(
        runtime,
        action="create_todo",
        target=title,
        parameters={
            "title": title,
            "workspace": workspace,
            "priority": priority,
        },
    )


def maos_list_tasks(runtime: Runtime, workspace: str = "") -> str:
    """List pending tasks.

    Goes through ActionExecutor → list_tasks action.
    Preserves permissions, logging, and event publishing.
    """
    return _run_action(
        runtime,
        action="list_tasks",
        target=workspace,
        parameters={"workspace": workspace},
    )


def maos_run_action(
    runtime: Runtime,
    action: str,
    target: str = "",
    parameters: dict[str, Any] | None = None,
) -> str:
    """Execute any registered MAOS action by name.

    This is the generic gateway to ActionExecutor. All checks, logging,
    and event publishing are preserved.
    """
    return _run_action(
        runtime,
        action=action,
        target=target,
        parameters=parameters or {},
    )


# --- internal helper -------------------------------------------------------


def _run_action(
    runtime: Runtime,
    *,
    action: str,
    target: str = "",
    parameters: dict[str, Any] | None = None,
) -> str:
    """Execute an action through ActionExecutor and return JSON result.

    This is the ONLY path to action execution. It always:
    1. Creates an ActionRequest
    2. Calls runtime.action_executor.execute() with db in context
    3. Returns the ActionResult as JSON

    The ActionExecutor handles:
    - permission validation
    - reversibility checks
    - handler execution
    - action_log table insertion
    - EventBus publish (ACTION_EXECUTED / ACTION_DENIED)
    """
    req = ActionRequest(
        action=action,
        target=target,
        parameters=parameters or {},
    )
    result = runtime.action_executor.execute(req, context={"db": runtime.db})
    log.info(
        "mcp.tool.action",
        action=action,
        ok=result.ok,
        duration_ms=result.duration_ms,
    )
    return json.dumps(result.to_dict(), default=str, indent=2)


# --- Onboarding tools -------------------------------------------------------


def maos_onboard(runtime: Runtime, format: str = "text") -> str:
    """Get the onboarding briefing for a new AI session.

    Read-only — assembles user preferences, hard-won lessons, recent session,
    active workspace, open tasks, and known failure patterns. This is the
    FIRST thing a new session should call before exploring any codebase.

    Args:
        format: "text" (default, readable briefing) or "json" (structured).
    """
    packet = build_onboarding_packet(runtime.db)
    if format == "json":
        return json.dumps(packet, default=str, indent=2)
    return format_onboarding_briefing(packet)


def maos_record_lesson(
    runtime: Runtime,
    slug: str,
    text: str,
    confidence: float = 0.9,
    source: str = "incident",
    category: str = "",
    keywords: str = "",
    reminder: str = "",
) -> str:
    """Record a hard-won lesson in the user model.

    Lessons are persisted with a `lesson.<slug>` key and surfaced in every
    future onboarding briefing. Use this when something goes wrong and the
    session learns a rule that should never be violated again.

    Args:
        slug: Short identifier, snake_case (e.g. "verify_before_pr").
        text: The lesson — what to do or what not to do.
        confidence: How strongly to hold this (0-1, default 0.9).
        source: Origin — "incident", "observed", or "manual".
        category: Lesson category — "tooling", "workflow", "environment",
            "api", or "architecture". Groups lessons in the briefing.
        keywords: Comma-separated trigger words. When these words appear in
            a tool call's arguments or result, the reminder is injected.
        reminder: One-line reminder shown on keyword match. Keep it short.
    """
    if not slug or not text:
        return json.dumps({"error": "slug and text are required"})
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None
    lesson = record_lesson(
        runtime.db, slug, text,
        confidence=confidence, source=source,
        category=category, keywords=kw_list, reminder=reminder,
    )
    # Invalidate the keyword cache so the new lesson is picked up immediately.
    global _lesson_keyword_cache
    _lesson_keyword_cache = {"lessons": [], "cached_at": 0.0}
    return json.dumps(lesson, default=str, indent=2)


def maos_list_lessons(runtime: Runtime) -> str:
    """List all recorded lessons.

    Read-only — returns every lesson stored in the user model, sorted by
    confidence (highest first).
    """
    lessons = list_lessons(runtime.db)
    return json.dumps(
        {"count": len(lessons), "lessons": lessons},
        default=str, indent=2,
    )


# --- Decision audit log tools -----------------------------------------------

VALID_DECISION_KINDS = ("decision", "observation", "escalation", "veto")


def maos_record_decision(
    runtime: Runtime,
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
    and reproducible. Decisions appear in the context packet and the onboarding
    briefing.

    Args:
        agent: Identifier of the agent making the decision (e.g. 'devin',
            'claude-3', 'human:david').
        action: The action being decided on (e.g. 'commit', 'deploy',
            'merge', 'escalate').
        rationale: Why the decision was made. This is the auditable reasoning
            trail.
        kind: Decision category. 'decision' = chose an action (default),
            'observation' = noted a fact, 'escalation' = sent to human,
            'veto' = blocked an action.
        tags: Comma-separated free-form tags for grouping decisions.
        confidence: Confidence level 0-1 (default 0).
    """
    if not agent or not action or not rationale:
        return json.dumps({"error": "agent, action, and rationale are required"})
    if kind not in VALID_DECISION_KINDS:
        return json.dumps({"error": f"kind must be one of: {', '.join(VALID_DECISION_KINDS)}"})

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    decision_id = runtime.db.log_decision(
        agent=agent,
        decision=action,
        reasoning=rationale,
        confidence=confidence,
        kind=kind,
        tags=tag_list,
    )

    # Fetch the stored record to return the digest + chain pointer.
    row = runtime.db.query_one("SELECT * FROM decisions WHERE id = ?", (decision_id,))
    receipt = {
        "id": decision_id,
        "digest": row["digest"],
        "previousDigest": row["prev_digest"],
        "timestamp": row["timestamp"],
        "agent": row["agent"],
        "action": row["decision"],
        "kind": row["kind"],
        "tags": json.loads(row["tags"] or "[]"),
    }
    return json.dumps(receipt, indent=2)


def maos_query_decisions(
    runtime: Runtime,
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
    if kind and kind not in VALID_DECISION_KINDS:
        return json.dumps({"error": f"kind must be one of: {', '.join(VALID_DECISION_KINDS)}"})
    limit = max(1, min(limit, 1000))

    sql = "SELECT id, timestamp, agent, decision, kind, tags, digest, prev_digest FROM decisions"
    conditions: list[str] = []
    params: list[Any] = []
    if agent:
        conditions.append("agent = ?")
        params.append(agent)
    if kind:
        conditions.append("kind = ?")
        params.append(kind)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = runtime.db.query(sql, tuple(params))
    decisions = [
        {
            "id": r["id"],
            "digest": r["digest"],
            "previousDigest": r["prev_digest"],
            "timestamp": r["timestamp"],
            "agent": r["agent"],
            "action": r["decision"],
            "kind": r["kind"],
            "tags": json.loads(r["tags"] or "[]"),
        }
        for r in rows
    ]
    return json.dumps({"count": len(decisions), "decisions": decisions}, indent=2)


def maos_verify_decisions(runtime: Runtime) -> str:
    """Verify the integrity of the decision hash chain.

    Checks that every decision's digest matches its recomputed value and that
    the chain links are intact. Returns valid=True if the chain is tamper-free.

    Use this to detect tampering with the audit log.
    """
    result = runtime.db.verify_decision_chain()
    return json.dumps(result, indent=2)


# --- Email tools ------------------------------------------------------------


def maos_send_email(
    runtime: Runtime, subject: str, body: str, to: str = "",
) -> str:
    """Send an email via Gmail.

    Uses OAuth credentials from ~/.gmail-mcp/. Tokens are refreshed
    automatically. The default recipient is read from the
    MAOS_EMAIL_RECIPIENT environment variable or the 'to' argument.

    Args:
        subject: Email subject line.
        body: Plain text body.
        to: Recipient email.
    """
    try:
        result = _send_email(subject, body, to=to)
        return json.dumps(result, default=str, indent=2)
    except EmailError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_escalate(
    runtime: Runtime,
    subject: str,
    body: str,
    context: dict[str, Any] | None = None,
    lesson_slug: str = "",
) -> str:
    """Send an escalation email — 'this needs human attention'.

    This is the anti-vibe-overload channel. When something isn't working,
    when a verification fails, when context is getting diluted with false
    confidence, escalate instead of glossing over it.

    The email is sent to the configured MAOS email recipient with a [MAOS]
    prefix. Optional structured context is appended as a section.

    Args:
        subject: Short summary of what needs attention.
        body: The detailed explanation of what happened.
        context: Optional structured context dict to append.
        lesson_slug: Optional lesson slug if related to a recorded lesson.
    """
    try:
        result = _escalate(
            subject,
            body,
            context=context,
            lesson_slug=lesson_slug or None,
        )
        return json.dumps(result, default=str, indent=2)
    except EmailError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# --- Google Drive tools -----------------------------------------------------


def maos_drive_list(
    runtime: Runtime, folder_id: str = "", page_size: int = 50, query: str = "",
) -> str:
    """List files in a Google Drive folder.

    Args:
        folder_id: Folder ID to list (default: root).
        page_size: Max files to return.
        query: Optional Drive query string.
    """
    try:
        files = _drive_list(
            folder_id=folder_id or None,
            page_size=min(page_size, 200),
            query=query or None,
        )
        return json.dumps({"success": True, "files": files, "count": len(files)}, default=str, indent=2)
    except DriveError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_drive_search(runtime: Runtime, query: str, page_size: int = 20) -> str:
    """Search Google Drive by name or content query.

    Args:
        query: Search query (e.g. "name contains 'report'").
        page_size: Max results.
    """
    try:
        files = _drive_search(query=query, page_size=min(page_size, 100))
        return json.dumps({"success": True, "files": files, "count": len(files)}, default=str, indent=2)
    except DriveError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_drive_upload(
    runtime: Runtime,
    name: str,
    content_b64: str,
    mime_type: str = "application/octet-stream",
    parent_folder_id: str = "",
) -> str:
    """Upload a file to Google Drive (5TB Workspace account).

    Args:
        name: Remote file name on Drive.
        content_b64: File content as base64-encoded string.
        mime_type: MIME type.
        parent_folder_id: Folder ID to upload into (default: root).
    """
    try:
        import base64

        content = base64.b64decode(content_b64)
        result = _drive_upload(
            name=name,
            content=content,
            mime_type=mime_type,
            parent_folder_id=parent_folder_id or None,
        )
        return json.dumps({"success": True, "file": result}, default=str, indent=2)
    except DriveError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_drive_download(runtime: Runtime, file_id: str) -> str:
    """Download a file from Google Drive by its file ID.

    Args:
        file_id: Google Drive file ID.
    """
    try:
        import base64

        content = _drive_download(file_id)
        return json.dumps(
            {"success": True, "file_id": file_id, "content_b64": base64.b64encode(content).decode(), "size": len(content)},
            indent=2,
        )
    except DriveError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_drive_mkdir(runtime: Runtime, name: str, parent_folder_id: str = "") -> str:
    """Create a folder on Google Drive (or get existing if it already exists).

    Args:
        name: Folder name to create.
        parent_folder_id: Parent folder ID (default: root).
    """
    try:
        result = _drive_mkdir(name=name, parent_folder_id=parent_folder_id or None)
        return json.dumps({"success": True, "folder": result}, default=str, indent=2)
    except DriveError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_drive_quota(runtime: Runtime) -> str:
    """Get Google Drive storage quota info for the configured account."""
    try:
        result = _drive_quota()
        return json.dumps({"success": True, "quota": result}, default=str, indent=2)
    except DriveError as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# --- Guard tools ------------------------------------------------------------


def maos_scan_vibes(runtime: Runtime, text: str) -> str:
    """Scan text for vibe language — unsupported success claims.

    This is the anti-vibe-overload filter. It catches phrases like
    "everything is working", "verified" without evidence, "all tests pass"
    without output, "looks good", and sub-agent agreement used as proof.

    Returns PASS, WARN, or BLOCK with flagged phrases and evidence required.

    Args:
        text: The text to scan (agent output, PR description, commit message).
    """
    if not text:
        return json.dumps({"error": "text is required"})
    result = scan_vibes(text)
    return json.dumps(result.to_dict(), indent=2)


def maos_check_guards(
    runtime: Runtime,
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

    This is the comprehensive pre-action check. Run it before opening a PR,
    committing a fix, or declaring success. If any guard returns BLOCK,
    do NOT proceed — fix the issues and re-run.

    Guards:
    1. VibeLanguageGuard — scans text for unsupported success claims
    2. PrPreflightGuard — hard checks before PR creation (if is_pr=True)
    3. LessonComplianceGuard — checks if action violates recorded lessons
    4. SubagentProofGuard — flags sub-agent agreement as sole evidence

    Args:
        text: The text to scan for vibes.
        action_description: What the agent is about to do.
        is_pr: Whether this is a PR-related action.
        is_bug_fix: Whether this is a bug fix (requires manual verification).
        has_test_output: Whether test output is provided.
        has_build_output: Whether build output is provided.
        has_manual_verification: Whether manual verification was done.
        claims_subagent_agreement: Whether sub-agent agreement is claimed.
        has_empirical_evidence: Whether empirical evidence is provided.
    """
    results = run_all_guards(
        runtime.db,
        text,
        action_description=action_description,
        is_pr=is_pr,
        is_bug_fix=is_bug_fix,
        has_test_output=has_test_output,
        has_build_output=has_build_output,
        has_manual_verification=has_manual_verification,
        claims_subagent_agreement=claims_subagent_agreement,
        has_empirical_evidence=has_empirical_evidence,
    )
    blocked, reasons = should_block(results)
    report = format_guard_report(results)
    return json.dumps({
        "blocked": blocked,
        "report": report,
        "results": [r.to_dict() for r in results],
    }, indent=2)


def maos_verify_command(
    runtime: Runtime,
    command: str,
    cwd: str = "",
    timeout: int = 300,
) -> str:
    """Execute a verification command and capture evidence.

    This is the evidence-PRODUCING verification tool. The system runs the
    command, not the agent. The agent cannot fake the exit code, stdout, or
    the commit hash at execution time. Use this instead of self-attested
    booleans like `has_test_output=True`.

    SECURITY: Only verification commands are allowed (test runners, linters,
    type checkers, build commands). The command must start with one of the
    allowed prefixes. Arbitrary shell commands are rejected.

    Allowed command prefixes:
        uv run pytest, uv run ruff, uv run mypy, uv run pyright,
        uv run pytest--, pytest, ruff, mypy, pyright,
        npm test, npm run lint, npm run typecheck, npm run build,
        cargo test, cargo clippy, cargo build, cargo check,
        go test, go vet, go build,
        make test, make lint, make check, make build,
        git status, git diff --stat

    Args:
        command: The verification command to execute (must be in the allowlist).
        cwd: Working directory (defaults to current directory).
        timeout: Maximum execution time in seconds.
    """
    from app.guards_evidence import capture_command, evidence_to_json
    if not command:
        return json.dumps({"error": "command is required"})

    # Security: restrict to verification commands only.
    # This prevents the tool from being abused to run arbitrary shell commands
    # (rm -rf, curl exfiltration, etc.) while still allowing the core use case
    # of capturing test/lint/build evidence.
    ALLOWED_PREFIXES = (
        "uv run pytest",
        "uv run ruff",
        "uv run mypy",
        "uv run pyright",
        "uv run pytest--",  # pytest with flags
        "pytest",
        "ruff",
        "mypy",
        "pyright",
        "npm test",
        "npm run lint",
        "npm run typecheck",
        "npm run build",
        "npm run check",
        "cargo test",
        "cargo clippy",
        "cargo build",
        "cargo check",
        "go test",
        "go vet",
        "go build",
        "make test",
        "make lint",
        "make check",
        "make build",
        "git status",
        "git diff --stat",
    )
    normalized = command.strip()
    if not any(normalized.startswith(p) for p in ALLOWED_PREFIXES):
        return json.dumps({
            "error": "command not in verification allowlist",
            "command": command,
            "hint": "Only test/lint/typecheck/build commands are allowed. "
                    "Use the CLI (guards verify) for unrestricted command execution.",
            "allowed_prefixes": list(ALLOWED_PREFIXES),
        }, indent=2)

    evidence = capture_command(command, cwd=cwd or None, timeout=timeout)
    return evidence_to_json(evidence)


def maos_inspect_diff(
    runtime: Runtime,
    repo_dir: str = "",
    base: str = "HEAD",
    stated_scope: str = "",
) -> str:
    """Inspect the git diff for risk patterns (secrets, debug, MCP changes, etc.).

    BLOCK for secrets/credentials. WARN for debug statements, dependency
    changes, binary files, scope violations, large diffs. REQUIRES_REVIEW
    for MCP tool/schema changes and security-sensitive paths.

    Args:
        repo_dir: Path to the git repository (defaults to cwd).
        base: Base ref to diff against (default "HEAD").
        stated_scope: The stated scope of the PR (for unrelated-file detection).
    """
    import os

    from app.guards_diff import inspect_diff
    repo = repo_dir or os.getcwd()
    result = inspect_diff(repo, base, stated_scope=stated_scope)
    return json.dumps(result.to_dict(), indent=2)


def maos_pr_readiness(
    runtime: Runtime,
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
    """Assess PR readiness via the state machine: DRAFT -> UNDERSTOOD -> VERIFIED -> PR_READY.

    This is the comprehensive readiness check. It requires evidence objects
    (produced by maos_verify_command), a structured PR description, and
    inspects the diff. If the state is not PR_READY, do NOT open the PR.

    The PR description must contain sections: Summary, Why, What Changed,
    Scope. For bug fixes also: Root Cause, Regression Test, Before, After.

    Args:
        repo_dir: Path to the git repository (defaults to cwd).
        pr_description: The PR description text (markdown with required sections).
        is_bug_fix: Whether this is a bug fix (requires manual verification).
        test_evidence_json: JSON evidence from maos_verify_command (tests).
        lint_evidence_json: JSON evidence from maos_verify_command (lint).
        build_evidence_json: JSON evidence from maos_verify_command (build).
        manual_evidence_json: JSON manual verification evidence.
        diff_base: Base ref for diff inspection.
        skip_diff: Skip diff inspection.
    """
    import os

    from app.guards_evidence import (
        CommandEvidence,
        ManualVerificationEvidence,
        load_evidence,
    )
    from app.guards_readiness import PRContent, assess_readiness

    repo = repo_dir or os.getcwd()

    # Load evidence objects from JSON
    test_ev = None
    lint_ev = None
    build_ev = None
    manual_ev = None
    if test_evidence_json:
        ev = load_evidence(test_evidence_json)
        if isinstance(ev, CommandEvidence):
            test_ev = ev
    if lint_evidence_json:
        ev = load_evidence(lint_evidence_json)
        if isinstance(ev, CommandEvidence):
            lint_ev = ev
    if build_evidence_json:
        ev = load_evidence(build_evidence_json)
        if isinstance(ev, CommandEvidence):
            build_ev = ev
    if manual_evidence_json:
        ev = load_evidence(manual_evidence_json)
        if isinstance(ev, ManualVerificationEvidence):
            manual_ev = ev

    # Parse PR content
    content = None
    if pr_description:
        content = PRContent.from_markdown(pr_description)

    report = assess_readiness(
        repo_dir=repo,
        content=content,
        is_bug_fix=is_bug_fix,
        test_evidence=test_ev,
        build_evidence=build_ev,
        lint_evidence=lint_ev,
        manual_evidence=manual_ev,
        diff_base=diff_base,
        skip_diff=skip_diff,
    )
    return json.dumps(report.to_dict(), default=str, indent=2)


# --- Diagnostic tools -------------------------------------------------------


def maos_diagnostics(runtime: Runtime, format: str = "text") -> str:
    """Check functionality and implementation status of all MAOS tools.

    Actually calls each read-only tool and verifies it returns valid JSON.
    Side-effect tools (email, task creation) are checked for handler
    existence but not called. Returns a per-tool status report.

    Args:
        format: "text" (default, readable report) or "json" (structured).
    """
    report = run_diagnostics(runtime)
    if format == "json":
        return json.dumps(report.to_dict(), default=str, indent=2)
    return format_diagnostic_report(report)


# --- Time awareness tools ---------------------------------------------------


def maos_time(runtime: Runtime, format: str = "text") -> str:
    """Get time awareness information for the current AI session.

    AI agents have no native concept of time. This tool gives the agent:
    - Current wall-clock time (with timezone and human-readable format)
    - Session elapsed time (how long this Devin session has been running)
    - Session start time (when the first tool call in this session happened)
    - Tool call count and frequency (calls per minute)
    - Idle time (how long since the last tool call)
    - Reading profile (files read, re-read ratio, reading velocity)
    - Relative time hints ("3 minutes since last tool call")

    A "session" is defined as a contiguous sequence of MCP tool calls with
    no gap longer than 10 minutes. If you step away for 10+ minutes, a new
    session starts on your next call.

    Args:
        format: "text" (default, human-readable) or "json" (structured).
    """
    from app.time import build_time_packet, format_time_packet

    packet = build_time_packet(runtime.db)
    if format == "json":
        return json.dumps(packet, default=str, indent=2)
    return format_time_packet(packet)


# --- Network intelligence tools --------------------------------------------


def maos_network_status(runtime: Runtime, format: str = "text") -> str:
    """Get current network health and bandwidth status.

    Read-only — queries the latest network sample from the DB. The network
    collector must be running (`maos network collect` or the daemon) for
    fresh data; otherwise this returns the last known state.

    Returns:
        - healthy: bool — is the network usable for cloud LLM calls?
        - interface: the active interface (en0, en6, etc.)
        - network_type: wifi / ethernet / unknown
        - connection_state: connected / disconnected / unknown
        - rx_rate_bps / tx_rate_bps: current bandwidth (bits/sec)
        - latency_ms: TCP RTT probe latency (best-effort)
        - packet_loss: interface error rate (best-effort, fraction)
        - bandwidth_quality: 0-100 health score
        - issues: list of detected problems

    Pass format='json' for structured output.
    """
    status = runtime.network.get_status()
    if format == "json":
        return json.dumps(status.to_dict(), default=str, indent=2)

    lines = [
        f"Healthy:     {status.healthy}",
        f"Interface:   {status.interface} ({status.network_type.value})",
        f"State:       {status.connection_state.value}",
        f"RX rate:     {_fmt_bps(status.rx_rate_bps)}",
        f"TX rate:     {_fmt_bps(status.tx_rate_bps)}",
        f"Latency:     {_fmt_latency(status.latency_ms)}",
        f"Packet loss: {_fmt_loss(status.packet_loss)}",
        f"Quality:     {status.bandwidth_quality}/100",
    ]
    if status.issues:
        lines.append(f"Issues:      {', '.join(status.issues)}")
    if status.timestamp:
        lines.append(f"Sampled:     {status.timestamp[:19]}")
    return "\n".join(lines)


def maos_network_history(
    runtime: Runtime,
    hours: int = 24,
    resolution: str = "hourly",
) -> str:
    """Get historical network bandwidth and latency data.

    Read-only — queries aggregated network history from the DB.

    Args:
        hours: Lookback window in hours (default 24).
        resolution: 'hourly' (default, indefinite retention) or 'minute'
            (30-day retention, more detail).

    Returns JSON with: totals (rx/tx bytes), series (per-bucket aggregates),
    and events (interruptions, degradation, interface changes).
    """
    history = runtime.network.get_history(hours=hours, resolution=resolution)
    return json.dumps(history, default=str, indent=2)


def _fmt_bps(bps: float) -> str:
    if bps <= 0:
        return "0 bps"
    units = ["bps", "Kbps", "Mbps", "Gbps"]
    i = 0
    v = float(bps)
    while v >= 1000 and i < len(units) - 1:
        v /= 1000
        i += 1
    return f"{v:.1f} {units[i]}"


def _fmt_latency(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    return f"{ms:.0f} ms"


def _fmt_loss(loss: float | None) -> str:
    if loss is None:
        return "n/a"
    return f"{loss * 100:.3f}%"


def maos_read_file(runtime: Runtime, path: str, max_lines: int = 2000) -> str:
    """Read a file with time awareness and reading behavior tracking.

    This is the time-aware version of file reading. It returns the file
    content PLUS a time awareness footer showing how long the session has
    been running, how many files have been read, and whether the agent is
    re-reading the same files (a sign of going in circles).

    Reading is the #1 repeated action for AI agents. By routing reads
    through this tool, every read becomes a time check and a behavioral
    checkpoint. The agent sees:

      [⏱ 45m · 23 calls · 12 files read (8 unique, 33% re-reads) · active 2m]

    If the re-read ratio is high (>50%), the agent is likely going in
    circles and should step back and reconsider its approach.

    Args:
        path: Absolute or relative path to the file to read.
        max_lines: Maximum lines to read (default 2000, same as built-in read).
    """
    from pathlib import Path

    from app.time import detect_session_start, reading_profile

    # Resolve path
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p

    try:
        if not p.exists():
            return json.dumps({"error": f"file not found: {path}"})
        if not p.is_file():
            return json.dumps({"error": f"not a file: {path}"})

        # Read the file
        content = p.read_text(errors="replace")
        lines = content.split("\n")
        line_count = len(lines)

        # Truncate if needed
        if max_lines and line_count > max_lines:
            content = "\n".join(lines[:max_lines])
            content += f"\n... ({line_count - max_lines} more lines, use max_lines to see more)"

        # Log the read for behavior tracking
        runtime.db.log_file_read(path=str(p), lines=min(line_count, max_lines))

        # Build time + reading awareness
        session = detect_session_start(runtime.db)
        reading = reading_profile(runtime.db)

        elapsed = session["elapsed_compact"]
        calls = session["tool_call_count"]
        files_read = reading["files_read"]
        unique = reading["unique_files"]
        re_read_pct = int(reading["re_read_ratio"] * 100)

        # Build the response with content + time awareness
        result = {
            "path": str(p),
            "lines": line_count,
            "content": content,
            "_time_awareness": {
                "elapsed": elapsed,
                "tool_calls": calls,
                "idle": session.get("idle_compact", "0s"),
                "active": session["is_active"],
                "files_read": files_read,
                "unique_files": unique,
                "re_read_ratio": reading["re_read_ratio"],
                "reads_per_minute": reading["reads_per_minute"],
                "assessment": reading["assessment"],
            },
        }

        # Add a warning if re-reading too much
        if reading["re_read_ratio"] > 0.5 and files_read > 5:
            result["_warning"] = (
                f"Re-reading {re_read_pct}% of files — you may be going in circles. "
                f"Consider stepping back and reconsidering your approach."
            )

        return json.dumps(result, default=str, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(p)})


# --- Storage management tools ------------------------------------------------


def maos_storage_scan(
    runtime: Runtime,
    mode: str = "full",
    format: str = "text",
) -> str:
    """Run a comprehensive storage analysis on the Mac.

    Decomposes the mysterious macOS "System Data" category into
    identifiable, reclaimable components. Combines:

    1. Native macOS introspection (tmutil, diskutil, /private/var/vm,
       unified log archive, Spotlight index, APFS snapshots)
    2. X-MaC disk engine (top directories/files by physical size)
    3. X-MaC clean engine (reclaimable caches, build artifacts, trash)
    4. NeoDisk scan cache detection
    5. User-managed cache scan (uv, huggingface, npm, cargo, etc.)

    Modes:
      - "full": Everything (default). Includes xmac disk + clean.
      - "fast": Native macOS only — no filesystem walk. Quick system
        data decomposition without xmac.
      - "system": Only the system data decomposition (no user caches,
        no xmac). Focused on the "System Data" mystery.
      - "caches": Only user cache scan.

    Args:
        mode: Scan mode — "full", "fast", "system", or "caches".
        format: "text" (human-readable report) or "json" (structured).
    """
    from app.storage import (
        decompose_system_data,
        format_storage_report,
        get_volume_stats,
        scan_storage,
        scan_user_caches,
    )

    if mode == "full":
        report = scan_storage(include_xmac=True)
    elif mode == "fast":
        report = scan_storage(include_xmac=False)
    elif mode == "system":
        from app.storage import StorageReport

        vol = get_volume_stats()
        sd = decompose_system_data()
        total_sys = sum(e.size_bytes for e in sd)
        total_rec = sum(e.size_bytes for e in sd if e.reclaimable)
        report = StorageReport(
            volume=vol,
            system_data=sd,
            total_system_data_bytes=total_sys,
            total_reclaimable_bytes=total_rec,
        )
    elif mode == "caches":
        from app.storage import StorageReport

        vol = get_volume_stats()
        uc = scan_user_caches()
        total_cac = sum(e.size_bytes for e in uc)
        total_rec = sum(e.size_bytes for e in uc if e.safe_to_clear)
        report = StorageReport(
            volume=vol,
            user_caches=uc,
            total_cache_bytes=total_cac,
            total_reclaimable_bytes=total_rec,
        )
    else:
        return json.dumps({"error": f"unknown mode: {mode}"})

    if format == "json":
        return json.dumps(report.to_dict(), default=str, indent=2)
    return format_storage_report(report)


def maos_storage_system_data(runtime: Runtime, format: str = "text") -> str:
    """Decompose macOS "System Data" into identifiable components.

    This is the focused tool for solving the "120 GB of system data"
    mystery. It checks every known location that macOS categorizes as
    "System Data" or "Other" in Storage settings:

    - sleepimage (RAM dump on sleep)
    - swapfiles (virtual memory)
    - unified log archive (diagnostics + uuidtext)
    - system logs (install.log, wifi.log, powermanagement)
    - Spotlight index
    - Asset cache (content caching)
    - Time Machine local snapshots
    - APFS OS update snapshots
    - per-user temp files (/private/var/folders)
    - system databases (/private/var/db)

    Each entry reports: actual size, whether it's reclaimable, whether
    it's safe to delete, and the exact command to reclaim it.

    Args:
        format: "text" (human-readable) or "json" (structured).
    """
    from app.storage import (
        StorageReport,
        decompose_system_data,
        format_storage_report,
        get_volume_stats,
    )

    vol = get_volume_stats()
    system_data = decompose_system_data()
    total_system = sum(e.size_bytes for e in system_data)
    total_reclaimable = sum(
        e.size_bytes for e in system_data if e.reclaimable
    )
    report = StorageReport(
        volume=vol,
        system_data=system_data,
        total_system_data_bytes=total_system,
        total_reclaimable_bytes=total_reclaimable,
    )

    if format == "json":
        return json.dumps(report.to_dict(), default=str, indent=2)
    return format_storage_report(report)


def maos_storage_fix_script(
    runtime: Runtime,
    output_path: str = "./storage-fixes.sh",
    include_caches: bool = True,
    include_system: bool = True,
    include_tm: bool = True,
) -> str:
    """Generate a reviewable remediation shell script for storage cleanup.

    ALL destructive commands are COMMENTED OUT in the generated script.
    The user reviews, uncomments what they want, and runs it. This tool
    NEVER executes destructive commands — it only produces the script.

    The script includes:
    - System data cleanup (sleepimage, log archive, Spotlight, etc.)
    - Time Machine snapshot deletion commands
    - User cache cleanup (uv, npm, cargo, brew, etc.)
    - X-MaC clean engine findings reference

    Args:
        output_path: Where to write the fix script.
        include_caches: Include user cache cleanup commands.
        include_system: Include system data cleanup commands.
        include_tm: Include Time Machine snapshot deletion commands.
    """
    from app.storage import generate_fix_script, scan_storage

    report = scan_storage(include_xmac=False)
    fix = generate_fix_script(
        report,
        output_path,
        include_caches=include_caches,
        include_system=include_system,
        include_tm=include_tm,
    )

    result = {
        "path": fix.path,
        "total_reclaimable_bytes": fix.total_reclaimable_bytes,
        "commands": fix.commands,
        "warnings": fix.warnings,
        "instructions": (
            f"Review {fix.path}, uncomment the commands you want to run, "
            f"then execute: bash {fix.path}"
        ),
    }
    return json.dumps(result, indent=2)


def maos_storage_maintain(runtime: Runtime) -> str:
    """Run safe macOS maintenance tasks via X-MaC.

    Delegates to ``xmac maintain`` which runs:
    - Flush DNS cache
    - Reindex Spotlight
    - Rebuild LaunchServices
    - Run periodic scripts
    - Purge RAM (inactive memory)
    - Clear Quick Look cache

    Safe tasks run automatically. Sudo-requiring tasks are emitted as
    findings for the user to review and run manually.
    """
    from app.storage import run_maintenance

    result = run_maintenance()
    return json.dumps(result, default=str, indent=2)


def maos_storage_delete_tm_snapshot(
    runtime: Runtime, date: str
) -> str:
    """Delete a Time Machine local snapshot by date.

    This is the #1 way to reclaim "System Data" space on macOS. Local
    snapshots pin purgeable space that only clears on the next backup.
    Deleting them reclaims the space immediately.

    Use ``maos_storage_scan`` first to list available snapshots and
    their dates.

    Args:
        date: The snapshot date string, e.g. "2026-07-24-152803".
    """
    from app.storage import delete_tm_snapshot

    success, message = delete_tm_snapshot(date)
    return json.dumps(
        {"success": success, "message": message, "date": date}
    )


def maos_storage_hidden(runtime: Runtime, format: str = "text") -> str:
    """Scan for hidden data consumers that macOS categorizes as 'System Data'.

    This goes beyond the standard system data decomposition and user caches.
    It finds the 'hidden' consumers that don't show up in standard disk
    usage analysis because they're in non-obvious locations:

    - Devin CLI session database (can grow to 3GB+ unbounded)
    - Devin CLI corrupt backup files, logs, and summaries
    - AI/ML tool data (LM Studio, Codex, Hermes, MLX, Ollama)
    - Developer tool data (uv tools, pipx, nvm, mise, pnpm store)
    - App support directories (Chrome, Warp, Cursor, Discord, etc.)
    - Cloud storage sync (Proton Drive, iCloud)
    - SDKs (Android SDK)
    - App containers (PlayCover, mediaanalysis)
    - System-level data (/Library/Updates, Xcode CLT, system assets)

    Each entry reports: actual size, whether it's safe to clear, whether
    it regenerates, and the exact command to reclaim it.

    Args:
        format: "text" (human-readable) or "json" (structured).
    """
    from app.storage import (
        StorageReport,
        format_storage_report,
        get_volume_stats,
        scan_hidden_data,
    )

    vol = get_volume_stats()
    hidden = scan_hidden_data()
    total_hidden = sum(e.size_bytes for e in hidden)
    total_rec = sum(e.size_bytes for e in hidden if e.safe_to_clear)
    report = StorageReport(
        volume=vol,
        user_caches=hidden,
        total_cache_bytes=total_hidden,
        total_reclaimable_bytes=total_rec,
    )

    if format == "json":
        return json.dumps(report.to_dict(), default=str, indent=2)
    return format_storage_report(report)


# =============================================================================
# Coding Agent Tools — file ops, search, shell, web, notebook, git, subagent
# These mirror Devin CLI's built-in tools so MAOS can act as a full coding agent
# when accessed via MCP by external clients (Grok web, other agents, etc.)
# =============================================================================


# --- File operation tools ---------------------------------------------------


def maos_write_file(runtime: Runtime, path: str, content: str) -> str:
    """Write content to a file, creating it if needed.

    Creates parent directories if they don't exist. Overwrites existing files.

    Args:
        path: Absolute or relative path to the file to write.
        content: The content to write to the file.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content)
        return json.dumps({
            "path": str(p),
            "bytes": len(content.encode()),
            "created": not existed,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(p)})


def maos_edit_file(
    runtime: Runtime,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit a file by replacing old_string with new_string.

    Performs exact string replacement. The old_string must be unique in the
    file unless replace_all=True.

    Args:
        path: Absolute or relative path to the file to edit.
        old_string: The exact text to replace.
        new_string: The text to replace it with.
        replace_all: If True, replace all occurrences. Default False.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p

    try:
        if not p.exists():
            return json.dumps({"error": f"file not found: {path}"})
        if not p.is_file():
            return json.dumps({"error": f"not a file: {path}"})

        content = p.read_text()
        count = content.count(old_string)

        if count == 0:
            return json.dumps({"error": "old_string not found in file"})
        if count > 1 and not replace_all:
            return json.dumps({
                "error": f"old_string appears {count} times — use replace_all=True or provide more context",
            })

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        p.write_text(new_content)

        return json.dumps({
            "path": str(p),
            "replacements": count if replace_all else 1,
            "success": True,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(p)})


# --- Search tools -----------------------------------------------------------


def maos_grep(
    runtime: Runtime,
    pattern: str,
    path: str = ".",
    glob_pattern: str = "",
    case_insensitive: bool = False,
    max_results: int = 100,
    context_lines: int = 0,
) -> str:
    """Search file contents using ripgrep.

    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search in (default current directory).
        glob_pattern: File pattern filter (e.g. '*.py').
        case_insensitive: Case-insensitive matching (default False).
        max_results: Maximum results to return (default 100).
        context_lines: Lines of context before/after each match (default 0).
    """
    import subprocess
    from pathlib import Path

    search_path = Path(path).expanduser()
    if not search_path.is_absolute():
        search_path = Path.cwd() / search_path

    cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
    if case_insensitive:
        cmd.append("-i")
    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])
    if glob_pattern:
        cmd.extend(["-g", glob_pattern])
    cmd.extend(["--max-count", str(max_results)])
    cmd.append(pattern)
    cmd.append(str(search_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return json.dumps({
            "pattern": pattern,
            "path": str(search_path),
            "matches": len(lines),
            "results": lines[:max_results],
            "truncated": len(lines) > max_results,
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "search timed out (30s)"})
    except FileNotFoundError:
        return json.dumps({"error": "ripgrep (rg) not installed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def maos_glob(
    runtime: Runtime,
    pattern: str,
    path: str = ".",
) -> str:
    """Find files by glob pattern.

    Supports brace expansion (e.g. '**/*.{ts,tsx}').

    Args:
        pattern: Glob pattern (e.g. '**/*.py', '*.json').
        path: Directory to search in (default current directory).
    """
    import glob as glob_mod
    from pathlib import Path

    search_path = Path(path).expanduser()
    if not search_path.is_absolute():
        search_path = Path.cwd() / search_path

    try:
        full_pattern = str(search_path / pattern)
        matches = sorted(glob_mod.glob(full_pattern, recursive=True))
        return json.dumps({
            "pattern": pattern,
            "path": str(search_path),
            "count": len(matches),
            "files": matches[:500],
            "truncated": len(matches) > 500,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def maos_find_file(
    runtime: Runtime,
    pattern: str,
    path: str = ".",
) -> str:
    """Find files by name pattern using recursive glob matching.

    Args:
        pattern: File name pattern (e.g. '*.py', 'test_*').
        path: Directory to search in (default current directory).
    """
    from pathlib import Path

    search_path = Path(path).expanduser()
    if not search_path.is_absolute():
        search_path = Path.cwd() / search_path

    try:
        matches = sorted(search_path.rglob(pattern))
        return json.dumps({
            "pattern": pattern,
            "path": str(search_path),
            "count": len(matches),
            "files": [str(m) for m in matches[:500]],
            "truncated": len(matches) > 500,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Shell execution tool ---------------------------------------------------

_BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf ~", "rm -rf /*", "rm -rf ~/*",
    "mkfs", "dd if=/dev/zero of=/dev/",
    "chmod -R 777 /",
}

_RESTRICTED_PREFIXES = {
    "sudo ", "su ", "kill -9",
    "shutdown", "reboot", "halt",
    "launchctl unload",
}


def maos_run_command(
    runtime: Runtime,
    command: str,
    cwd: str = "",
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> str:
    """Execute a shell command and return stdout, stderr, and exit code.

    Blocked commands include rm -rf /, mkfs, and other destructive operations.
    Restricted commands (sudo, kill -9, shutdown) are rejected.

    Args:
        command: The shell command to execute.
        cwd: Working directory (default current directory).
        timeout: Timeout in seconds (default 120).
        env: Additional environment variables.
    """
    import os
    import subprocess
    from pathlib import Path

    cmd_lower = command.lower().strip()
    for blocked in _BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            return json.dumps({"error": f"blocked: dangerous pattern '{blocked}'", "command": command})

    for prefix in _RESTRICTED_PREFIXES:
        if cmd_lower.startswith(prefix):
            return json.dumps({"error": f"restricted: prefix '{prefix}' requires user confirmation", "command": command})

    work_dir = Path(cwd).expanduser() if cwd else Path.cwd()
    if not work_dir.is_absolute():
        work_dir = Path.cwd() / work_dir

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=str(work_dir), timeout=timeout, env=full_env,
        )
        return json.dumps({
            "command": command,
            "cwd": str(work_dir),
            "exit_code": result.returncode,
            "stdout": result.stdout[:50000],
            "stderr": result.stderr[:50000],
            "stdout_truncated": len(result.stdout) > 50000,
            "stderr_truncated": len(result.stderr) > 50000,
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"command timed out after {timeout}s", "command": command})
    except Exception as e:
        return json.dumps({"error": str(e), "command": command})


# --- Web tools --------------------------------------------------------------


def maos_web_search(runtime: Runtime, query: str, num_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: Search query string.
        num_results: Maximum results to return (default 5).
    """
    try:
        from app.tools.builtin import WebSearchTool
        tool = WebSearchTool()
        result = tool.execute({"query": query, "num_results": num_results})
        if result.ok:
            return json.dumps({"results": result.output}, default=str, indent=2)
        return json.dumps({"error": result.error}, default=str, indent=2)
    except ImportError:
        import subprocess
        from pathlib import Path
        result = subprocess.run(
            ["uv", "run", "app", "search", query, "--json"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path.home() / "mac-ai-os"),
        )
        return result.stdout or json.dumps({"error": result.stderr})
    except Exception as e:
        return json.dumps({"error": str(e)})


def maos_fetch_url(runtime: Runtime, url: str) -> str:
    """Fetch a web page and return its content as readable text.

    Args:
        url: The URL to fetch.
    """
    try:
        from app.tools.builtin import WebFetchTool
        tool = WebFetchTool()
        result = tool.execute({"url": url})
        if result.ok:
            return json.dumps({"content": result.output}, default=str, indent=2)
        return json.dumps({"error": result.error}, default=str, indent=2)
    except ImportError:
        import subprocess
        from pathlib import Path
        result = subprocess.run(
            ["uv", "run", "app", "browse", url, "--json"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path.home() / "mac-ai-os"),
        )
        return result.stdout or json.dumps({"error": result.stderr})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Notebook tools ---------------------------------------------------------


def maos_notebook_read(runtime: Runtime, path: str) -> str:
    """Read a Jupyter notebook (.ipynb) and extract all cells with outputs.

    Args:
        path: Path to the .ipynb file.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p

    try:
        if not p.exists():
            return json.dumps({"error": f"file not found: {path}"})
        if p.suffix != ".ipynb":
            return json.dumps({"error": f"not a notebook: {path} (expected .ipynb)"})

        nb = json.loads(p.read_text())
        cells = []
        for i, cell in enumerate(nb.get("cells", [])):
            entry = {
                "cell_number": i,
                "cell_type": cell.get("cell_type", "code"),
                "source": "".join(cell.get("source", [])),
            }
            if cell.get("cell_type") == "code":
                outputs = []
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "stream":
                        outputs.append({"type": "stream", "name": out.get("name", ""), "text": "".join(out.get("text", []))})
                    elif out.get("output_type") == "execute_result":
                        data = out.get("data", {})
                        outputs.append({"type": "result", "text": "".join(data.get("text/plain", []))})
                    elif out.get("output_type") == "error":
                        outputs.append({"type": "error", "ename": out.get("ename", ""), "evalue": out.get("evalue", ""), "traceback": out.get("traceback", [])})
                entry["outputs"] = outputs
            cells.append(entry)

        return json.dumps({"path": str(p), "cell_count": len(cells), "cells": cells}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(p)})


def maos_notebook_edit(
    runtime: Runtime,
    path: str,
    cell_number: int,
    new_source: str,
    cell_type: str = "code",
    edit_mode: str = "replace",
) -> str:
    """Edit a cell in a Jupyter notebook.

    Args:
        path: Path to the .ipynb file.
        cell_number: 0-based index of the cell to edit.
        new_source: New source content for the cell.
        cell_type: Cell type — 'code', 'markdown', or 'raw' (for insert).
        edit_mode: 'replace' (default), 'insert', or 'delete'.
    """
    from pathlib import Path

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p

    try:
        if not p.exists() and edit_mode != "insert":
            return json.dumps({"error": f"file not found: {path}"})
        if p.exists() and p.suffix != ".ipynb":
            return json.dumps({"error": f"not a notebook: {path}"})

        if p.exists():
            nb = json.loads(p.read_text())
        else:
            nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}

        cells = nb.get("cells", [])

        if edit_mode == "delete":
            if 0 <= cell_number < len(cells):
                cells.pop(cell_number)
            else:
                return json.dumps({"error": f"cell {cell_number} out of range (0-{len(cells)-1})"})
        elif edit_mode == "insert":
            new_cell = {"cell_type": cell_type, "source": new_source.splitlines(keepends=True), "metadata": {}}
            if cell_type == "code":
                new_cell["execution_count"] = None
                new_cell["outputs"] = []
            cells.insert(cell_number, new_cell)
        else:
            if 0 <= cell_number < len(cells):
                cells[cell_number]["source"] = new_source.splitlines(keepends=True)
                if cell_type != cells[cell_number].get("cell_type", "code"):
                    cells[cell_number]["cell_type"] = cell_type
                if cell_type == "code" and "outputs" not in cells[cell_number]:
                    cells[cell_number]["outputs"] = []
                    cells[cell_number]["execution_count"] = None
            else:
                return json.dumps({"error": f"cell {cell_number} out of range (0-{len(cells)-1})"})

        nb["cells"] = cells
        p.write_text(json.dumps(nb, indent=1))

        return json.dumps({"path": str(p), "cell_count": len(cells), "edit_mode": edit_mode, "cell_number": cell_number, "success": True}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "path": str(p)})


# --- Git tools --------------------------------------------------------------


def _git_run(runtime: Runtime, args: list[str], cwd: str = "") -> tuple[int, str, str]:
    """Run a git command and return (exit_code, stdout, stderr)."""
    import subprocess
    from pathlib import Path

    work_dir = Path(cwd).expanduser() if cwd else Path.cwd()
    if not work_dir.is_absolute():
        work_dir = Path.cwd() / work_dir

    result = subprocess.run(["git"] + args, capture_output=True, text=True, timeout=30, cwd=str(work_dir))
    return result.returncode, result.stdout, result.stderr


def maos_git_status(runtime: Runtime, repo_dir: str = "") -> str:
    """Get git status for a repository.

    Args:
        repo_dir: Path to the git repository (default current directory).
    """
    code, out, err = _git_run(runtime, ["status", "--porcelain=v2", "--branch"], cwd=repo_dir)
    if code != 0:
        return json.dumps({"error": err.strip()})
    code2, out2, _ = _git_run(runtime, ["status"], cwd=repo_dir)
    return json.dumps({"repo": repo_dir or str(Path.cwd()), "porcelain": out.strip(), "status": out2.strip()}, indent=2)


def maos_git_diff(
    runtime: Runtime,
    repo_dir: str = "",
    staged: bool = False,
    base: str = "",
    target: str = "",
) -> str:
    """Get git diff for a repository.

    Args:
        repo_dir: Path to the git repository.
        staged: If True, show staged changes (default False).
        base: Base ref for comparison (e.g. 'main', 'HEAD~1').
        target: Target ref for comparison.
    """
    args = ["diff"]
    if staged:
        args.append("--staged")
    if base and target:
        args.append(f"{base}...{target}")
    elif base:
        args.append(base)

    code, out, err = _git_run(runtime, args, cwd=repo_dir)
    if code != 0:
        return json.dumps({"error": err.strip()})
    return json.dumps({"repo": repo_dir or str(Path.cwd()), "staged": staged, "base": base, "target": target, "diff": out[:50000], "truncated": len(out) > 50000}, indent=2)


def maos_git_commit(
    runtime: Runtime,
    message: str,
    repo_dir: str = "",
    add_all: bool = False,
) -> str:
    """Stage files and create a git commit.

    Args:
        message: Commit message.
        repo_dir: Path to the git repository.
        add_all: If True, stage all changes (git add -A) before committing.
    """
    if add_all:
        _git_run(runtime, ["add", "-A"], cwd=repo_dir)
    code, out, err = _git_run(runtime, ["commit", "-m", message], cwd=repo_dir)
    if code != 0:
        return json.dumps({"error": err.strip(), "stdout": out.strip()})
    code2, hash_out, _ = _git_run(runtime, ["rev-parse", "HEAD"], cwd=repo_dir)
    return json.dumps({"repo": repo_dir or str(Path.cwd()), "commit": hash_out.strip(), "message": message, "output": out.strip(), "success": True}, indent=2)


def maos_git_log(
    runtime: Runtime,
    repo_dir: str = "",
    limit: int = 20,
    oneline: bool = True,
) -> str:
    """Get git log for a repository.

    Args:
        repo_dir: Path to the git repository.
        limit: Maximum number of commits to return (default 20).
        oneline: If True, use --oneline format (default True).
    """
    args = ["log", f"-{limit}"]
    if oneline:
        args.append("--oneline")
    code, out, err = _git_run(runtime, args, cwd=repo_dir)
    if code != 0:
        return json.dumps({"error": err.strip()})
    commits = out.strip().split("\n") if out.strip() else []
    return json.dumps({"repo": repo_dir or str(Path.cwd()), "commits": commits, "count": len(commits)}, indent=2)


def maos_git_branch(
    runtime: Runtime,
    repo_dir: str = "",
    action: str = "list",
    branch_name: str = "",
) -> str:
    """Manage git branches.

    Args:
        repo_dir: Path to the git repository.
        action: 'list' (default), 'create', 'switch', 'delete'.
        branch_name: Branch name for create/switch/delete actions.
    """
    if action == "list":
        code, out, err = _git_run(runtime, ["branch", "-a"], cwd=repo_dir)
    elif action == "create":
        code, out, err = _git_run(runtime, ["checkout", "-b", branch_name], cwd=repo_dir)
    elif action == "switch":
        code, out, err = _git_run(runtime, ["checkout", branch_name], cwd=repo_dir)
    elif action == "delete":
        code, out, err = _git_run(runtime, ["branch", "-d", branch_name], cwd=repo_dir)
    else:
        return json.dumps({"error": f"unknown action: {action}"})

    if code != 0:
        return json.dumps({"error": err.strip()})
    return json.dumps({"repo": repo_dir or str(Path.cwd()), "action": action, "branch": branch_name, "output": out.strip(), "success": True}, indent=2)


# --- Subagent tool ----------------------------------------------------------


def maos_run_subagent(
    runtime: Runtime,
    task: str,
    spec: str,
    title: str = "",
    profile: str = "general",
    is_background: bool = False,
) -> str:
    """Spawn a subagent to handle a task autonomously.

    Uses the devin CLI to run a subagent with MAOS tool access. The
    subagent inherits all 48 MAOS MCP tools (task mgmt, memory, git,
    file ops, web, guards, storage, subagent) plus the devin CLI's
    own tools. The MAOS onboarding briefing is injected so the
    subagent knows who the user is, what lessons exist, and what's
    in progress.

    The mandatory ``spec`` flag tells the subagent WHERE to run —
    which AI system/provider to route through. This is not optional;
    every subagent invocation must declare its target system so the
    handoff is traceable and the right context is loaded.

    Supported specs:
        devin         — Devin CLI with GLM-5.2 (default, deepest MCP integration)
        grokhack      — Grok CLI with grokhack personas (security research)
        claude-code   — Claude Code via OpenRouter (model-agnostic, CV/code work)
        local         — MAOS local models (Phi-3.5-mini, private execution)
        openrouter    — Any OpenRouter model (specify in the task prompt)

    Args:
        task: The task or prompt to give the subagent.
        spec: MANDATORY. Which system to route to: devin, grokhack,
              claude-code, local, or openrouter.
        title: Short title for the subagent (optional).
        profile: 'general' (default) or 'explore' (read-only).
        is_background: If True, run in background (default False).
    """
    import os
    import subprocess

    # --- Spec routing -------------------------------------------------------
    # The spec determines which CLI/system to invoke and what context
    # prefix to inject. Each spec has its own system prompt that tells
    # the subagent what tools it has and how to behave.
    SPECS: dict[str, dict[str, str]] = {
        "devin": {
            "cli": os.environ.get("DEVIN_PATH", "devin"),
            "flag": "--print",
            "prefix": (
                "You are a MAOS subagent running via Devin CLI with GLM-5.2. "
                "You have access to MAOS MCP tools (maos_onboard, "
                "maos_search_memory, maos_get_context, maos_list_tasks, "
                "maos_create_task, maos_read_file, maos_write_file, "
                "maos_edit_file, maos_grep, maos_run_command, "
                "maos_git_status, maos_git_diff, maos_git_commit, "
                "maos_web_search, maos_fetch_url, and more). "
                "Call maos_onboard first to get the user briefing, lessons, "
                "and active workspace context. Then execute the task.\n\n"
            ),
        },
        "grokhack": {
            "cli": os.environ.get("GROK_PATH", "grok"),
            "flag": "--print",
            "prefix": (
                "You are a MAOS subagent running via Grok CLI with grokhack. "
                "You have access to Grok personas (hacker, orchestrator, "
                "explore) and MAOS MCP tools. Load the hacker persona for "
                "security research tasks. Call maos_onboard first for user "
                "context and lessons. Respect exploit boundary enforcement "
                "from ~/.grok/memory/MEMORY.md.\n\n"
            ),
        },
        "claude-code": {
            "cli": os.environ.get("CLAUDE_PATH", "claude"),
            "flag": "--print",
            "prefix": (
                "You are a MAOS subagent running via Claude Code through "
                "OpenRouter (model-agnostic). You have access to Claude Code "
                "skills (deep-research, code-review, verify) and MAOS MCP "
                "tools. Call maos_onboard first for user context and lessons. "
                "Use the verify skill before completing non-trivial changes.\n\n"
            ),
        },
        "local": {
            "cli": "uv",
            "flag": "run",
            "prefix": (
                "You are a MAOS subagent running via local MAOS models "
                "(Phi-3.5-mini-4bit). You have access to all MAOS CLI commands "
                "and MCP tools. This is a private execution path — no data "
                "leaves the machine. Call maos_onboard first for context.\n\n"
            ),
            "extra_args": ["app", "do"],
        },
        "openrouter": {
            "cli": os.environ.get("DEVIN_PATH", "devin"),
            "flag": "--print",
            "prefix": (
                "You are a MAOS subagent running via Devin CLI with an "
                "OpenRouter model. Specify the exact model in your task. "
                "You have access to all MAOS MCP tools. Call maos_onboard "
                "first for user context and lessons.\n\n"
            ),
        },
    }

    if spec not in SPECS:
        available = ", ".join(SPECS)
        return json.dumps({
            "error": f"Unknown spec '{spec}'. Available: {available}",
            "task": task,
            "hint": "The spec flag is MANDATORY. Choose: devin, grokhack, claude-code, local, or openrouter.",
        }, indent=2)

    spec_config = SPECS[spec]
    cli_path = spec_config["cli"]
    cli_flag = spec_config["flag"]
    maos_prefix = spec_config["prefix"]
    extra_args = spec_config.get("extra_args", [])

    full_task = maos_prefix + task

    try:
        cmd = [cli_path, cli_flag, *extra_args, full_task]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300 if not is_background else 5)

        if is_background:
            return json.dumps({
                "title": title, "task": task, "spec": spec,
                "profile": profile, "background": True,
                "message": f"Background subagent started via {spec}.",
            }, indent=2)

        return json.dumps({
            "title": title, "task": task, "spec": spec, "profile": profile,
            "exit_code": result.returncode,
            "output": result.stdout[:20000],
            "error": result.stderr[:5000] if result.stderr else "",
            "success": result.returncode == 0,
        }, indent=2)
    except FileNotFoundError:
        return json.dumps({
            "error": f"CLI not found at '{cli_path}' for spec '{spec}'",
            "task": task,
            "hint": f"Make sure the {spec} CLI is installed and in PATH.",
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"subagent timed out (300s) via {spec}", "task": task}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "task": task, "spec": spec}, indent=2)


# --- Todo management tool ---------------------------------------------------


def maos_todo_write(
    runtime: Runtime,
    todos: list[dict[str, str]] | None = None,
) -> str:
    """Create or update a structured todo list for the current session.

    Each todo has a content description and status (pending, in_progress, completed).
    Calling this replaces the entire todo list.

    Args:
        todos: List of todo items, each with 'content' and 'status' keys.
    """
    if todos is None:
        todos = []

    try:
        runtime.db.set_preference("session_todos", json.dumps(todos))
    except Exception:
        pass

    pending = [t for t in todos if t.get("status") == "pending"]
    in_progress = [t for t in todos if t.get("status") == "in_progress"]
    completed = [t for t in todos if t.get("status") == "completed"]

    return json.dumps({
        "total": len(todos),
        "pending": len(pending),
        "in_progress": len(in_progress),
        "completed": len(completed),
        "todos": todos,
    }, indent=2)


# --- Ask user tool ----------------------------------------------------------


def maos_ask_user(
    runtime: Runtime,
    questions: list[dict[str, Any]] | None = None,
) -> str:
    """Present questions to the user and collect their answers.

    For HTTP/non-interactive sessions, returns a prompt that the calling
    client should present to the user.

    Args:
        questions: List of question objects with 'question', 'header', and optional 'options'.
    """
    if not questions:
        return json.dumps({"error": "questions list is required"})

    formatted = []
    for q in questions:
        entry = {"question": q.get("question", ""), "header": q.get("header", "")}
        if q.get("options"):
            entry["options"] = q["options"]
            entry["multi_select"] = q.get("multi_select", False)
        formatted.append(entry)

    return json.dumps({
        "type": "user_questions",
        "questions": formatted,
        "instructions": "Present these questions to the user and collect their answers.",
    }, indent=2)


# --- Slack tools -------------------------------------------------------------
# These let any MCP-connected agent read and send Slack messages, list
# channels, search history, and trigger incremental syncs. Tokens are
# loaded from runtime.settings.slack (env vars), never hardcoded.


def _slack_bot_client(runtime: Runtime) -> Any:
    """Build a Slack WebClient using the bot token from config.

    Raises a clear error if the bot token is not configured.
    """
    cfg = runtime.settings.slack
    if not cfg.bot_token:
        raise RuntimeError("Slack bot token not configured — set SLACK_BOT_TOKEN")
    from slack_sdk import WebClient
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

    client = WebClient(token=cfg.bot_token)
    client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=3))
    return client


def _slack_user_client(runtime: Runtime) -> Any:
    """Build a Slack WebClient using the user token from config.

    The user token is needed for search.messages and history sync.
    Raises a clear error if the user token is not configured.
    """
    cfg = runtime.settings.slack
    if not cfg.user_token:
        raise RuntimeError("Slack user token not configured — set SLACK_USER_TOKEN")
    from slack_sdk import WebClient
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

    client = WebClient(token=cfg.user_token)
    client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=3))
    return client


def _slack_team_id(runtime: Runtime) -> str | None:
    """Get the configured Slack team ID (or None for token default)."""
    return runtime.settings.slack.team_id or None


def maos_slack_send(
    runtime: Runtime,
    channel: str,
    message: str,
    thread_ts: str = "",
) -> str:
    """Send a message to a Slack channel or DM.

    Uses the bot token (chat.postMessage). The channel can be a channel
    ID, channel name (e.g. #general), or user ID for a DM.

    Args:
        channel: Channel ID, channel name, or user ID for DM.
        message: The message text to send (Slack mrkdwn supported).
        thread_ts: Optional timestamp of the parent message to reply in a thread.
    """
    if not channel or not message:
        return json.dumps({"error": "channel and message are required"})

    try:
        client = _slack_bot_client(runtime)
        kwargs: dict[str, Any] = {"channel": channel, "text": message}
        team_id = _slack_team_id(runtime)
        if team_id:
            kwargs["team_id"] = team_id
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        response = client.chat_postMessage(**kwargs)
        return json.dumps(
            {
                "success": bool(response.get("ok")),
                "channel": response.get("channel", channel),
                "ts": response.get("ts", ""),
                "message": response.get("message", {}),
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_slack_read(
    runtime: Runtime,
    channel: str,
    limit: int = 50,
    oldest: str = "",
) -> str:
    """Read recent messages from a Slack channel.

    Uses the bot token (conversations.history). Returns messages as JSON
    with text, user, ts, and thread info.

    Args:
        channel: Channel ID or name to read from.
        limit: Maximum messages to return (default 50, max 200).
        oldest: Optional Unix timestamp — only return messages after this.
    """
    if not channel:
        return json.dumps({"error": "channel is required"})

    limit = max(1, min(limit, 200))
    try:
        client = _slack_bot_client(runtime)
        kwargs: dict[str, Any] = {"channel": channel, "limit": limit}
        team_id = _slack_team_id(runtime)
        if team_id:
            kwargs["team_id"] = team_id
        if oldest:
            kwargs["oldest"] = oldest
        response = client.conversations_history(**kwargs)

        messages = []
        for msg in response.get("messages") or []:
            messages.append(
                {
                    "ts": msg.get("ts", ""),
                    "user": msg.get("user", ""),
                    "text": msg.get("text", ""),
                    "thread_ts": msg.get("thread_ts"),
                    "subtype": msg.get("subtype"),
                    "reply_count": msg.get("reply_count", 0),
                    "reactions": msg.get("reactions", []),
                }
            )

        return json.dumps(
            {
                "success": bool(response.get("ok")),
                "channel": channel,
                "count": len(messages),
                "has_more": bool(response.get("has_more", False)),
                "messages": messages,
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_slack_channels(
    runtime: Runtime,
    types: str = "public_channel,private_channel",
) -> str:
    """List all visible Slack channels.

    Uses the bot token (conversations.list). Returns channel names, IDs,
    and type (public, private, im, mpim).

    Args:
        types: Comma-separated channel types to list (default:
            public_channel,private_channel).
    """
    try:
        client = _slack_bot_client(runtime)
        channels: list[dict[str, Any]] = []
        cursor = ""
        while True:
            kwargs: dict[str, Any] = {
                "types": types,
                "exclude_archived": True,
                "limit": 200,
            }
            team_id = _slack_team_id(runtime)
            if team_id:
                kwargs["team_id"] = team_id
            if cursor:
                kwargs["cursor"] = cursor
            response = client.conversations_list(**kwargs)
            for ch in response.get("channels") or []:
                channel_type = "public_channel"
                if ch.get("is_im"):
                    channel_type = "im"
                elif ch.get("is_mpim"):
                    channel_type = "mpim"
                elif ch.get("is_private"):
                    channel_type = "private_channel"
                channels.append(
                    {
                        "id": ch.get("id", ""),
                        "name": ch.get("name", ""),
                        "type": channel_type,
                        "is_member": bool(ch.get("is_member", False)),
                        "num_members": ch.get("num_members"),
                        "topic": (ch.get("topic") or {}).get("value", ""),
                        "purpose": (ch.get("purpose") or {}).get("value", ""),
                    }
                )
            metadata = response.get("response_metadata") or {}
            cursor = str(metadata.get("next_cursor") or "").strip()
            if not cursor:
                break

        return json.dumps(
            {
                "success": True,
                "count": len(channels),
                "channels": channels,
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def maos_slack_search(
    runtime: Runtime,
    query: str,
    count: int = 20,
) -> str:
    """Search Slack messages across the workspace.

    Uses the user token (search.messages) — requires SLACK_USER_TOKEN.
    Returns matching messages with channel, author, timestamp, and permalink.

    Args:
        query: Search query (Slack search syntax supported).
        count: Maximum results to return (default 20, max 100).
    """
    if not query:
        return json.dumps({"error": "query is required"})

    count = max(1, min(count, 100))
    try:
        client = _slack_user_client(runtime)
        response = client.search_messages(query=query, count=count)

        matches = []
        for msg in (response.get("messages") or {}).get("matches") or []:
            matches.append(
                {
                    "ts": msg.get("ts", ""),
                    "text": msg.get("text", ""),
                    "user": msg.get("user", ""),
                    "username": msg.get("username", ""),
                    "channel": (msg.get("channel") or {}).get("name", ""),
                    "channel_id": (msg.get("channel") or {}).get("id", ""),
                    "permalink": msg.get("permalink", ""),
                    "team": (msg.get("team") or {}).get("name", ""),
                }
            )

        return json.dumps(
            {
                "success": bool(response.get("ok")),
                "query": query,
                "count": len(matches),
                "total": (response.get("messages") or {}).get("total", 0),
                "messages": matches,
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# --- Project sync tools -----------------------------------------------------
# These call the project-sync CLI to manage 2-way project sync between
# local ~/Projects/ and GDrive with blockchain evidence. Cloud agents use
# these to push, pull, register forks, and resolve.


_PROJECT_SYNC_BIN = os.environ.get("MAOS_PROJECT_SYNC_BIN", "project-sync")


def maos_project_sync_status(runtime: Runtime) -> str:
    """Show sync state for all projects.

    Calls the project-sync CLI to display which projects are in sync,
    which have local changes to push, and which have pending forks from
    cloud AI agents.
    """
    try:
        result = subprocess.run(
            [_PROJECT_SYNC_BIN, "status"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def maos_project_sync_push(
    runtime: Runtime, project: str, agent: str = "cloud",
) -> str:
    """Push a project to GDrive and record blockchain evidence.

    Uploads the current local state of the project to Google Drive and
    creates a hash-chained evidence entry.

    Args:
        project: Project name in ~/Projects/.
        agent: Agent name performing the push (default 'cloud').
    """
    if not project:
        return json.dumps({"error": "project is required"})
    try:
        result = subprocess.run(
            [_PROJECT_SYNC_BIN, "push", project, "--agent", agent],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def maos_project_sync_pull(
    runtime: Runtime, project: str, agent: str = "cloud",
) -> str:
    """Pull a project from GDrive.

    Downloads the latest cloud state of the project from Google Drive and
    updates the local copy.

    Args:
        project: Project name in ~/Projects/.
        agent: Agent name performing the pull (default 'cloud').
    """
    if not project:
        return json.dumps({"error": "project is required"})
    try:
        result = subprocess.run(
            [_PROJECT_SYNC_BIN, "pull", project, "--agent", agent],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def maos_project_sync_forks(runtime: Runtime) -> str:
    """List pending forks — cloud AI changes not yet pulled locally.

    A fork is registered when a cloud agent modifies a project. Use this
    to check if any cloud agents have made changes that need to be
    resolved.
    """
    try:
        result = subprocess.run(
            [_PROJECT_SYNC_BIN, "forks"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def maos_project_sync_fork(
    runtime: Runtime, project: str, agent: str, hash: str,
) -> str:
    """Register a cloud AI fork.

    Call this when a cloud agent has modified a project, recording the
    project hash after the changes. This creates a pending fork that
    must later be resolved by pulling from GDrive.

    Args:
        project: Project name.
        agent: Agent name (e.g. 'devin', 'grokhack').
        hash: Project hash after cloud AI changes.
    """
    if not project or not agent or not hash:
        return json.dumps({"error": "project, agent, and hash are required"})
    try:
        result = subprocess.run(
            [_PROJECT_SYNC_BIN, "fork", project, agent, hash],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def maos_project_sync_resolve(runtime: Runtime, project: str) -> str:
    """Resolve a fork by pulling from GDrive.

    After a cloud AI fork has been registered, this pulls the latest
    state from Google Drive to reconcile the local copy and clears the
    pending fork.

    Args:
        project: Project name to resolve.
    """
    if not project:
        return json.dumps({"error": "project is required"})
    try:
        result = subprocess.run(
            [_PROJECT_SYNC_BIN, "resolve", project],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def maos_slack_sync(runtime: Runtime, days: int = 30) -> str:
    """Trigger an incremental Slack history sync into MAOS.

    Pulls recent messages from visible channels into the local SQLite
    store and refreshes the RAG index so messages become searchable via
    maos_search_memory. Uses the user token.

    Args:
        days: How many days of history to sync (default 30).
    """
    cfg = runtime.settings.slack
    if not cfg.user_token:
        return json.dumps(
            {"success": False, "error": "Slack user token not configured — set SLACK_USER_TOKEN"},
            indent=2,
        )

    try:
        from pathlib import Path

        from app.slack.ingest import SlackIngestor, SlackStore

        store_path = Path(cfg.ingest_db).expanduser()
        if not store_path.is_absolute():
            store_path = Path(__file__).resolve().parents[2] / store_path

        ingestor = SlackIngestor(
            cfg.user_token,
            SlackStore(store_path),
            team_id=cfg.team_id,
            days=max(1, days),
            max_messages=cfg.ingest_max_messages,
            include_dms=cfg.ingest_include_dms,
            channels=tuple(cfg.ingest_channels),
            retriever=runtime.retriever,
        )
        result = ingestor.sync(full=False)
        return json.dumps(
            {
                "success": not result.errors,
                **result.as_dict(),
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)
