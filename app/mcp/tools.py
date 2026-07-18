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
import time
from typing import Any

from app.actions.models import ActionRequest
from app.diagnostics import format_diagnostic_report, run_diagnostics
from app.events.models import MCP_TOOL_CALLED, Event
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
    """
    start = time.time()
    success = True
    error = ""
    try:
        result = fn(runtime, *args, **kwargs)
        # Append ambient time footer to every tool response.
        # This is the key integration: the agent can't avoid seeing
        # how long it's been working because it's in every response.
        result = _append_time_footer(runtime, name, result)
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
    """
    if not slug or not text:
        return json.dumps({"error": "slug and text are required"})
    lesson = record_lesson(
        runtime.db, slug, text, confidence=confidence, source=source,
    )
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


# --- Email tools ------------------------------------------------------------


def maos_send_email(
    runtime: Runtime, subject: str, body: str, to: str = "",
) -> str:
    """Send an email via Gmail (david.nichols.ops@gmail.com).

    Uses OAuth credentials from ~/.gmail-mcp/. Tokens are refreshed
    automatically. Default recipient is david himself.

    Args:
        subject: Email subject line.
        body: Plain text body.
        to: Recipient email (defaults to david.nichols.ops@gmail.com).
    """
    try:
        result = _send_email(subject, body, to=to or "david.nichols.ops@gmail.com")
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

    The email is sent to david.nichols.ops@gmail.com with a [MAOS] prefix.
    Optional structured context is appended as a section.

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
