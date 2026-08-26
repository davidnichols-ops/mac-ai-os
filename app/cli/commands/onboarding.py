"""`maos onboard` — session onboarding and lesson management.

Commands:
  maos onboard              — print the onboarding briefing
  maos onboard --json       — structured JSON output
  maos onboard lessons      — list recorded lessons
  maos onboard record       — record a new lesson
  maos onboard email        — send an email to david
  maos onboard escalate     — send an escalation email
"""

from __future__ import annotations

import json

import typer

from app.gmail import EmailError
from app.gmail import escalate as _escalate
from app.gmail import send_email as _send_email
from app.onboarding import (
    build_onboarding_packet,
    format_onboarding_briefing,
    list_lessons,
    record_lesson,
)
from app.runtime import build_runtime
from app.ui.console import print_panel

onboarding_app = typer.Typer(
    name="onboard",
    help="Session onboarding and lesson management.",
    no_args_is_help=False,
)


@onboarding_app.callback(invoke_without_command=True)
def onboard(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Print the onboarding briefing for a new session."""
    if ctx.invoked_subcommand is not None:
        return
    rt = build_runtime()
    packet = build_onboarding_packet(rt.db)
    if json_output:
        typer.echo(json.dumps(packet, default=str, indent=2))
    else:
        typer.echo(format_onboarding_briefing(packet))


@onboarding_app.command("lessons")
def lessons_cmd() -> None:
    """List all recorded lessons, grouped by category."""
    rt = build_runtime()
    lessons = list_lessons(rt.db)
    if not lessons:
        typer.echo("No lessons recorded yet.")
        return
    # Group by category.
    categories: dict[str, list[dict]] = {}
    for lesson in lessons:
        cat = lesson.get("category", "") or "general"
        categories.setdefault(cat, []).append(lesson)
    cat_order = ["tooling", "workflow", "environment", "api", "architecture", "general"]
    cat_labels = {
        "tooling": "Tooling & Infrastructure",
        "workflow": "Workflow & Process",
        "environment": "Machine & Environment",
        "api": "External Services & APIs",
        "architecture": "Architecture & Design",
        "general": "General",
    }
    lines: list[str] = []
    for cat in cat_order:
        cat_lessons = categories.get(cat)
        if not cat_lessons:
            continue
        label = cat_labels.get(cat, cat.title())
        lines.append(f"\n  --- {label} ({len(cat_lessons)}) ---")
        for lesson in cat_lessons:
            kw = lesson.get("keywords", [])
            kw_tag = f"  triggers: {', '.join(kw)}" if kw else ""
            lines.append(f"  [{lesson['confidence']:.0%}] {lesson['slug']}: {lesson['text'][:120]}")
            lines.append(f"        source={lesson['source']}  updated={lesson['updated_at']}{kw_tag}")
            if lesson.get("reminder"):
                lines.append(f"        reminder: {lesson['reminder']}")
    print_panel("\n".join(lines), title=f"Lessons ({len(lessons)})")


@onboarding_app.command("record")
def record_cmd(
    slug: str = typer.Argument(..., help="Short snake_case identifier"),
    text: str = typer.Argument(..., help="The lesson text"),
    confidence: float = typer.Option(0.9, "--confidence", "-c", help="Confidence 0-1"),
    source: str = typer.Option("incident", "--source", "-s", help="incident|observed|manual"),
    category: str = typer.Option("", "--category", help="tooling|workflow|environment|api|architecture"),
    keywords: str = typer.Option("", "--keywords", "-k", help="Comma-separated trigger words"),
    reminder: str = typer.Option("", "--reminder", help="One-line reminder shown on keyword match"),
) -> None:
    """Record a new lesson."""
    rt = build_runtime()
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None
    record_lesson(
        rt.db, slug, text,
        confidence=confidence, source=source,
        category=category, keywords=kw_list, reminder=reminder,
    )
    typer.echo(f"Recorded lesson: {slug}")
    typer.echo(f"  {text}")
    typer.echo(f"  confidence={confidence}  source={source}  category={category or 'none'}")
    if kw_list:
        typer.echo(f"  keywords: {', '.join(kw_list)}")
    if reminder:
        typer.echo(f"  reminder: {reminder}")


@onboarding_app.command("email")
def email_cmd(
    subject: str = typer.Argument(..., help="Email subject"),
    body: str = typer.Argument(..., help="Email body (plain text)"),
    to: str = typer.Option("", "--to", help="Recipient (default: MAOS_EMAIL_RECIPIENT)"),
) -> None:
    """Send an email via Gmail."""
    try:
        result = _send_email(subject, body, to=to)
        typer.echo(f"Sent: {result['subject']} (id={result['message_id']})")
    except EmailError as e:
        typer.echo(f"Failed: {e}", err=True)
        raise typer.Exit(1) from e


@onboarding_app.command("escalate")
def escalate_cmd(
    subject: str = typer.Argument(..., help="Short summary of what needs attention"),
    body: str = typer.Argument(..., help="The detailed explanation"),
    lesson_slug: str = typer.Option("", "--lesson", "-l", help="Related lesson slug"),
) -> None:
    """Send an escalation email — 'this needs human attention'."""
    try:
        result = _escalate(subject, body, lesson_slug=lesson_slug or None)
        typer.echo(f"Escalated: {result['subject']} (id={result['message_id']})")
    except EmailError as e:
        typer.echo(f"Failed: {e}", err=True)
        raise typer.Exit(1) from e


@onboarding_app.command("session-hook")
def session_hook_cmd() -> None:
    """SessionStart hook for Devin — injects onboarding context.

    Outputs hookSpecificOutput JSON on stdout for Devin to inject as
    additionalContext. Designed to be used as a SessionStart hook in
    ~/.config/devin/config.json.
    """
    import json as _json

    try:
        rt = build_runtime()
        packet = build_onboarding_packet(rt.db)
    except Exception:
        # Don't block session start if onboarding fails
        typer.echo(_json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "MAOS onboarding unavailable. Run 'uv run app onboard' manually.",
            }
        }))
        return

    # Build a concise context string from the packet
    lines: list[str] = []
    lines.append("=== MAOS ONBOARDING BRIEFING (MANDATORY READ) ===")
    lines.append("Read this BEFORE any work. Lessons below are hard-won — do not repeat them.")

    # Lessons (top 10 by confidence)
    lessons = packet.get("lessons", [])
    if lessons:
        lines.append(f"\nLESSONS ({len(lessons)} total, showing top 10):")
        for lesson in lessons[:10]:
            lines.append(f"  [{lesson['confidence']:.0%}] {lesson['slug']}: {lesson['text'][:150]}")
        if len(lessons) > 10:
            lines.append(f"  ... and {len(lessons) - 10} more. Run 'uv run app onboard' for full briefing.")

    # Open tasks
    tasks = packet.get("open_tasks", [])
    if tasks:
        lines.append(f"\nOPEN TASKS ({len(tasks)}):")
        for t in tasks[:5]:
            lines.append(f"  [{t['priority']}] {t['title']}")

    # Active workspace
    ws = packet.get("workspace", {})
    if ws.get("name"):
        lines.append(f"\nACTIVE WORKSPACE: {ws['name']} ({ws.get('path', '')})")

    # PR governance reminder
    lines.append("\nPR GOVERNANCE: Run evidence-producing guards before declaring done.")
    lines.append("  uv run app guards verify 'uv run pytest' --save test.json")
    lines.append("  uv run app guards verify 'uv run ruff check app tests' --save lint.json")

    # Onboarding enforcement
    lines.append("\nONBOARDING: You MUST run 'uv run app onboard' at session start.")
    lines.append("If you learned something this session, record it BEFORE stopping:")
    lines.append("  uv run app onboard record <slug> '<lesson text>'")

    typer.echo(_json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n".join(lines),
        }
    }))


@onboarding_app.command("check-lesson-recorded")
def check_lesson_recorded_cmd(
    hours: int = typer.Option(4, "--hours", "-h", help="Lookback window in hours"),
) -> None:
    """Stop hook for Devin — blocks exit if work was done but no lesson recorded.

    Checks if git commits were made in the last N hours across active
    workspaces. If commits exist but no lesson was recorded in the same
    window, exits with code 2 (block) and a message telling the agent
    to record a lesson.

    Exit codes:
      0 — OK (no work done, or lesson was recorded)
      2 — BLOCK (work was done but no lesson recorded)
    """
    import subprocess
    from datetime import UTC, datetime, timedelta

    from app.onboarding import list_lessons

    rt = build_runtime()
    now = datetime.now(UTC)
    since = (now - timedelta(hours=hours)).isoformat()

    # Check for lessons recorded in the window
    lessons = list_lessons(rt.db)
    recent_lessons = [
        lesson for lesson in lessons
        if lesson.get("updated_at", "") >= since
    ]

    if recent_lessons:
        # Lesson was recorded — OK
        typer.echo(f"OK: {len(recent_lessons)} lesson(s) recorded in last {hours}h.")
        return

    # Check for git commits in active workspaces
    # Use direct git log (not the MAOS index, which may be stale)
    workspaces = rt.db.query(
        "SELECT path FROM workspaces WHERE path IS NOT NULL ORDER BY last_activity DESC LIMIT 10"
    )
    ws_paths = [r["path"] for r in workspaces] if workspaces else []

    # Also check common project dirs
    import os
    from pathlib import Path
    for proj in ["~/Projects", "~/mac-ai-os"]:
        root = Path(proj).expanduser()
        if root.exists():
            for entry in os.listdir(root):
                p = root / entry
                if (p / ".git").exists() and str(p) not in ws_paths:
                    ws_paths.append(str(p))

    commit_count = 0
    commit_repos: list[str] = []

    for ws_path in ws_paths:
        if not Path(ws_path).exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", ws_path, "log", f"--since={hours} hours ago",
                 "--format=%H"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                count = len([line for line in result.stdout.strip().splitlines() if line.strip()])
                if count > 0:
                    commit_count += count
                    commit_repos.append(f"{Path(ws_path).name} ({count})")
        except (subprocess.TimeoutExpired, OSError):
            continue

    if commit_count == 0:
        # No work done — OK to stop
        typer.echo(f"OK: No commits in last {hours}h. No lesson required.")
        return

    # Work was done but no lesson recorded — BLOCK
    repo_summary = ", ".join(commit_repos[:5])
    typer.echo(
        f"BLOCK: {commit_count} commit(s) in last {hours}h ({repo_summary}) "
        f"but NO lesson recorded. If this session learned something, "
        f"record it before stopping:\n"
        f"  uv run app onboard record <slug> '<lesson text>' "
        f"--source incident --category workflow\n"
        f"If this session didn't learn anything new, re-run with "
        f"--hours 0 to override.",
        err=True,
    )
    raise typer.Exit(2)
