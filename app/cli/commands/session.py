"""`maos session` — session management and reconstruction commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel

session_app = typer.Typer(name="session", help="Session reconstruction commands.", no_args_is_help=True)


@session_app.command("detect")
def detect() -> None:
    """Auto-detect sessions from event gaps."""
    rt = build_runtime()
    from app.sessions import detect_sessions

    sessions = detect_sessions(rt.db)
    if not sessions:
        print_panel("No sessions detected. Run some activity first (index, telemetry, git).",
                    title="Session Detection")
        return

    lines = [f"Detected {len(sessions)} session(s):"]
    for s in sessions:
        lines.append(
            f"  #{s['id']}  {s['start_time'][:19]} → {s['end_time'][:19]}  "
            f"({s['event_count']} events, project={s['project'] or '?'})"
        )
    print_panel("\n".join(lines), title="Session Detection")


@session_app.command("reconstruct")
def reconstruct(
    gap_minutes: int = typer.Option(30, "--gap", help="Gap threshold in minutes"),
    summarize_top: int = typer.Option(0, "--summarize", "-s",
        help="Summarize top N sessions with LLM (0 = skip)"),
) -> None:
    """Reconstruct work sessions from git commit history.

    Uses git commit timestamps as activity signals — works even when
    shell history lacks timestamps. Also correlates shell commands to
    sessions by content matching.
    """
    from datetime import timedelta
    rt = build_runtime()
    from app.sessions import detect_sessions_from_git

    sessions = detect_sessions_from_git(
        rt.db, gap_threshold=timedelta(minutes=gap_minutes),
    )

    if not sessions:
        print_panel(
            "No sessions detected from git history.\n"
            "Make sure git commits are indexed (maos git index <repo>).",
            title="Session Reconstruction",
        )
        return

    lines = [f"Reconstructed {len(sessions)} session(s) from git history:"]
    for s in sessions:
        repos = ", ".join(s.get("repos", [])[:3])
        lines.append(
            f"  #{s['id']}  {s['start_time'][:19]} → {s['end_time'][:19]}  "
            f"({s['commit_count']} commits, {s['project'] or '?'})"
        )
        if repos:
            lines.append(f"    repos: {repos}")
    print_panel("\n".join(lines), title="Session Reconstruction")

    # Optionally summarize top sessions.
    if summarize_top > 0:
        from app.agents.session import SessionAgent

        agent = SessionAgent(planner=rt.planner, db=rt.db, retriever=rt.retriever)
        top = sorted(sessions, key=lambda s: s["commit_count"], reverse=True)[:summarize_top]
        for s in top:
            typer.echo(f"\n--- Session #{s['id']} ({s['project']}) ---")
            result = agent.execute(f"Summarize session {s['id']}")
            if result.ok:
                print_markdown(result.answer)
            else:
                typer.echo(f"  (summary failed: {result.error})")


@session_app.command("list")
def list_sessions(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of sessions to show"),
) -> None:
    """List recent sessions."""
    rt = build_runtime()
    sessions = rt.db.recent_sessions(limit=limit)
    if not sessions:
        typer.echo("No sessions recorded. Run `maos session detect` first.")
        return

    lines = [f"{'ID':>4s}  {'Start':>19s}  {'End':>19s}  {'Events':>6s}  Project"]
    for s in sessions:
        lines.append(
            f"{s['id']:4d}  {s['start_time'][:19]:>19s}  {s['end_time'][:19]:>19s}  "
            f"{s['event_count']:6d}  {s['project'] or '?'}"
        )
    print_panel("\n".join(lines), title=f"Sessions ({len(sessions)})")


@session_app.command("summarize")
def summarize(
    session_id: int = typer.Argument(None, help="Session ID (defaults to latest)"),
) -> None:
    """Summarize a session using the LLM."""
    rt = build_runtime()
    from app.sessions import summarize_session

    if session_id is None:
        latest = rt.db.latest_session()
        if not latest:
            typer.echo("No sessions found. Run `maos session detect` first.")
            raise typer.Exit(1)
        session_id = latest["id"]

    data = summarize_session(rt.db, session_id)
    if not data:
        typer.echo(f"Session {session_id} not found.")
        raise typer.Exit(1)

    from app.agents.session import SessionAgent

    agent = SessionAgent(planner=rt.planner, db=rt.db, retriever=rt.retriever)
    result = agent.execute(f"Summarize session {session_id}")
    if result.ok:
        print_markdown(result.answer)
    else:
        typer.echo(result.error or "failed", err=True)
        raise typer.Exit(1)


@session_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question about what you were working on"),
) -> None:
    """Ask about your recent work activity."""
    rt = build_runtime()
    from app.agents.session import SessionAgent

    agent = SessionAgent(planner=rt.planner, db=rt.db, retriever=rt.retriever)
    result = agent.execute(question)
    if result.ok:
        print_markdown(result.answer)
    else:
        typer.echo(result.error or "failed", err=True)
        raise typer.Exit(1)
