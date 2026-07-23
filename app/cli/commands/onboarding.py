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
    """List all recorded lessons."""
    rt = build_runtime()
    lessons = list_lessons(rt.db)
    if not lessons:
        typer.echo("No lessons recorded yet.")
        return
    lines: list[str] = []
    for lesson in lessons:
        lines.append(f"  [{lesson['confidence']:.0%}] {lesson['slug']}: {lesson['text']}")
        lines.append(f"        source={lesson['source']}  updated={lesson['updated_at']}")
    print_panel("\n".join(lines), title=f"Lessons ({len(lessons)})")


@onboarding_app.command("record")
def record_cmd(
    slug: str = typer.Argument(..., help="Short snake_case identifier"),
    text: str = typer.Argument(..., help="The lesson text"),
    confidence: float = typer.Option(0.9, "--confidence", "-c", help="Confidence 0-1"),
    source: str = typer.Option("incident", "--source", "-s", help="incident|observed|manual"),
) -> None:
    """Record a new lesson."""
    rt = build_runtime()
    record_lesson(rt.db, slug, text, confidence=confidence, source=source)
    typer.echo(f"Recorded lesson: {slug}")
    typer.echo(f"  {text}")
    typer.echo(f"  confidence={confidence}  source={source}")


@onboarding_app.command("email")
def email_cmd(
    subject: str = typer.Argument(..., help="Email subject"),
    body: str = typer.Argument(..., help="Email body (plain text)"),
    to: str = typer.Option("", "--to", help="Recipient (default: david)"),
) -> None:
    """Send an email to david via Gmail."""
    try:
        result = _send_email(subject, body, to=to or "david.nichols.ops@gmail.com")
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
