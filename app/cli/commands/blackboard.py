"""`maos blackboard` — shared working memory commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

blackboard_app = typer.Typer(name="blackboard", help="Blackboard (working memory) commands.", no_args_is_help=True)


@blackboard_app.command("show")
def show() -> None:
    """Show the current blackboard state."""
    rt = build_runtime()
    text = rt.blackboard.to_text()
    print_panel(text, title="Blackboard")


@blackboard_app.command("clear")
def clear() -> None:
    """Clear the blackboard."""
    rt = build_runtime()
    rt.blackboard.clear()
    rt.blackboard.persist(rt.db)
    typer.echo("Blackboard cleared.")


@blackboard_app.command("goal")
def set_goal(
    goal: str = typer.Argument(..., help="The current goal"),
) -> None:
    """Set the current goal on the blackboard."""
    rt = build_runtime()
    rt.blackboard.set_goal(goal)
    rt.blackboard.persist(rt.db)
    typer.echo(f"Goal set: {goal}")


@blackboard_app.command("cycle")
def attention_cycle() -> None:
    """Run one attention cycle to update the blackboard."""
    rt = build_runtime()
    context = rt.attention.cycle()
    typer.echo("Attention cycle complete.")
    typer.echo(f"  Workspace: {context.get('workspace', 'none')}")
    typer.echo(f"  Goal: {context.get('goal', 'none')}")
    typer.echo(f"  Health: {context.get('health', {}).get('status', 'unknown')}")
    typer.echo(f"  Problems: {context.get('open_problems', 0)}")
    typer.echo(f"  Suggestions: {context.get('suggestions', 0)}")
    typer.echo(f"  Confidence: {rt.blackboard.confidence:.2f}")


@blackboard_app.command("attention")
def attention_status() -> None:
    """Show attention daemon status."""
    rt = build_runtime()
    status = "running" if rt.attention.is_running else "stopped"
    typer.echo(f"Attention daemon: {status}")
    typer.echo(f"  Interval: {rt.attention.interval}s")
    typer.echo(f"  Last cycle: {rt.attention.last_cycle or 'never'}")
