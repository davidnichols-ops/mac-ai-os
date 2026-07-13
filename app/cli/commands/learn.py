"""`maos learn` — learning and procedural memory commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

learn_app = typer.Typer(name="learn", help="Learning / procedural memory commands.", no_args_is_help=True)


@learn_app.command("summary")
def summary() -> None:
    """Show what the system has learned."""
    rt = build_runtime()
    s = rt.learner.learned_summary()

    lines: list[str] = []

    if s["actions"]:
        lines.append("Action success rates:")
        for a in s["actions"]:
            lines.append(f"  {a['action']:>20s}  {a['successes']}/{a['total']}  "
                         f"({a['rate']:.0%})  conf={a['confidence']:.2f}")
        lines.append("")

    if s["workflows"]:
        lines.append("Workflow success rates:")
        for w in s["workflows"]:
            lines.append(f"  {w['workflow']:>25s}  {w['successes']}/{w['total']}  "
                         f"conf={w['confidence']:.2f}")
        lines.append("")

    if s["preferences"]:
        lines.append("Learned preferences:")
        for p in s["preferences"]:
            lines.append(f"  {p['key']:>20s} = {p['value'][:40]}  "
                         f"(conf={p['confidence']:.2f}, {p['source']})")
        lines.append("")

    lines.append(f"Total user_model entries: {s['total_entries']}")
    print_panel("\n".join(lines), title="Learned")


@learn_app.command("action")
def record_action(
    action: str = typer.Argument(..., help="Action name"),
    success: bool = typer.Option(True, "--success/--fail", help="Whether it succeeded"),
    error: str = typer.Option("", "--error", "-e", help="Error message if failed"),
) -> None:
    """Manually record an action outcome for learning."""
    rt = build_runtime()
    rt.learner.record_action_outcome(action, target="", success=success, error=error)
    stats = rt.learner.action_success_rate(action)
    if stats:
        typer.echo(f"Recorded: {action} success={success}")
        typer.echo(f"  Rate: {stats['successes']}/{stats['total']} ({stats['rate']:.0%})")
