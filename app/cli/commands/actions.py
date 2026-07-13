"""`maos actions` — action layer commands."""

from __future__ import annotations

import json

import typer

from app.actions.models import ActionRequest
from app.runtime import build_runtime
from app.ui.console import print_panel

actions_app = typer.Typer(name="actions", help="Action layer commands.", no_args_is_help=True)


@actions_app.command("list")
def list_actions() -> None:
    """List all registered actions."""
    rt = build_runtime()
    actions = rt.action_registry.list_actions()
    if not actions:
        typer.echo("No actions registered.")
        return

    lines = [f"{'Name':>20s}  {'Reversible':>10s}  Permissions  Description"]
    for a in actions:
        rev = "yes" if a.reversible else "no"
        perms = ", ".join(a.permissions)
        lines.append(f"{a.name:>20s}  {rev:>10s}  {perms:<20s}  {a.description}")
    print_panel("\n".join(lines), title=f"Actions ({len(actions)})")


@actions_app.command("run")
def run(
    action: str = typer.Argument(..., help="Action name"),
    target: str = typer.Argument("", help="Action target"),
    params: str = typer.Option("{}", "--params", "-p", help="JSON parameters"),
) -> None:
    """Execute an action directly."""
    rt = build_runtime()
    try:
        parameters = json.loads(params)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON params: {e}", err=True)
        raise typer.Exit(1) from e

    request = ActionRequest(action=action, target=target, parameters=parameters)
    result = rt.action_executor.execute(request, context={"db": rt.db})

    if result.ok:
        print_panel(result.result, title=f"Action: {action}")
    else:
        typer.echo(f"Action failed: {result.error}", err=True)
        raise typer.Exit(1)


@actions_app.command("log")
def log(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries"),
) -> None:
    """Show the action log."""
    rt = build_runtime()
    entries = rt.db.recent_actions(limit=limit)
    if not entries:
        typer.echo("No actions logged.")
        return

    lines = [f"{'Time':>19s}  {'Action':>20s}  {'Target':>15s}  {'OK':>2s}  Result"]
    for e in entries:
        ok = "Y" if e["executed"] and not e["error"] else "N"
        result = (e["result"] or e["error"] or "")[:40]
        lines.append(
            f"{e['timestamp'][:19]:>19s}  {e['action_name']:>20s}  "
            f"{(e['target'] or '')[:15]:>15s}  {ok:>2s}  {result}"
        )
    print_panel("\n".join(lines), title=f"Action Log ({len(entries)})")


@actions_app.command("decisions")
def decisions(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of decisions"),
) -> None:
    """Show the decision log."""
    rt = build_runtime()
    entries = rt.db.recent_decisions(limit=limit)
    if not entries:
        typer.echo("No decisions logged.")
        return

    lines = [f"{'Time':>19s}  {'Agent':>15s}  {'Decision':>25s}  {'Conf':>5s}  Reasoning"]
    for e in entries:
        lines.append(
            f"{e['timestamp'][:19]:>19s}  {e['agent']:>15s}  {e['decision']:>25s}  "
            f"{e['confidence']:5.2f}  {e['reasoning_summary'][:50]}"
        )
    print_panel("\n".join(lines), title=f"Decision Log ({len(entries)})")
