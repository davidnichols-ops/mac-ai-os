"""`maos workflow` — workflow execution commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel

workflow_app = typer.Typer(name="workflow", help="Workflow commands.", no_args_is_help=True)


@workflow_app.command("morning")
def morning() -> None:
    """Run the morning routine: reconstruct session, check state, prepare workspace."""
    rt = build_runtime()
    from app.workflows.morning import format_morning_report, morning_workflow

    wf = morning_workflow()
    result = rt.workflow_engine.run(wf, context={"action_executor": rt.action_executor})

    report = format_morning_report(result)
    print_markdown(report)


@workflow_app.command("list")
def list_runs(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of runs to show"),
) -> None:
    """List recent workflow runs."""
    rt = build_runtime()
    runs = rt.db.recent_workflow_runs(limit=limit)
    if not runs:
        typer.echo("No workflow runs recorded.")
        return

    lines = [f"{'ID':>4s}  {'Time':>19s}  {'Workflow':>30s}  {'Status':>10s}  Steps"]
    for r in runs:
        lines.append(
            f"{r['id']:4d}  {r['timestamp'][:19]:>19s}  {r['workflow']:>30s}  "
            f"{r['status']:>10s}  {r['steps_done']}/{r['steps_total']}"
        )
    print_panel("\n".join(lines), title=f"Workflow Runs ({len(runs)})")


@workflow_app.command("show")
def show(
    run_id: int = typer.Argument(..., help="Workflow run ID"),
) -> None:
    """Show details of a specific workflow run."""
    rt = build_runtime()
    runs = rt.db.recent_workflow_runs(limit=100)
    run = next((r for r in runs if r["id"] == run_id), None)
    if not run:
        typer.echo(f"Run {run_id} not found.")
        raise typer.Exit(1)

    lines = [
        f"Workflow: {run['workflow']}",
        f"Status:   {run['status']}",
        f"Steps:    {run['steps_done']}/{run['steps_total']}",
        f"Duration: {run['duration_ms'] or 0}ms",
    ]
    if run["result"]:
        lines.append(f"\nResult:\n{run['result']}")
    if run["error"]:
        lines.append(f"\nError:\n{run['error']}")
    print_panel("\n".join(lines), title=f"Workflow Run #{run_id}")
