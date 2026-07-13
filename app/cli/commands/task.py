"""`maos task` — task management commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

task_app = typer.Typer(name="task", help="Task management commands.", no_args_is_help=True)


@task_app.command("add")
def add(
    title: str = typer.Argument(..., help="Task title"),
    workspace: str = typer.Option("", "--workspace", "-w", help="Related workspace"),
    priority: int = typer.Option(0, "--priority", "-p", help="Priority (higher = more important)"),
) -> None:
    """Create a new task."""
    rt = build_runtime()
    task_id = rt.db.create_task(title, source="manual", related_workspace=workspace, priority=priority)
    print_panel(f"Created task #{task_id}: {title}", title="Task Created")


@task_app.command("list")
def list_tasks(
    workspace: str = typer.Option("", "--workspace", "-w", help="Filter by workspace"),
    all: bool = typer.Option(False, "--all", help="Show all tasks (not just pending)"),
) -> None:
    """List tasks."""
    rt = build_runtime()
    if all:
        tasks = rt.db.all_tasks()
    else:
        tasks = rt.db.pending_tasks(workspace=workspace or None)

    if not tasks:
        typer.echo("No tasks found.")
        return

    lines = [f"{'ID':>4s}  {'Status':>10s}  {'Pri':>3s}  Title"]
    for t in tasks:
        lines.append(f"{t['id']:4d}  {t['status']:>10s}  {t['priority']:3d}  {t['title']}")
    print_panel("\n".join(lines), title=f"Tasks ({len(tasks)})")


@task_app.command("done")
def done(
    task_id: int = typer.Argument(..., help="Task ID to complete"),
) -> None:
    """Mark a task as completed."""
    rt = build_runtime()
    rt.db.complete_task(task_id)
    typer.echo(f"Completed task #{task_id}")


@task_app.command("delete")
def delete(
    task_id: int = typer.Argument(..., help="Task ID to delete"),
) -> None:
    """Delete a task."""
    rt = build_runtime()
    rt.db.delete_task(task_id)
    typer.echo(f"Deleted task #{task_id}")
