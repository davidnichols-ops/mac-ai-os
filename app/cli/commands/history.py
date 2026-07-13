"""`maos history` — shell command history commands."""

from __future__ import annotations

from datetime import UTC

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

history_app = typer.Typer(name="history", help="Shell history commands.", no_args_is_help=True)


@history_app.command("import")
def import_history(
    shell: str = typer.Option("auto", "--shell", "-s", help="Shell type: auto, zsh, or bash"),
) -> None:
    """Import shell history into the database."""
    rt = build_runtime()
    from app.observers.shell import import_shell_history

    count = import_shell_history(rt.db, shell=shell)
    total = rt.db.shell_command_count()
    print_panel(
        f"Imported {count} new commands.\nTotal in database: {total}",
        title="Shell History Import",
    )


@history_app.command("search")
def search(
    pattern: str = typer.Argument(..., help="Search pattern"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
) -> None:
    """Search shell command history."""
    rt = build_runtime()
    from app.observers.shell import search_history

    results = search_history(rt.db, pattern, limit=limit)
    if not results:
        typer.echo(f"No commands matching '{pattern}'.")
        return

    lines = [f"{'Time':>19s}  Command"]
    for r in results:
        lines.append(f"{r['timestamp'][:19]:>19s}  {r['command'][:70]}")
    print_panel("\n".join(lines), title=f"History Search ({len(results)} results)")


@history_app.command("recent")
def recent(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of commands to show"),
) -> None:
    """Show recent shell commands."""
    rt = build_runtime()
    from datetime import datetime, timedelta

    since = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    commands = rt.db.shell_commands_since(since, limit=limit)
    if not commands:
        typer.echo("No commands in database. Run `maos history import` first.")
        return

    lines = [f"{'Time':>19s}  Command"]
    for c in commands:
        lines.append(f"{c['timestamp'][:19]:>19s}  {c['command'][:70]}")
    print_panel("\n".join(lines), title=f"Recent Commands ({len(commands)})")
