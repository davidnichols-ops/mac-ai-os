"""`maos umbrella` and `maos portfolio` — unified control-plane views."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.umbrella import build_snapshot, render_snapshot, snapshot_json


def umbrella(
    json_output: bool = typer.Option(False, "--json", help="Output the snapshot as JSON"),
) -> None:
    """Show the unified MAOS umbrella view: computer, portfolio, repos, CLIs, MCP configs."""
    rt = build_runtime()
    snapshot = build_snapshot(rt.db)
    if json_output:
        typer.echo(snapshot_json(snapshot))
    else:
        typer.echo(render_snapshot(snapshot))


def portfolio(
    json_output: bool = typer.Option(False, "--json", help="Output the snapshot as JSON"),
) -> None:
    """Alias for `maos umbrella` — show the unified portfolio view."""
    umbrella(json_output=json_output)
