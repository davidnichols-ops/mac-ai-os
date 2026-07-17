"""`maos time` — time awareness for AI agents."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.time import build_time_packet, format_time_packet

time_app = typer.Typer(
    name="time",
    help="Time awareness — current time, session duration, tool call stats.",
    no_args_is_help=False,
)


@time_app.callback(invoke_without_command=True)
def time_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current time and session duration information."""
    if ctx.invoked_subcommand is not None:
        return
    rt = build_runtime()
    packet = build_time_packet(rt.db)
    if json_output:
        import json
        typer.echo(json.dumps(packet, default=str, indent=2))
    else:
        typer.echo(format_time_packet(packet))
