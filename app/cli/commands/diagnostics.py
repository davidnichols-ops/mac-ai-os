"""`maos diagnostics` — check all MAOS tools are working."""

from __future__ import annotations

import typer

from app.diagnostics import format_diagnostic_report, run_diagnostics
from app.runtime import build_runtime

diagnostics_app = typer.Typer(
    name="diagnostics",
    help="Check functionality and implementation status of all MAOS tools.",
    no_args_is_help=False,
)


@diagnostics_app.callback(invoke_without_command=True)
def diagnostics(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run diagnostics on all MAOS MCP tools."""
    if ctx.invoked_subcommand is not None:
        return
    rt = build_runtime()
    report = run_diagnostics(rt)
    if json_output:
        import json
        typer.echo(json.dumps(report.to_dict(), default=str, indent=2))
    else:
        typer.echo(format_diagnostic_report(report))
