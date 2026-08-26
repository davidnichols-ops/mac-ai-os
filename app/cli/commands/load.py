"""`maos load` — MAOS Load Workbench.

Pressure-test every MAOS stack at scale, find practical break points, and
produce a report. Default mode uses synthetic data to be I/O-bound on the
local SQLite store; use `--real-telemetry` to exercise psutil.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from app.load import LoadWorkbench
from app.ui.console import print_panel

load_app = typer.Typer(
    name="load",
    help="Synthetic user-interaction load workbench.",
    no_args_is_help=True,
)


@load_app.command("run")
def run(
    interactions: int = typer.Argument(1000, help="Number of synthetic user interactions"),
    db_path: str = typer.Option("data/load.db", "--db", help="Path to load-test database"),
    home: str = typer.Option("", "--home", help="Synthetic home directory (default: system temp)"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Commit every N interactions"),
    real_telemetry: bool = typer.Option(False, "--real-telemetry", help="Use real psutil telemetry"),
    json_output: bool = typer.Option(False, "--json", help="Emit report as JSON"),
) -> None:
    """Run N synthetic interactions through every MAOS stack and report."""
    # Default to a temp directory so we don't clutter the real home.
    home_path = Path(home) if home else Path.home() / ".maos-load"
    home_path.mkdir(parents=True, exist_ok=True)

    bench = LoadWorkbench(
        db_path=db_path,
        home=home_path,
        batch_size=batch_size,
        real_telemetry=real_telemetry,
    )
    report = bench.run(interactions)

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        f"Interactions: {report.interactions}",
        f"Duration:     {report.duration_s:.2f}s",
        f"Throughput:   {report.throughput_ips} ips",
        f"DB size:      {report.db_size_mb} MB",
        f"Projected 50M: {report.projected_50m_h:.2f}h",
        "",
        "Stack hits:",
    ]
    for name, count in sorted(report.stack_hits.items(), key=lambda x: -x[1]):
        lines.append(f"  {name}: {count}")

    if report.errors:
        lines.append("")
        lines.append("Errors:")
        for name, count in sorted(report.errors.items()):
            lines.append(f"  {name}: {count}")

    if report.blockers:
        lines.append("")
        lines.append("Blockers for 50M:")
        for blocker in report.blockers:
            lines.append(f"  - {blocker}")

    print_panel("\n".join(lines), title="MAOS Load Report")
