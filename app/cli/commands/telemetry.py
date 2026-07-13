"""`maos telemetry` — system telemetry commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

telemetry_app = typer.Typer(name="telemetry", help="System telemetry commands.", no_args_is_help=True)


@telemetry_app.command("status")
def status() -> None:
    """Show the latest telemetry snapshot."""
    from app.telemetry.collector import _collect_snapshot, _collect_top_processes

    snap = _collect_snapshot()
    procs = _collect_top_processes(10)

    lines = [
        f"Time:      {snap['timestamp'][:19]}",
        f"CPU:       {snap['cpu_percent']:.1f}% ({snap['cpu_cores']} cores)",
        f"Memory:    {snap['memory_used_gb']:.1f} / {snap['memory_total_gb']:.1f} GB ({snap['memory_percent']:.0f}%)",
        f"Swap:      {snap.get('swap_used_gb', 0):.1f} / {snap.get('swap_total_gb', 0):.1f} GB",
        f"Disk R/W:  {snap.get('disk_read_mb', 0):.0f} / {snap.get('disk_write_mb', 0):.0f} MB",
        f"Processes: {snap.get('process_count', 0)}",
        f"Thermal:   {snap.get('thermal_state', 'unknown')}",
    ]
    if snap.get("battery_percent") is not None:
        plugged = "plugged" if snap.get("battery_plugged") else "on battery"
        lines.append(f"Battery:   {snap['battery_percent']}% ({plugged})")
    lines.append(f"GPU:       {snap.get('gpu_usage') or 'n/a (needs root)'}")
    lines.append(f"NE:        {snap.get('neural_engine') or 'n/a (no public API)'}")

    if procs:
        lines.append("\nTop processes by memory:")
        for p in procs[:5]:
            lines.append(f"  {p['name'][:20]:>20s}  {p['memory_mb']:6.0f}MB  cpu={p['cpu']:.1f}%")

    print_panel("\n".join(lines), title="Telemetry Status")


@telemetry_app.command("collect")
def collect(
    seconds: int = typer.Option(30, "--seconds", "-s", help="How long to collect"),
    interval: float = typer.Option(5.0, "--interval", "-i", help="Seconds between samples"),
) -> None:
    """Collect telemetry for a duration, then show a summary."""
    rt = build_runtime()
    from app.telemetry.collector import TelemetryCollector

    collector = TelemetryCollector(rt.db, rt.bus, interval_seconds=interval)
    collector.start()
    typer.echo(f"Collecting telemetry for {seconds}s (every {interval}s)...")
    import time
    time.sleep(seconds)
    collector.stop()

    count = rt.db.telemetry_count()
    latest = rt.db.latest_telemetry()
    print_panel(
        f"Collected {count} snapshots total.\n"
        f"Latest: CPU={latest['cpu_percent']:.1f}% "
        f"Mem={latest['memory_used_gb']:.1f}/{latest['memory_total_gb']:.1f}GB",
        title="Telemetry Collection",
    )


@telemetry_app.command("history")
def history(
    hours: int = typer.Option(1, "--hours", "-h", help="Look back N hours"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max snapshots to show"),
) -> None:
    """Show telemetry history from the database."""
    rt = build_runtime()
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    rows = rt.db.telemetry_since(since, limit=limit)

    if not rows:
        print_panel(f"No telemetry data in the last {hours} hour(s).", title="History")
        return

    lines = [f"{'Time':>19s}  {'CPU%':>5s}  {'Mem GB':>7s}  {'Swap GB':>7s}  {'Procs':>5s}"]
    for r in rows:
        lines.append(
            f"{r['timestamp'][11:19]:>19s}  {r['cpu_percent']:5.1f}  "
            f"{r['memory_used_gb']:5.1f}/{r['memory_total_gb']:.0f}  "
            f"{r['swap_used_gb']:7.1f}  {r['process_count']:5d}"
        )
    print_panel("\n".join(lines), title=f"Telemetry History ({len(rows)} snapshots)")
