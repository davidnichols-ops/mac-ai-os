"""`maos timeline` — query the activity event stream."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel


def timeline(
    hours: int = typer.Option(1, "--hours", "-h", help="Look back N hours"),
    event_type: str = typer.Option(None, "--type", "-t", help="Filter by event type"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max events to show"),
) -> None:
    """Show the activity timeline from the events table."""
    rt = build_runtime()
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    events = rt.db.query_events(event_type=event_type, since=since, limit=limit)

    if not events:
        print_panel(f"No events in the last {hours} hour(s).", title="Timeline")
        return

    lines = [f"{'Time':>19s}  {'Type':>22s}  Payload"]
    for e in events:
        payload_str = e["payload"]
        try:
            payload = json.loads(payload_str)
            # Summarize payload to one line.
            summary = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:3])
        except Exception:
            summary = payload_str[:60]
        lines.append(f"{e['timestamp'][11:19]:>19s}  {e['type']:>22s}  {summary}")

    print_panel("\n".join(lines), title=f"Activity Timeline ({len(events)} events)")


def event_types() -> None:
    """Show all event types that have been recorded."""
    rt = build_runtime()
    rows = rt.db.query(
        "SELECT type, COUNT(*) as n FROM events GROUP BY type ORDER BY n DESC"
    )
    if not rows:
        typer.echo("No events recorded yet.")
        return
    lines = [f"  {r['n']:>5d}  {r['type']}" for r in rows]
    print_panel("\n".join(lines), title="Event Types")
