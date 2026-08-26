"""Daily digest builder — summarizes device state for Slack.

Gathers network, system, tasks, and events data into a single digest
that can be posted to Slack on a schedule or on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)


def build_digest(db: Database, *, days: int = 1) -> dict[str, Any]:
    """Build a daily digest from the MAOS database.

    Args:
        db: MAOS database.
        days: Look back N days (default 1 = last 24h).

    Returns:
        {network, system, tasks, events} dict.
    """
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    digest: dict[str, Any] = {}

    # Network
    try:
        from app.network import NetworkProvider
        provider = NetworkProvider(db)
        history = provider.get_history(hours=days * 24, resolution="hourly")
        events = db.network_events_since(since, limit=500)
        interruptions = [e for e in events if e["event_type"] == "NETWORK_INTERRUPTED"]
        digest["network"] = {
            "total_rx_gb": round(history["totals"]["rx_bytes"] / 1e9, 2),
            "total_tx_gb": round(history["totals"]["tx_bytes"] / 1e9, 2),
            "interruptions": len(interruptions),
            "hourly_samples": len(history["series"]),
        }
    except Exception as e:
        log.warning("digest.network_failed", error=str(e))
        digest["network"] = {"error": str(e)}

    # System (telemetry averages)
    try:
        rows = db.telemetry_since(since, limit=10000)
        if rows:
            cpu_vals = [r["cpu_percent"] for r in rows]
            mem_vals = [r["memory_used_gb"] for r in rows]
            digest["system"] = {
                "cpu_avg": round(sum(cpu_vals) / len(cpu_vals), 1),
                "cpu_peak": round(max(cpu_vals), 1),
                "mem_avg": round(sum(mem_vals) / len(mem_vals), 1),
                "samples": len(rows),
            }
        else:
            digest["system"] = {"samples": 0}
    except Exception as e:
        digest["system"] = {"error": str(e)}

    # Tasks
    try:
        pending = db.pending_tasks()
        digest["tasks"] = {
            "pending": len(pending),
            "completed": 0,  # would need to query completed tasks
        }
    except Exception:
        digest["tasks"] = {"pending": 0}

    # Events
    try:
        rows = db.query(
            "SELECT type, COUNT(*) as count FROM events WHERE timestamp >= ? "
            "GROUP BY type ORDER BY count DESC LIMIT 10",
            (since,),
        )
        total_row = db.query_one(
            "SELECT COUNT(*) as n FROM events WHERE timestamp >= ?", (since,)
        )
        digest["events"] = {
            "total": total_row["n"] if total_row else 0,
            "by_type": {r["type"]: r["count"] for r in rows},
        }
    except Exception:
        digest["events"] = {"total": 0}

    return digest
