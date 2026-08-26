"""NetworkProvider — the public network intelligence API for MAOS.

This is the ONLY interface the rest of MAOS should use. It hides the
collector, the storage schema, and the macOS API details behind three
methods:

    get_status()        -> NetworkStatus  (current health snapshot)
    get_history()       -> dict           (aggregated historical data)
    get_quality_score() -> int            (0-100 health score)

The provider is read-only against the DB. It does not run the collector —
the collector is started by the runtime / CLI `network collect` command.
This separation means a query never blocks on a network probe.

Quality scoring:
    The score combines latency, packet loss, and connection state into a
    single 0-100 number. It's deliberately coarse — the goal is "is the
    network good enough for cloud LLM calls right now?", not a precise
    QoE metric. The thresholds are documented inline so they can be tuned.

    - 100 = perfect (low latency, no loss, connected)
    - 70-99 = good (usable for cloud calls)
    - 40-69 = degraded (consider local models, delay large uploads)
    - 0-39 = poor (cloud calls likely to fail or be very slow)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import Database
from app.log import get_logger
from app.network.types import ConnectionState, NetworkStatus, NetworkType

log = get_logger(__name__)

# Scoring thresholds. Documented so they can be tuned without hunting.
LATENCY_GOOD_MS = 50.0       # below this: no penalty
LATENCY_DEGRADED_MS = 200.0  # above this: significant penalty
LATENCY_POOR_MS = 500.0      # above this: heavy penalty
LOSS_GOOD = 0.0              # fraction
LOSS_DEGRADED = 0.01         # 1%
LOSS_POOR = 0.05             # 5%


class NetworkProvider:
    """Read-only network intelligence provider.

    Constructed with a Database. Stateless across calls — every method
    queries fresh data. Safe to call from any thread (the DB connection is
    shared but SQLite handles concurrent reads).
    """

    def __init__(self, db: Database):
        self.db = db

    def get_status(self) -> NetworkStatus:
        """Current network health. The 'is the network healthy?' answer."""
        row = self.db.latest_network_sample()
        if not row:
            return NetworkStatus(
                healthy=False,
                interface="unknown",
                network_type=NetworkType.UNKNOWN,
                connection_state=ConnectionState.UNKNOWN,
                issues=["no_samples"],
                timestamp=datetime.now(UTC).isoformat(),
            )

        ntype = NetworkType(row["network_type"]) if row["network_type"] else NetworkType.UNKNOWN
        state = ConnectionState(row["connection_state"]) if row["connection_state"] else ConnectionState.UNKNOWN
        latency = row["latency_ms"]
        loss = row["packet_loss"]

        issues: list[str] = []
        if state != ConnectionState.CONNECTED:
            issues.append("disconnected")
        if latency is not None and latency > LATENCY_DEGRADED_MS:
            issues.append("high_latency")
        if loss is not None and loss > LOSS_DEGRADED:
            issues.append("packet_loss")

        score = self.get_quality_score()
        healthy = state == ConnectionState.CONNECTED and score >= 70 and not issues

        return NetworkStatus(
            healthy=healthy,
            interface=row["interface"],
            network_type=ntype,
            connection_state=state,
            rx_rate_bps=row["rx_rate_bps"] or 0.0,
            tx_rate_bps=row["tx_rate_bps"] or 0.0,
            latency_ms=latency,
            packet_loss=loss,
            bandwidth_quality=score,
            issues=issues,
            timestamp=row["timestamp"],
        )

    def get_quality_score(self) -> int:
        """0-100 network quality score. Computed from the latest sample."""
        row = self.db.latest_network_sample()
        if not row:
            return 0

        state = row["connection_state"]
        if state != ConnectionState.CONNECTED.value:
            return 0

        score = 100.0

        # Latency penalty.
        latency = row["latency_ms"]
        if latency is not None:
            if latency > LATENCY_POOR_MS:
                score -= 40.0
            elif latency > LATENCY_DEGRADED_MS:
                # Linear penalty between DEGRADED and POOR.
                score -= 20.0 + 20.0 * (latency - LATENCY_DEGRADED_MS) / (LATENCY_POOR_MS - LATENCY_DEGRADED_MS)
            elif latency > LATENCY_GOOD_MS:
                # Linear penalty between GOOD and DEGRADED.
                score -= 10.0 * (latency - LATENCY_GOOD_MS) / (LATENCY_DEGRADED_MS - LATENCY_GOOD_MS)

        # Packet loss penalty.
        loss = row["packet_loss"]
        if loss is not None:
            if loss > LOSS_POOR:
                score -= 30.0
            elif loss > LOSS_DEGRADED:
                score -= 15.0 + 15.0 * (loss - LOSS_DEGRADED) / (LOSS_POOR - LOSS_DEGRADED)
            elif loss > 0.0:
                score -= 5.0 * (loss / LOSS_DEGRADED)

        # RTT probe failure (latency is None but we're "connected") — likely
        # a captive portal or DNS issue. Small penalty.
        if latency is None and row["rtt_ok"] is not None and not row["rtt_ok"]:
            score -= 15.0

        return max(0, min(100, int(round(score))))

    def get_history(
        self,
        *,
        hours: int = 24,
        resolution: str = "hourly",
    ) -> dict[str, Any]:
        """Aggregated historical bandwidth + latency.

        Args:
            hours: Lookback window in hours.
            resolution: 'hourly' (default, indefinite retention) or 'minute'
                (30-day retention, more detail).

        Returns:
            {
                'resolution': 'hourly'|'minute',
                'since': <iso>,
                'totals': {'rx_bytes': N, 'tx_bytes': N},
                'series': [{bucket, interface, rx_bytes, tx_bytes, ...}, ...],
                'events': [{timestamp, event_type, interface, ...}, ...],
            }
        """
        since_dt = datetime.now(UTC) - timedelta(hours=hours)
        since = since_dt.isoformat()
        since_bucket_minute = since_dt.strftime("%Y-%m-%dT%H:%M")
        since_bucket_hour = since_dt.strftime("%Y-%m-%dT%H")

        if resolution == "minute":
            rows = self.db.network_minute_since(since_bucket_minute, limit=2000)
        else:
            rows = self.db.network_hourly_since(since_bucket_hour, limit=1000)

        series = [dict(r) for r in rows]
        total_rx = sum(r["rx_bytes"] for r in rows)
        total_tx = sum(r["tx_bytes"] for r in rows)

        events = [dict(e) for e in self.db.network_events_since(since, limit=200)]

        return {
            "resolution": resolution,
            "since": since,
            "totals": {"rx_bytes": total_rx, "tx_bytes": total_tx},
            "series": series,
            "events": events,
        }

    def get_daily_usage(self, days: int = 7) -> list[dict[str, Any]]:
        """Per-day RX/TX totals for the last N days. From hourly aggregates."""
        since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H")
        rows = self.db.query(
            "SELECT substr(bucket, 1, 10) as day, "
            "SUM(rx_bytes) as rx_bytes, SUM(tx_bytes) as tx_bytes, "
            "AVG(latency_avg) as latency_avg "
            "FROM network_hourly WHERE bucket >= ? "
            "GROUP BY day ORDER BY day ASC",
            (since,),
        )
        return [dict(r) for r in rows]

    def get_hourly_pattern(self, days: int = 30) -> list[dict[str, Any]]:
        """Average usage by hour-of-day (0-23). Reveals daily patterns."""
        since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H")
        rows = self.db.query(
            "SELECT CAST(substr(bucket, 12, 2) AS INTEGER) as hour, "
            "AVG(rx_bytes) as avg_rx_bytes, AVG(tx_bytes) as avg_tx_bytes, "
            "AVG(latency_avg) as avg_latency "
            "FROM network_hourly WHERE bucket >= ? "
            "GROUP BY hour ORDER BY hour ASC",
            (since,),
        )
        return [dict(r) for r in rows]
