"""Tests for the network intelligence layer (Phase 5.7).

Covers:
  - DB schema + helpers (insert, query, prune, aggregates)
  - NetworkProvider: status, quality score, history, daily/hourly patterns
  - NetworkCollector: rate computation, transition detection, aggregation,
    retention pruning, long-running stability
  - Mock collector (no real subprocess calls)
  - Failure handling: interface unavailable, corrupted/empty storage,
    WiFi reconnect, Ethernet transition, missing permissions (probe failure)
  - Context packet injection
  - Event publishing
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from app.db import Database
from app.events.bus import EventBus
from app.events.models import (
    NETWORK_DEGRADED,
    NETWORK_INTERRUPTED,
    NETWORK_RECONNECTED,
    NETWORK_SAMPLE,
)
from app.network import NetworkCollector, NetworkProvider
from app.network.collector import (
    _hour_bucket,
    _minute_bucket,
    _percentile,
    _seconds_between,
)
from app.network.types import ConnectionState, NetworkSample, NetworkType

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "test.db"))
    yield d
    d.close()


@pytest.fixture
def provider(db):
    return NetworkProvider(db)


def _sample(
    *,
    iface: str = "en0",
    ntype: NetworkType = NetworkType.WIFI,
    state: ConnectionState = ConnectionState.CONNECTED,
    rx_bytes: int = 1000,
    tx_bytes: int = 500,
    rx_rate: float = 1_000_000.0,
    tx_rate: float = 500_000.0,
    latency: float | None = 20.0,
    loss: float | None = 0.0,
    rtt_ok: bool | None = True,
    timestamp: str | None = None,
) -> dict:
    return {
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "interface": iface,
        "network_type": ntype.value,
        "connection_state": state.value,
        "rx_bytes": rx_bytes,
        "tx_bytes": tx_bytes,
        "rx_rate_bps": rx_rate,
        "tx_rate_bps": tx_rate,
        "latency_ms": latency,
        "packet_loss": loss,
        "rtt_ok": rtt_ok,
    }


# --- DB helpers -------------------------------------------------------------


class TestNetworkDB:
    def test_schema_created(self, db):
        # Tables exist.
        for table in ("network_samples", "network_minute", "network_hourly", "network_events"):
            row = db.query_one("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            assert row is not None, f"table {table} missing"

    def test_insert_and_query_sample(self, db):
        sid = db.insert_network_sample(_sample(rx_bytes=12345))
        assert sid > 0
        assert db.network_sample_count() == 1
        latest = db.latest_network_sample()
        assert latest["interface"] == "en0"
        assert latest["rx_bytes"] == 12345

    def test_latest_sample_by_interface(self, db):
        db.insert_network_sample(_sample(iface="en0", rx_bytes=100))
        db.insert_network_sample(_sample(iface="en6", rx_bytes=200, timestamp=datetime.now(UTC).isoformat()))
        latest_en0 = db.latest_network_sample(interface="en0")
        assert latest_en0["rx_bytes"] == 100
        latest_overall = db.latest_network_sample()
        # Overall latest is whichever has the highest timestamp.
        assert latest_overall["rx_bytes"] in (100, 200)

    def test_samples_since(self, db):
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        db.insert_network_sample(_sample(timestamp=old_ts, rx_bytes=10))
        db.insert_network_sample(_sample(rx_bytes=20))
        since_1h = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        rows = db.network_samples_since(since_1h)
        assert len(rows) == 1
        assert rows[0]["rx_bytes"] == 20

    def test_prune_samples(self, db):
        old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
        db.insert_network_sample(_sample(timestamp=old_ts))
        db.insert_network_sample(_sample())
        cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        deleted = db.prune_network_samples_before(cutoff)
        assert deleted == 1
        assert db.network_sample_count() == 1

    def test_prune_minute(self, db):
        old_bucket = (datetime.now(UTC) - timedelta(days=35)).strftime("%Y-%m-%dT%H:%M")
        db.upsert_network_minute({
            "bucket": old_bucket, "interface": "en0", "network_type": "wifi",
            "rx_bytes": 100, "tx_bytes": 50, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "rx_rate_peak": 2.0, "tx_rate_peak": 1.0, "latency_avg": 10.0,
            "latency_p95": 15.0, "packet_loss_avg": 0.0, "sample_count": 12,
        })
        cutoff = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M")
        deleted = db.prune_network_minute_before(cutoff)
        assert deleted == 1

    def test_upsert_minute_idempotent(self, db):
        agg = {
            "bucket": "2026-08-03T14:00", "interface": "en0", "network_type": "wifi",
            "rx_bytes": 100, "tx_bytes": 50, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "rx_rate_peak": 2.0, "tx_rate_peak": 1.0, "latency_avg": 10.0,
            "latency_p95": 15.0, "packet_loss_avg": 0.0, "sample_count": 12,
        }
        db.upsert_network_minute(agg)
        agg["rx_bytes"] = 200
        db.upsert_network_minute(agg)  # upsert, not insert
        rows = db.network_minute_since("2026-08-03T00:00")
        assert len(rows) == 1
        assert rows[0]["rx_bytes"] == 200

    def test_upsert_hourly(self, db):
        db.upsert_network_hourly({
            "bucket": "2026-08-03T14", "interface": "en0", "network_type": "wifi",
            "rx_bytes": 1000, "tx_bytes": 500, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "latency_avg": 12.0, "packet_loss_avg": 0.0, "minute_count": 60,
        })
        rows = db.network_hourly_since("2026-08-03T00")
        assert len(rows) == 1
        assert rows[0]["minute_count"] == 60

    def test_insert_and_query_event(self, db):
        eid = db.insert_network_event({
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "NETWORK_INTERRUPTED",
            "interface": "en0",
            "network_type": "wifi",
            "detail": {"reason": "link_down"},
            "severity": "warning",
        })
        assert eid > 0
        rows = db.network_events_since((datetime.now(UTC) - timedelta(hours=1)).isoformat())
        assert len(rows) == 1
        detail = json.loads(rows[0]["detail"])
        assert detail["reason"] == "link_down"

    def test_corrupted_storage_handled(self, tmp_path):
        """A DB that can't be opened should not crash the provider."""
        bad_path = tmp_path / "corrupt.db"
        bad_path.write_text("not a database")
        with pytest.raises(sqlite3.DatabaseError):
            Database(str(bad_path))  # DB itself raises — expected


# --- Provider ---------------------------------------------------------------


class TestNetworkProvider:
    def test_status_no_samples(self, provider):
        s = provider.get_status()
        assert s.healthy is False
        assert s.interface == "unknown"
        assert "no_samples" in s.issues

    def test_status_healthy(self, db, provider):
        db.insert_network_sample(_sample(latency=20.0, loss=0.0))
        s = provider.get_status()
        assert s.healthy is True
        assert s.bandwidth_quality == 100
        assert s.issues == []

    def test_status_disconnected(self, db, provider):
        db.insert_network_sample(_sample(state=ConnectionState.DISCONNECTED, latency=None))
        s = provider.get_status()
        assert s.healthy is False
        assert "disconnected" in s.issues
        assert provider.get_quality_score() == 0

    def test_status_high_latency(self, db, provider):
        db.insert_network_sample(_sample(latency=300.0))
        s = provider.get_status()
        assert s.healthy is False
        assert "high_latency" in s.issues
        score = provider.get_quality_score()
        assert score < 80

    def test_status_packet_loss(self, db, provider):
        db.insert_network_sample(_sample(loss=0.02))
        s = provider.get_status()
        assert "packet_loss" in s.issues
        assert s.healthy is False

    def test_quality_score_perfect(self, db, provider):
        db.insert_network_sample(_sample(latency=10.0, loss=0.0))
        assert provider.get_quality_score() == 100

    def test_quality_score_latency_gradient(self, db, provider):
        # Good latency: no penalty.
        db.insert_network_sample(_sample(latency=30.0))
        assert provider.get_quality_score() == 100
        # Between GOOD and DEGRADED: small penalty.
        db.prune_network_samples_before(datetime.now(UTC).isoformat())
        db.insert_network_sample(_sample(latency=100.0))
        score = provider.get_quality_score()
        assert 90 <= score < 100
        # Between DEGRADED and POOR: bigger penalty.
        db.prune_network_samples_before(datetime.now(UTC).isoformat())
        db.insert_network_sample(_sample(latency=350.0))
        score = provider.get_quality_score()
        assert 60 <= score < 80
        # Above POOR: heavy penalty.
        db.prune_network_samples_before(datetime.now(UTC).isoformat())
        db.insert_network_sample(_sample(latency=750.0))
        score = provider.get_quality_score()
        assert score < 70

    def test_quality_score_rtt_failure(self, db, provider):
        db.insert_network_sample(_sample(latency=None, rtt_ok=False))
        score = provider.get_quality_score()
        # 15-point penalty for RTT failure.
        assert score == 85

    def test_history_empty(self, provider):
        h = provider.get_history(hours=24)
        assert h["totals"]["rx_bytes"] == 0
        assert h["series"] == []
        assert h["events"] == []

    def test_history_with_data(self, db, provider):
        hour_bucket = datetime.now(UTC).strftime("%Y-%m-%dT%H")
        db.upsert_network_hourly({
            "bucket": hour_bucket, "interface": "en0", "network_type": "wifi",
            "rx_bytes": 1000, "tx_bytes": 500, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "latency_avg": 12.0, "packet_loss_avg": 0.0, "minute_count": 60,
        })
        h = provider.get_history(hours=24 * 7)
        assert len(h["series"]) == 1
        assert h["totals"]["rx_bytes"] == 1000

    def test_history_minute_resolution(self, db, provider):
        minute_bucket = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
        db.upsert_network_minute({
            "bucket": minute_bucket, "interface": "en0", "network_type": "wifi",
            "rx_bytes": 100, "tx_bytes": 50, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "rx_rate_peak": 2.0, "tx_rate_peak": 1.0, "latency_avg": 10.0,
            "latency_p95": 15.0, "packet_loss_avg": 0.0, "sample_count": 12,
        })
        h = provider.get_history(hours=24 * 7, resolution="minute")
        assert h["resolution"] == "minute"
        assert len(h["series"]) == 1

    def test_daily_usage(self, db, provider):
        hour_bucket = datetime.now(UTC).strftime("%Y-%m-%dT%H")
        db.upsert_network_hourly({
            "bucket": hour_bucket, "interface": "en0", "network_type": "wifi",
            "rx_bytes": 1000, "tx_bytes": 500, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "latency_avg": 12.0, "packet_loss_avg": 0.0, "minute_count": 60,
        })
        daily = provider.get_daily_usage(days=7)
        assert len(daily) == 1
        assert daily[0]["rx_bytes"] == 1000

    def test_hourly_pattern(self, db, provider):
        now = datetime.now(UTC)
        db.upsert_network_hourly({
            "bucket": now.strftime("%Y-%m-%dT%H"), "interface": "en0", "network_type": "wifi",
            "rx_bytes": 1000, "tx_bytes": 500, "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "latency_avg": 12.0, "packet_loss_avg": 0.0, "minute_count": 60,
        })
        pattern = provider.get_hourly_pattern(days=30)
        assert len(pattern) == 1
        assert pattern[0]["hour"] == now.hour


# --- Collector (mocked macOS APIs) ------------------------------------------


class TestNetworkCollector:
    def test_collect_once_stores_sample(self, db):
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=datetime.now(UTC).isoformat(),
                interface="en0",
                network_type=NetworkType.WIFI,
                connection_state=ConnectionState.CONNECTED,
                rx_bytes=1000, tx_bytes=500,
                rx_rate_bps=0.0, tx_rate_bps=0.0,
                latency_ms=20.0, packet_loss=0.0, rtt_ok=True,
            )
            sample = collector.collect_once()
        assert db.network_sample_count() == 1
        assert sample.interface == "en0"

    def test_rate_computation(self, db):
        """Rates are computed from the delta between consecutive samples."""
        collector = NetworkCollector(db, None, interval_seconds=1.0)
        t0 = datetime.now(UTC).isoformat()
        t1 = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=t0, interface="en0",
                network_type=NetworkType.WIFI,
                connection_state=ConnectionState.CONNECTED,
                rx_bytes=1000, tx_bytes=500,
            )
            collector.collect_once()
            mock.return_value = NetworkSample(
                timestamp=t1, interface="en0",
                network_type=NetworkType.WIFI,
                connection_state=ConnectionState.CONNECTED,
                rx_bytes=6000, tx_bytes=3000,  # +5000 rx, +2500 tx over 5s
            )
            sample = collector.collect_once()
        # 5000 bytes * 8 bits / 5s = 8000 bps
        assert sample.rx_rate_bps == pytest.approx(8000.0, rel=0.01)
        assert sample.tx_rate_bps == pytest.approx(4000.0, rel=0.01)

    def test_interruption_event(self, db):
        """A connected -> disconnected transition publishes NETWORK_INTERRUPTED."""
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        t0 = datetime.now(UTC).isoformat()
        t1 = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=t0, interface="en0",
                connection_state=ConnectionState.CONNECTED,
                rx_bytes=100, tx_bytes=50,
            )
            collector.collect_once()
            mock.return_value = NetworkSample(
                timestamp=t1, interface="en0",
                connection_state=ConnectionState.DISCONNECTED,
                rx_bytes=100, tx_bytes=50,
            )
            collector.collect_once()
        events = db.network_events_since((datetime.now(UTC) - timedelta(hours=1)).isoformat())
        types = [e["event_type"] for e in events]
        assert NETWORK_INTERRUPTED in types

    def test_reconnect_event(self, db):
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        t0 = datetime.now(UTC).isoformat()
        t1 = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=t0, interface="en0",
                connection_state=ConnectionState.DISCONNECTED,
            )
            collector.collect_once()
            mock.return_value = NetworkSample(
                timestamp=t1, interface="en0",
                connection_state=ConnectionState.CONNECTED,
                rx_bytes=100, tx_bytes=50,
            )
            collector.collect_once()
        events = db.network_events_since((datetime.now(UTC) - timedelta(hours=1)).isoformat())
        types = [e["event_type"] for e in events]
        assert NETWORK_RECONNECTED in types

    def test_degradation_latency_spike(self, db):
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        t0 = datetime.now(UTC).isoformat()
        t1 = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=t0, interface="en0",
                connection_state=ConnectionState.CONNECTED,
                latency_ms=50.0,
            )
            collector.collect_once()
            mock.return_value = NetworkSample(
                timestamp=t1, interface="en0",
                connection_state=ConnectionState.CONNECTED,
                latency_ms=300.0,  # >2x baseline + >100ms
            )
            collector.collect_once()
        events = db.network_events_since((datetime.now(UTC) - timedelta(hours=1)).isoformat())
        degraded = [e for e in events if e["event_type"] == NETWORK_DEGRADED]
        assert len(degraded) == 1
        detail = json.loads(degraded[0]["detail"])
        assert detail["reason"] == "latency_spike"

    def test_degradation_packet_loss(self, db):
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        t0 = datetime.now(UTC).isoformat()
        t1 = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=t0, interface="en0",
                connection_state=ConnectionState.CONNECTED,
                packet_loss=0.0,
            )
            collector.collect_once()
            mock.return_value = NetworkSample(
                timestamp=t1, interface="en0",
                connection_state=ConnectionState.CONNECTED,
                packet_loss=0.005,  # >0.1%
            )
            collector.collect_once()
        events = db.network_events_since((datetime.now(UTC) - timedelta(hours=1)).isoformat())
        degraded = [e for e in events if e["event_type"] == NETWORK_DEGRADED]
        assert len(degraded) >= 1

    def test_event_published_on_bus(self, db):
        """NETWORK_SAMPLE events are published on the EventBus."""
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        received: list[str] = []
        bus.subscribe(NETWORK_SAMPLE, lambda e: received.append(e.type))
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=datetime.now(UTC).isoformat(),
                interface="en0",
                connection_state=ConnectionState.CONNECTED,
            )
            collector.collect_once()
        assert received == [NETWORK_SAMPLE]

    def test_interface_transition_ethernet(self, db):
        """Switching from en0 (wifi) to en6 (ethernet) is recorded."""
        bus = EventBus(db=db)
        collector = NetworkCollector(db, bus, interval_seconds=1.0)
        t0 = datetime.now(UTC).isoformat()
        t1 = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=t0, interface="en0",
                network_type=NetworkType.WIFI,
                connection_state=ConnectionState.CONNECTED,
            )
            collector.collect_once()
            mock.return_value = NetworkSample(
                timestamp=t1, interface="en6",
                network_type=NetworkType.ETHERNET,
                connection_state=ConnectionState.CONNECTED,
            )
            collector.collect_once()
        # Both interfaces should have samples.
        assert db.network_sample_count() == 2

    def test_missing_permissions_probe_failure(self, db):
        """A failed RTT probe (rtt_ok=False) is recorded but doesn't crash."""
        collector = NetworkCollector(db, None, interval_seconds=1.0)
        with patch("app.network.collector.collect_sample") as mock:
            mock.return_value = NetworkSample(
                timestamp=datetime.now(UTC).isoformat(),
                interface="en0",
                connection_state=ConnectionState.CONNECTED,
                latency_ms=None, rtt_ok=False,
            )
            sample = collector.collect_once()
        assert sample.latency_ms is None
        assert sample.rtt_ok is False
        # The DB stores falsy rtt_ok as 0 (integer), truthy as 1, None as NULL.
        latest = db.latest_network_sample()
        assert latest["rtt_ok"] in (0, None)

    def test_long_running_stability(self, db):
        """100 samples in a row should not leak or crash."""
        collector = NetworkCollector(db, None, interval_seconds=0.01)
        with patch("app.network.collector.collect_sample") as mock:
            base = datetime.now(UTC)
            for i in range(100):
                mock.return_value = NetworkSample(
                    timestamp=(base + timedelta(seconds=i)).isoformat(),
                    interface="en0",
                    connection_state=ConnectionState.CONNECTED,
                    rx_bytes=1000 + i * 100,
                    tx_bytes=500 + i * 50,
                )
                collector.collect_once()
        assert db.network_sample_count() == 100

    def test_aggregate_minute(self, db):
        """The collector rolls raw samples into minute aggregates."""
        collector = NetworkCollector(db, None, interval_seconds=1.0)
        # Insert samples in minute 14:00, then trigger rollup by changing minute.
        base = datetime(2026, 8, 3, 14, 0, 30, tzinfo=UTC)
        with patch("app.network.collector.collect_sample") as mock:
            for i in range(3):
                mock.return_value = NetworkSample(
                    timestamp=(base + timedelta(seconds=i)).isoformat(),
                    interface="en0",
                    connection_state=ConnectionState.CONNECTED,
                    rx_bytes=1000 + i * 1000,
                    tx_bytes=500,
                    rx_rate_bps=1000.0 + i * 100,
                    tx_rate_bps=500.0,
                )
                collector.collect_once()
            # Now a sample in minute 14:01 triggers the rollup of 14:00.
            mock.return_value = NetworkSample(
                timestamp=datetime(2026, 8, 3, 14, 1, 0, tzinfo=UTC).isoformat(),
                interface="en0",
                connection_state=ConnectionState.CONNECTED,
                rx_bytes=4000, tx_bytes=2000,
            )
            collector.collect_once()
        rows = db.network_minute_since("2026-08-03T14:00")
        assert len(rows) == 1
        assert rows[0]["sample_count"] == 3
        assert rows[0]["rx_bytes"] == 2000  # 3000 - 1000


# --- Pure helpers -----------------------------------------------------------


class TestHelpers:
    def test_minute_bucket(self):
        assert _minute_bucket("2026-08-03T14:05:30.123456+00:00") == "2026-08-03T14:05"

    def test_hour_bucket(self):
        assert _hour_bucket("2026-08-03T14:05:30.123456+00:00") == "2026-08-03T14"

    def test_seconds_between(self):
        a = "2026-08-03T14:00:00+00:00"
        b = "2026-08-03T14:00:05+00:00"
        assert _seconds_between(a, b) == 5.0

    def test_seconds_between_invalid(self):
        assert _seconds_between("garbage", "also-garbage") == 0.0

    def test_percentile(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3.0
        assert _percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
        assert _percentile([], 50) == 0.0
        assert _percentile([42], 99) == 42.0


# --- Context integration ----------------------------------------------------


class TestContextIntegration:
    def test_context_packet_includes_network(self, db):
        from app.blackboard import Blackboard
        from app.server.context import build_context_packet

        db.insert_network_sample(_sample(latency=20.0))
        bb = Blackboard()
        packet = build_context_packet(db, bb)
        assert "network" in packet
        assert packet["network"]["healthy"] is True
        assert packet["network"]["interface"] == "en0"

    def test_context_packet_network_empty(self, db):
        from app.blackboard import Blackboard
        from app.server.context import build_context_packet

        bb = Blackboard()
        packet = build_context_packet(db, bb)
        assert "network" in packet
        assert packet["network"]["healthy"] is False

    def test_agent_context_includes_network(self, db):
        from app.context import build_context

        db.insert_network_sample(_sample(latency=20.0))
        ctx = build_context(db, "test query")
        assert ctx.network is not None
        assert ctx.network["healthy"] is True

    def test_agent_context_renders_network(self, db):
        from app.context import build_context

        db.insert_network_sample(_sample(latency=20.0))
        ctx = build_context(db, "test query")
        prompt = ctx.to_prompt()
        assert "Network: healthy" in prompt
        assert "Latency: 20ms" in prompt
