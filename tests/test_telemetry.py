from datetime import UTC

from app.db import Database
from app.telemetry.collector import TelemetryCollector, _collect_snapshot


def test_collect_snapshot():
    snap = _collect_snapshot()
    assert "timestamp" in snap
    assert "cpu_percent" in snap
    assert "memory_total_gb" in snap
    # GPU should be None (not available without root).
    assert snap.get("gpu_usage") is None


def test_telemetry_storage(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    collector = TelemetryCollector(db, interval_seconds=1.0)
    snap = collector.collect_once()

    assert db.telemetry_count() == 1
    latest = db.latest_telemetry()
    assert latest is not None
    assert latest["cpu_percent"] == snap["cpu_percent"]

    # Check processes were stored.
    procs = db.processes_for_telemetry(latest["id"])
    assert len(procs) > 0
    db.close()


def test_telemetry_since(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    collector = TelemetryCollector(db, interval_seconds=1.0)
    collector.collect_once()

    from datetime import datetime, timedelta
    since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    rows = db.telemetry_since(since)
    assert len(rows) == 1
    db.close()
