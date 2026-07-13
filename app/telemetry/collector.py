"""Telemetry collector — periodically samples system state and stores it.

Collects what's honestly available via psutil on macOS without root:
  - CPU percent, core count
  - Memory: total, used, percent, swap
  - Disk I/O: read/write MB/s
  - Battery: percent, plugged
  - Thermal state: best-effort via `pmset -g therm`
  - Process list: top N by CPU and memory

NOT available without root or private APIs (left as None):
  - GPU utilization (needs `powermetrics` with sudo or Metal private API)
  - Neural Engine utilization (no public API)

The collector runs in a background thread and publishes SYSTEM_SNAPSHOT events
on the event bus. Each snapshot is stored in the telemetry_snapshots table
with top processes in process_snapshots.
"""

from __future__ import annotations

import subprocess
import threading
from datetime import UTC, datetime
from typing import Any

from app.db import Database
from app.events.bus import EventBus
from app.events.models import SYSTEM_SNAPSHOT, Event
from app.log import get_logger

log = get_logger(__name__)


def _collect_snapshot() -> dict[str, Any]:
    """Collect a single telemetry snapshot using psutil."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return {"timestamp": datetime.now(UTC).isoformat(), "cpu_percent": 0}

    snap: dict[str, Any] = {"timestamp": datetime.now(UTC).isoformat()}
    snap["cpu_percent"] = psutil.cpu_percent(interval=0.5)
    snap["cpu_cores"] = psutil.cpu_count(logical=True) or 0

    vm = psutil.virtual_memory()
    snap["memory_total_gb"] = round(vm.total / (1024**3), 2)
    snap["memory_used_gb"] = round(vm.used / (1024**3), 2)
    snap["memory_percent"] = vm.percent

    try:
        swap = psutil.swap_memory()
        snap["swap_used_gb"] = round(swap.used / (1024**3), 2)
        snap["swap_total_gb"] = round(swap.total / (1024**3), 2)
    except Exception:
        snap["swap_used_gb"] = 0
        snap["swap_total_gb"] = 0

    try:
        disk = psutil.disk_io_counters()
        if disk:
            snap["disk_read_mb"] = round(disk.read_bytes / (1024**2), 1)
            snap["disk_write_mb"] = round(disk.write_bytes / (1024**2), 1)
        else:
            snap["disk_read_mb"] = 0
            snap["disk_write_mb"] = 0
    except Exception:
        snap["disk_read_mb"] = 0
        snap["disk_write_mb"] = 0

    batt = psutil.sensors_battery()
    if batt is not None:
        snap["battery_percent"] = round(batt.percent)
        snap["battery_plugged"] = 1 if batt.power_plugged else 0
    else:
        snap["battery_percent"] = None
        snap["battery_plugged"] = None

    snap["thermal_state"] = _thermal_state()
    snap["process_count"] = len(psutil.pids())

    # GPU and Neural Engine: not reliably available without root.
    snap["gpu_usage"] = None
    snap["neural_engine"] = None

    return snap


def _collect_top_processes(n: int = 10) -> list[dict[str, Any]]:
    """Collect top processes by memory (with CPU percent)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return []

    procs: list[dict[str, Any]] = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            mem_pct = float(info.get("memory_percent") or 0.0)
            mem_total_gb = psutil.virtual_memory().total / (1024**3)
            procs.append({
                "pid": info["pid"],
                "name": info["name"] or "unknown",
                "cpu": float(info.get("cpu_percent") or 0.0),
                "memory_mb": round(mem_pct * mem_total_gb * 1024 / 100, 1),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda x: x["memory_mb"], reverse=True)
    return procs[:n]


def _thermal_state() -> str:
    """Best-effort thermal state via pmset (no root required)."""
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True, timeout=2)
        for line in out.splitlines():
            line = line.strip()
            if line and "CPU" in line:
                return line
        return "nominal"
    except Exception:
        return "unknown"


class TelemetryCollector:
    """Background thread that collects and stores telemetry snapshots."""

    def __init__(
        self,
        db: Database,
        bus: EventBus | None = None,
        *,
        interval_seconds: float = 5.0,
        top_processes: int = 10,
    ):
        self.db = db
        self.bus = bus
        self.interval = interval_seconds
        self.top_n = top_processes
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("telemetry.collector.started", interval=self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("telemetry.collector.stopped")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.collect_once()
            except Exception as e:
                log.warning("telemetry.collect.failed", error=str(e))
            self._stop_event.wait(self.interval)

    def collect_once(self) -> dict[str, Any]:
        """Collect one snapshot, store it, publish an event. Returns the snapshot."""
        snap = _collect_snapshot()
        procs = _collect_top_processes(self.top_n)
        telemetry_id = self.db.insert_telemetry(snap, procs)
        snap["id"] = telemetry_id

        if self.bus:
            self.bus.publish(Event(
                SYSTEM_SNAPSHOT,
                {
                    "cpu_percent": snap["cpu_percent"],
                    "memory_used_gb": snap["memory_used_gb"],
                    "memory_total_gb": snap["memory_total_gb"],
                    "telemetry_id": telemetry_id,
                },
            ))
        log.debug("telemetry.collected", cpu=snap["cpu_percent"], mem=snap["memory_used_gb"])
        return snap
