"""Workspace snapshot observer — periodic capture of project state.

Captures: git branch, git status, modified files, recent commands, active
processes. Runs on demand or periodically. Each snapshot is stored in the
workspace_snapshots table and an event is published.

This is what makes the system feel persistent — instead of asking "what was
I working on?", you can ask "what was I working on three weeks ago?"
"""

from __future__ import annotations

import subprocess
import threading
from datetime import UTC, datetime
from typing import Any

from app.db import Database
from app.events.bus import EventBus
from app.events.models import SNAPSHOT_CAPTURED, Event
from app.log import get_logger

log = get_logger(__name__)


def capture_snapshot(
    workspace_path: str, workspace_name: str, db: Database, bus: EventBus | None = None,
) -> dict[str, Any]:
    """Capture a single workspace snapshot."""
    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "workspace": workspace_name,
        "git_branch": None,
        "git_status": None,
        "modified_files": "",
        "recent_commands": "",
        "active_processes": "",
        "telemetry_id": None,
    }

    # Git state.
    try:
        branch = subprocess.run(
            ["git", "-C", workspace_path, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0:
            snapshot["git_branch"] = branch.stdout.strip()

        status = subprocess.run(
            ["git", "-C", workspace_path, "status", "--short"],
            capture_output=True, text=True, timeout=5,
        )
        if status.returncode == 0:
            status_text = status.stdout.strip()
            snapshot["git_status"] = status_text[:500] if status_text else "clean"
            modified = [line.strip() for line in status_text.split("\n") if line.strip()]
            snapshot["modified_files"] = "\n".join(modified[:20])
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Recent shell commands (last 5 for this workspace).
    try:
        commands = db.search_shell_commands("", limit=5)
        # This gets all recent commands; filter by directory if possible.
        cmd_lines = [c["command"][:60] for c in commands[:5]]
        snapshot["recent_commands"] = "\n".join(cmd_lines)
    except Exception:
        pass

    # Active processes (top 5 by memory).
    try:
        import psutil

        procs = []
        for p in sorted(psutil.process_iter(["name", "memory_info"]),
                        key=lambda x: x.info["memory_info"].rss if x.info["memory_info"] else 0,
                        reverse=True)[:5]:
            name = p.info["name"] or "?"
            mem = p.info["memory_info"].rss // (1024 * 1024) if p.info["memory_info"] else 0
            procs.append(f"{name} ({mem}MB)")
        snapshot["active_processes"] = "\n".join(procs)
    except Exception:
        pass

    # Latest telemetry ID.
    try:
        rows = db.query("SELECT id FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT 1")
        if rows:
            snapshot["telemetry_id"] = rows[0]["id"]
    except Exception:
        pass

    # Store in database.
    snapshot_id = db.insert_snapshot(snapshot)

    # Publish event.
    if bus:
        bus.publish(Event(SNAPSHOT_CAPTURED, {
            "workspace": workspace_name,
            "snapshot_id": snapshot_id,
            "modified_count": len(snapshot["modified_files"].split("\n"))
                if snapshot["modified_files"] else 0,
        }))

    log.info("snapshot.captured", workspace=workspace_name,
             snapshot_id=snapshot_id, branch=snapshot["git_branch"])
    return {"id": snapshot_id, **snapshot}


def capture_all_workspaces(db: Database, bus: EventBus | None = None) -> list[dict[str, Any]]:
    """Capture snapshots for all registered workspaces."""
    workspaces = db.all_workspaces()
    snapshots = []
    for ws in workspaces:
        try:
            snap = capture_snapshot(ws["path"], ws["name"], db, bus)
            snapshots.append(snap)
        except Exception as e:
            log.warning("snapshot.failed", workspace=ws["name"], error=str(e))
    log.info("snapshot.batch_done", count=len(snapshots))
    return snapshots


class SnapshotCollector:
    """Background thread that captures workspace snapshots periodically."""

    def __init__(self, db: Database, bus: EventBus | None = None, *, interval: float = 600.0):
        self.db = db
        self.bus = bus
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="snapshot-collector")
        self._thread.start()
        log.info("snapshot.collector.started", interval=self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("snapshot.collector.stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                capture_all_workspaces(self.db, self.bus)
            except Exception as e:
                log.warning("snapshot.collector.error", error=str(e))
            self._stop.wait(self.interval)
