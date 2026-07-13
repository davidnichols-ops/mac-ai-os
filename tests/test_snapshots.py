"""Tests for workspace snapshots and the snapshot collector."""

import subprocess
from datetime import UTC, datetime

from app.db import Database
from app.events.bus import EventBus
from app.observers.snapshots import capture_all_workspaces, capture_snapshot


def test_snapshot_storage(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    snapshot = {
        "timestamp": datetime.now(UTC).isoformat(),
        "workspace": "test-project",
        "git_branch": "main",
        "git_status": "M file.py",
        "modified_files": "M file.py",
        "recent_commands": "git status",
        "active_processes": "python (100MB)",
        "telemetry_id": None,
    }
    sid = db.insert_snapshot(snapshot)
    assert sid > 0

    latest = db.latest_snapshot()
    assert latest["workspace"] == "test-project"
    assert latest["git_branch"] == "main"
    db.close()


def test_capture_snapshot(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)

    # Create a fake git repo.
    repo = tmp_path / "myproject"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)
    (repo / "README.md").write_text("# Test")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

    # Register as workspace.
    db.upsert_workspace(str(repo), "myproject", git_repo=True)

    snapshot = capture_snapshot(str(repo), "myproject", db, bus)
    assert snapshot["id"] > 0
    assert snapshot["git_branch"] in ("main", "master")
    assert snapshot["workspace"] == "myproject"

    # Check event was published.
    events = db.query_events(limit=10)
    assert any(e["type"] == "SNAPSHOT_CAPTURED" for e in events)
    db.close()


def test_capture_all_workspaces(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    # Register two workspaces.
    ws1 = tmp_path / "project1"
    ws1.mkdir()
    (ws1 / "file.py").write_text("print('hi')")
    db.upsert_workspace(str(ws1), "project1")

    ws2 = tmp_path / "project2"
    ws2.mkdir()
    (ws2 / "file.js").write_text("console.log('hi')")
    db.upsert_workspace(str(ws2), "project2")

    snapshots = capture_all_workspaces(db)
    assert len(snapshots) == 2
    assert db.snapshot_count() == 2
    db.close()
