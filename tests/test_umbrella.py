from __future__ import annotations

from app.db import Database
from app.umbrella import build_snapshot, render_snapshot, snapshot_json


def test_umbrella_snapshot_collects_local_surface(tmp_path) -> None:
    home = tmp_path
    (home / "Projects" / "demo" / ".git").mkdir(parents=True)
    (home / "bin").mkdir()
    (home / "bin" / "custom-cli").write_text("#!/bin/sh\n")
    (home / "mac-ai-os" / "docs").mkdir(parents=True)
    (home / "mac-ai-os" / "docs" / "vision.md").write_text("vision")
    db = Database(str(tmp_path / "maos.db"))
    db.create_task("Umbrella test")

    snapshot = build_snapshot(db, home=home)

    assert snapshot.pending_tasks == 1
    assert snapshot.repositories[0]["name"] == "demo"
    assert str(home / "bin" / "custom-cli") in snapshot.custom_clis
    assert "docs/vision.md" in snapshot.docs
    assert "MAOS Umbrella" in render_snapshot(snapshot)
    assert '"repositories"' in snapshot_json(snapshot)
    db.close()
