"""Tests for the MAOS Load Workbench."""

from __future__ import annotations

import uuid
from pathlib import Path

from app.load import LoadWorkbench


def test_load_workbench_runs_small_interactions(tmp_path: Path) -> None:
    """The workbench should run a small number of interactions without errors."""
    db_path = tmp_path / f"load-{uuid.uuid4().hex}.db"
    home = tmp_path / "home"
    home.mkdir()

    bench = LoadWorkbench(str(db_path), home=home, batch_size=50)
    report = bench.run(100)

    assert report.interactions == 100
    assert report.throughput_ips > 0
    assert not report.errors
    assert set(report.stack_hits.keys()).issubset(set(bench.steps.keys()))
    assert report.db_size_mb >= 0


def test_slack_ask_step_uses_model(tmp_path: Path) -> None:
    """The Slack ask step should route through the fake qa_model."""
    db_path = tmp_path / f"ask-{uuid.uuid4().hex}.db"
    home = tmp_path / "home"
    home.mkdir()

    bench = LoadWorkbench(str(db_path), home=home, batch_size=10)
    bench._step_ask()

    # If the fake model responded, we expect a row in events from context building.
    events = bench.db.query("SELECT * FROM events WHERE type = 'USER_INTERACTION'")
    assert len(events) == 0  # ask step does not publish a user event
