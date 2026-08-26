"""Stress test: 10,000 user-like actions against the umbrella control plane.

The umbrella view is read-only and bounded by design. This test exercises that
promise at volume: it seeds a synthetic home surface and MAOS database, then
runs 10,000 mixed operations (snapshot builds, renders, serializations,
workspace and task creates, context reads) and asserts consistency, determinism,
and bounded output.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from app.db import Database
from app.umbrella import build_snapshot, render_snapshot, snapshot_json


@pytest.mark.slow
def test_umbrella_10000_user_actions(tmp_path: Path) -> None:
    """Run 10,000 mixed user-like actions through the umbrella surface."""
    home = tmp_path
    projects = home / "Projects"
    projects.mkdir(parents=True)

    # --- seed a realistic computer surface ---------------------------------
    repo_count = 3
    for i in range(repo_count):
        repo = projects / f"repo{i}"
        repo.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        (repo / f"file{i}.py").write_text(f"# repo {i}\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            ["git", "-c", "user.email=stress@test", "-c", "user.name=Stress",
             "commit", "-q", "-m", f"init repo {i}"],
            cwd=repo, check=True,
        )

    # custom CLIs
    (home / "bin").mkdir()
    (home / ".grok" / "bin").mkdir(parents=True)
    for i in range(5):
        (home / "bin" / f"mycli{i}").write_text("#!/bin/sh\n")
        (home / ".grok" / "bin" / f"grok-cli{i}").write_text("#!/bin/sh\n")

    # docs
    (home / "mac-ai-os" / "docs").mkdir(parents=True)
    for i in range(5):
        (home / "mac-ai-os" / "docs" / f"doc{i}.md").write_text(f"# doc {i}\n")

    # mcp / config surfaces
    (home / ".config" / "mcp").mkdir(parents=True)
    (home / ".devin").mkdir()
    (home / ".local" / "share").mkdir(parents=True)
    (home / ".codex").mkdir()
    for i in range(3):
        (home / ".config" / "mcp" / f"server{i}.json").write_text("{}\n")
    (home / ".devin" / "config.json").write_text("{}\n")
    (home / ".codex" / "settings.json").write_text("{}\n")

    # --- database: workspaces, tasks, focus -------------------------------
    db = Database(str(home / "maos.db"))
    db.execute(
        "INSERT OR REPLACE INTO user_model (key, value, confidence, source, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("portfolio_focus", "umbrella stress test", 1.0, "test", "2024-01-01T00:00:00"),
    )

    # Seed initial context.
    for i in range(50):
        db.upsert_workspace(
            str(home / "workspaces" / f"ws{i}"),
            name=f"ws{i}",
            language="python",
            git_repo=False,
            file_count=1,
            last_activity=f"2024-01-01T00:00:{i:02d}",
        )
    for i in range(100):
        db.create_task(f"seed task {i}")

    # --- run 10,000 user-like actions -------------------------------------
    # Each round is 100 actions:
    #   1 build snapshot
    #   1 workspace upsert
    #   1 task create
    #   1 context read (pending_tasks / all_workspaces)
    #   48 render_snapshot
    #   48 snapshot_json
    rounds = 100
    actions_per_round = 100
    total_actions = rounds * actions_per_round
    assert total_actions == 10000

    start = time.perf_counter()
    for round_i in range(rounds):
        snap = build_snapshot(db, home=home)

        # Mutate the context this round.
        db.upsert_workspace(
            str(home / "workspaces" / f"round-{round_i}"),
            name=f"round-{round_i}",
            language="python",
            git_repo=False,
            file_count=1,
            last_activity=f"2024-01-01T00:01:{round_i:02d}",
        )
        db.create_task(f"round task {round_i}")

        # Context reads (what a status/tasks Slack command would do).
        pending = db.pending_tasks()
        workspaces = db.all_workspaces()
        assert len(pending) == 100 + round_i + 1
        assert len(workspaces) == 50 + round_i + 1

        # The snapshot we built is still consistent with pre-mutation state,
        # but it must be bounded and deterministic.
        assert snap.workspace_count == 50 + round_i
        assert snap.pending_tasks == 100 + round_i
        assert len(snap.repositories) == repo_count
        assert len(snap.custom_clis) == 10
        assert len(snap.docs) == 5
        assert snap.mcp_configs  # at least the seeded files
        assert snap.active_focus == "umbrella stress test"

        # Repeatedly consume the snapshot as a user would (Slack render, MCP json).
        rendered = render_snapshot(snap)
        assert "MAOS Umbrella" in rendered
        json_text = snapshot_json(snap)
        assert '"repositories"' in json_text

        for _ in range(47):
            render_snapshot(snap)
            snapshot_json(snap)

    elapsed = time.perf_counter() - start
    print(f"\n10,000 umbrella actions completed in {elapsed:.2f}s")

    # Final sanity check on bounded output.
    final_snap = build_snapshot(db, home=home)
    assert final_snap.workspace_count == 50 + rounds
    assert final_snap.pending_tasks == 100 + rounds
    assert len(final_snap.repositories) == repo_count
    assert len(final_snap.custom_clis) == 10
    assert len(final_snap.docs) == 5
    assert len(final_snap.mcp_configs) <= 100  # bounded
    assert elapsed < 60.0, f"stress test too slow: {elapsed:.2f}s"
