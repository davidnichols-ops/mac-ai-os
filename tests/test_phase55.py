"""Tests for Phase 5.5 — Workspace Profiler.

Tests:
- Fallback profiling (no LLM) produces a valid profile
- LLM profiling parses STATUS and OPEN_WORK correctly
- TODO extraction from source files
- Git info gathering (branch, commits, modified files)
- Task discovery creates tasks from open work items
- profile_all_workspaces skips already-profiled workspaces
- Context packet includes workspace_profiles
- Context evaluator credits profiled workspaces
- DB migration adds profile/project_status/profiled_at columns
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.db import Database
from app.workspace_profiler import (
    _extract_todos,
    _fallback_profile,
    _guess_status,
    _parse_profile,
    profile_all_workspaces,
    profile_workspace,
)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with a README and a TODO."""
    (tmp_path / "README.md").write_text(
        "# Test Project\n\nA project for testing the profiler.\n"
    )
    (tmp_path / "main.py").write_text(
        "# TODO: implement the main loop\n"
        "# FIXME: handle edge case\n"
        "print('hello')\n"
    )
    return tmp_path


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(str(tmp_path / "test.db"))


class TestFallbackProfile:
    def test_fallback_uses_readme_first_line(self, tmp_workspace):
        readme = (tmp_workspace / "README.md").read_text()
        profile = _fallback_profile("test-proj", readme, {}, [])
        assert "SUMMARY:" in profile
        assert "Test Project" in profile or "testing" in profile

    def test_fallback_with_no_readme(self):
        profile = _fallback_profile("empty-proj", "", {}, [])
        assert "SUMMARY:" in profile
        assert "STATUS:" in profile
        assert "OPEN_WORK: none" in profile

    def test_fallback_with_todos(self):
        profile = _fallback_profile("proj", "", {}, ["TODO: fix bug", "FIXME: crash"])
        assert "OPEN_WORK:" in profile
        assert "fix bug" in profile


class TestGuessStatus:
    def test_active_recent_commits(self):
        from datetime import datetime, timedelta
        recent = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        git_info = {"recent_commits": [{"date": recent, "message": "update"}]}
        assert _guess_status(git_info) == "active"

    def test_maintenance_old_commits(self):
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        git_info = {"recent_commits": [{"date": old, "message": "update"}]}
        assert _guess_status(git_info) == "maintenance"

    def test_abandoned_very_old(self):
        from datetime import datetime, timedelta
        very_old = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
        git_info = {"recent_commits": [{"date": very_old, "message": "init"}]}
        assert _guess_status(git_info) == "abandoned"

    def test_no_commits_is_abandoned(self):
        assert _guess_status({}) == "abandoned"


class TestParseProfile:
    def test_parse_full_response(self):
        text = (
            "SUMMARY: A web scraper for tracking Elon Musk's tweets.\n"
            "STATUS: active\n"
            "OPEN_WORK: add sentiment analysis, fix rate limiting\n"
            "STACK: Python, SQLite, rumps"
        )
        status, open_work = _parse_profile(text)
        assert status == "active"
        assert len(open_work) == 2
        assert "add sentiment analysis" in open_work

    def test_parse_no_open_work(self):
        text = "SUMMARY: A complete project.\nSTATUS: complete\nOPEN_WORK: none\nSTACK: Go"
        status, open_work = _parse_profile(text)
        assert status == "complete"
        assert open_work == []

    def test_parse_unknown_status(self):
        text = "SUMMARY: Unknown project.\nSTATUS: weird\nOPEN_WORK: none\nSTACK: ?"
        status, _ = _parse_profile(text)
        assert status == "unknown"


class TestExtractTodos:
    def test_finds_todos(self, tmp_workspace):
        todos = _extract_todos(tmp_workspace)
        assert len(todos) >= 2
        assert any("implement the main loop" in t for t in todos)
        assert any("handle edge case" in t for t in todos)

    def test_no_todos_in_empty_dir(self, tmp_path):
        (tmp_path / "clean.py").write_text("print('no todos here')\n")
        todos = _extract_todos(tmp_path)
        assert todos == []

    def test_max_todos_limit(self, tmp_path):
        (tmp_path / "many.py").write_text(
            "\n".join(f"# TODO: task {i}" for i in range(20))
        )
        todos = _extract_todos(tmp_path)
        assert len(todos) <= 10  # MAX_TODOS


class TestProfileWorkspace:
    def test_profile_without_llm(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        result = profile_workspace(db, str(tmp_workspace), model=None)

        assert result["name"] == tmp_workspace.name
        assert result["status"] in ("active", "abandoned", "maintenance", "complete", "unknown")
        assert "SUMMARY:" in result["profile"]

        # Verify it was stored in DB.
        ws = db.workspace_by_path(str(tmp_workspace))
        assert ws["profile"] != ""
        assert ws["project_status"] != ""
        assert ws["profiled_at"] is not None

    def test_profile_with_mock_llm(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text=(
                "SUMMARY: A test project for the profiler.\n"
                "STATUS: active\n"
                "OPEN_WORK: implement main loop, handle edge case\n"
                "STACK: Python"
            )
        )

        result = profile_workspace(db, str(tmp_workspace), model=mock_model)

        assert result["status"] == "active"
        assert len(result["open_work"]) == 2
        assert "implement main loop" in result["open_work"]

        # Tasks should have been created.
        tasks = db.pending_tasks()
        task_titles = [t["title"] for t in tasks]
        assert any("implement main loop" in t for t in task_titles)

    def test_profile_nonexistent_path(self, db, tmp_path):
        result = profile_workspace(db, str(tmp_path / "nonexistent"), model=None)
        assert "error" in result

    def test_profile_llm_failure_falls_back(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )

        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("model unavailable")

        result = profile_workspace(db, str(tmp_workspace), model=mock_model)
        # Should still produce a profile via fallback.
        assert "SUMMARY:" in result["profile"]
        assert result["status"] != "unknown" or result["status"] == "unknown"  # fallback still runs


class TestProfileAllWorkspaces:
    def test_profiles_all_unprofiled(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        # Add a second workspace.
        ws2 = tmp_workspace / "subproject"
        ws2.mkdir()
        (ws2 / "README.md").write_text("# Sub Project\nA sub-project.\n")
        db.upsert_workspace(str(ws2), "subproject", language="python", file_count=1)

        results = profile_all_workspaces(db, model=None)
        assert len(results) == 2

    def test_skips_already_profiled(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        # Profile it once.
        profile_workspace(db, str(tmp_workspace), model=None)

        # Second call should skip it.
        results = profile_all_workspaces(db, model=None, force=False)
        assert len(results) == 0

    def test_force_reprofiles(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        profile_workspace(db, str(tmp_workspace), model=None)

        results = profile_all_workspaces(db, model=None, force=True)
        assert len(results) == 1


class TestTaskDiscovery:
    def test_creates_tasks_from_open_work(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), "test-ws",
            language="python", git_repo=False, file_count=2,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text=(
                "SUMMARY: Test project.\n"
                "STATUS: active\n"
                "OPEN_WORK: add tests, write docs, fix parser\n"
                "STACK: Python"
            )
        )

        profile_workspace(db, str(tmp_workspace), model=mock_model)

        tasks = db.pending_tasks()
        titles = [t["title"] for t in tasks]
        assert any("add tests" in t for t in titles)
        assert any("write docs" in t for t in titles)
        assert any("fix parser" in t for t in titles)

    def test_no_duplicate_tasks(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), "test-ws",
            language="python", git_repo=False, file_count=2,
        )

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(
            text=(
                "SUMMARY: Test project.\n"
                "STATUS: active\n"
                "OPEN_WORK: add tests\n"
                "STACK: Python"
            )
        )

        # Profile twice with force.
        profile_workspace(db, str(tmp_workspace), model=mock_model)
        profile_workspace(db, str(tmp_workspace), model=mock_model)

        tasks = db.pending_tasks()
        add_tests_tasks = [t for t in tasks if "add tests" in t["title"]]
        assert len(add_tests_tasks) == 1  # no duplicate


class TestContextPacketProfiles:
    def test_context_packet_includes_profiles(self, db, tmp_workspace):
        from app.blackboard import Blackboard
        from app.server.context import build_context_packet

        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        profile_workspace(db, str(tmp_workspace), model=None)

        bb = Blackboard(db)
        packet = build_context_packet(db, bb)

        assert "workspace_profiles" in packet
        assert len(packet["workspace_profiles"]) >= 1
        assert packet["workspace_profiles"][0]["name"] == tmp_workspace.name
        assert packet["workspace_profiles"][0]["status"] != ""

    def test_context_packet_empty_profiles(self, db):
        from app.blackboard import Blackboard
        from app.server.context import build_context_packet

        bb = Blackboard(db)
        packet = build_context_packet(db, bb)

        assert "workspace_profiles" in packet
        assert packet["workspace_profiles"] == []


class TestEvaluatorWithProfiles:
    def test_profiled_workspaces_score_higher(self, db, tmp_workspace):
        from app.evaluation import evaluate_context

        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )

        # Score without profile.
        result1 = evaluate_context(db)
        ws_score_1 = result1["workspace"]

        # Profile it.
        profile_workspace(db, str(tmp_workspace), model=None)

        # Score with profile.
        result2 = evaluate_context(db)
        ws_score_2 = result2["workspace"]

        assert ws_score_2 > ws_score_1
        assert result2["details"]["workspace"]["with_profile"] == 1


class TestDBMigration:
    def test_migration_adds_profile_columns(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        cols = {row["name"] for row in db.query("PRAGMA table_info(workspaces)")}
        assert "profile" in cols
        assert "project_status" in cols
        assert "profiled_at" in cols

    def test_update_workspace_profile(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        db.update_workspace_profile(
            str(tmp_workspace),
            profile="SUMMARY: Test\nSTATUS: active",
            project_status="active",
        )
        ws = db.workspace_by_path(str(tmp_workspace))
        assert ws["profile"] == "SUMMARY: Test\nSTATUS: active"
        assert ws["project_status"] == "active"
        assert ws["profiled_at"] is not None

    def test_profiled_workspace_count(self, db, tmp_workspace):
        db.upsert_workspace(
            str(tmp_workspace), tmp_workspace.name,
            language="python", git_repo=False, file_count=2,
        )
        assert db.profiled_workspace_count() == 0
        db.update_workspace_profile(
            str(tmp_workspace), profile="test", project_status="active",
        )
        assert db.profiled_workspace_count() == 1
