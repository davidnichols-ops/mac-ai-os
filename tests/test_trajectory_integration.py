"""Tests for the optional Trajectory OS context bridge.

Covers:
  - Happy path: markdown context returned
  - Best-effort failure: missing command, timeout, non-zero exit, empty output
  - Observability: every failure path emits a structured log event
  - status(): configuration/availability snapshot for diagnostics
"""

from __future__ import annotations

import subprocess

from app.integrations.trajectory import TrajectoryContextProvider


def test_context_provider_returns_markdown(monkeypatch):
    class Result:
        returncode = 0
        stdout = "# Trajectory OS Context — daily\n\n## Constitution"

    def fake_run(*args, **kwargs):
        assert args[0][:4] == ["trajectory", "context", "daily", "--format"]
        return Result()

    monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
    assert TrajectoryContextProvider().get() == Result.stdout


def test_context_provider_is_best_effort_on_missing_command(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("trajectory")

    monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
    assert TrajectoryContextProvider(command="missing").get() is None


def test_context_provider_passes_database_path(monkeypatch):
    seen = {}

    class Result:
        returncode = 0
        stdout = "context"

    def fake_run(args, **kwargs):
        seen["args"] = args
        return Result()

    monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
    assert TrajectoryContextProvider(db_path="/tmp/trajectory.db").get("identity") == "context"
    assert seen["args"][-2:] == ["--db-path", "/tmp/trajectory.db"]


# --- Observability: failure paths emit structured log events -----------------


class TestFailurePathLogging:
    """Every failure path must emit a log event so context-ingestion gaps
    are debuggable instead of silent."""

    def test_command_not_found_logs_debug(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("trajectory")

        monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
        from app.integrations.trajectory import log as traj_log

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            traj_log, "debug",
            lambda event, **kw: events.append((event, kw)),
        )
        monkeypatch.setattr(traj_log, "warning", lambda *a, **kw: None)
        result = TrajectoryContextProvider(command="missing").get()
        assert result is None
        assert len(events) == 1
        assert events[0][0] == "trajectory.context.unavailable"
        assert events[0][1]["reason"] == "command_not_found"
        assert events[0][1]["command"] == "missing"

    def test_timeout_logs_warning(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="trajectory", timeout=10.0)

        monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
        from app.integrations.trajectory import log as traj_log

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(traj_log, "debug", lambda *a, **kw: None)
        monkeypatch.setattr(
            traj_log, "warning",
            lambda event, **kw: events.append((event, kw)),
        )
        result = TrajectoryContextProvider(timeout_s=10.0).get()
        assert result is None
        assert len(events) == 1
        assert events[0][0] == "trajectory.context.timeout"
        assert events[0][1]["timeout_s"] == 10.0

    def test_nonzero_exit_logs_warning_with_returncode(self, monkeypatch):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "database locked"

        monkeypatch.setattr(
            "app.integrations.trajectory.subprocess.run",
            lambda *a, **kw: Result(),
        )
        from app.integrations.trajectory import log as traj_log

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(traj_log, "debug", lambda *a, **kw: None)
        monkeypatch.setattr(
            traj_log, "warning",
            lambda event, **kw: events.append((event, kw)),
        )
        result = TrajectoryContextProvider().get("identity")
        assert result is None
        assert len(events) == 1
        assert events[0][0] == "trajectory.context.nonzero_exit"
        assert events[0][1]["returncode"] == 1
        assert "database locked" in events[0][1]["stderr"]

    def test_empty_output_logs_debug(self, monkeypatch):
        class Result:
            returncode = 0
            stdout = "   \n  "
            stderr = ""

        monkeypatch.setattr(
            "app.integrations.trajectory.subprocess.run",
            lambda *a, **kw: Result(),
        )
        from app.integrations.trajectory import log as traj_log

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            traj_log, "debug",
            lambda event, **kw: events.append((event, kw)),
        )
        monkeypatch.setattr(traj_log, "warning", lambda *a, **kw: None)
        result = TrajectoryContextProvider().get()
        assert result is None
        assert len(events) == 1
        assert events[0][0] == "trajectory.context.empty_output"

    def test_os_error_logs_warning(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise PermissionError("permission denied")

        monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
        from app.integrations.trajectory import log as traj_log

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(traj_log, "debug", lambda *a, **kw: None)
        monkeypatch.setattr(
            traj_log, "warning",
            lambda event, **kw: events.append((event, kw)),
        )
        result = TrajectoryContextProvider().get()
        assert result is None
        assert len(events) == 1
        assert events[0][0] == "trajectory.context.os_error"

    def test_success_emits_no_logs(self, monkeypatch):
        class Result:
            returncode = 0
            stdout = "# Context\n\nBody"

        monkeypatch.setattr(
            "app.integrations.trajectory.subprocess.run",
            lambda *a, **kw: Result(),
        )
        from app.integrations.trajectory import log as traj_log

        debug_calls: list = []
        warn_calls: list = []
        monkeypatch.setattr(traj_log, "debug", lambda *a, **kw: debug_calls.append(a))
        monkeypatch.setattr(traj_log, "warning", lambda *a, **kw: warn_calls.append(a))
        result = TrajectoryContextProvider().get()
        assert result is not None
        assert debug_calls == []
        assert warn_calls == []


# --- status() diagnostics ---------------------------------------------------


class TestStatus:
    def test_status_reports_configuration(self):
        provider = TrajectoryContextProvider(
            command="trajectory",
            db_path="/tmp/traj.db",
            timeout_s=15.0,
        )
        s = provider.status()
        assert s["configured_command"] == "trajectory"
        assert s["db_path"] == "/tmp/traj.db"
        assert s["timeout_s"] == 15.0
        # command_on_path/available are booleans (depend on the machine).
        assert isinstance(s["command_on_path"], bool)
        assert isinstance(s["available"], bool)

    def test_status_available_matches_command_on_path(self):
        provider = TrajectoryContextProvider(command="definitely_not_a_real_cmd_xyz")
        s = provider.status()
        assert s["command_on_path"] is False
        assert s["available"] is False

    def test_status_does_not_invoke_subprocess(self, monkeypatch):
        """status() must be safe to call at any time — it must not shell out."""
        called = []

        def fake_run(*args, **kwargs):
            called.append(args)
            raise AssertionError("status() must not invoke subprocess.run")

        monkeypatch.setattr("app.integrations.trajectory.subprocess.run", fake_run)
        provider = TrajectoryContextProvider()
        provider.status()  # should not raise
        assert called == []

    def test_status_with_custom_command(self):
        provider = TrajectoryContextProvider(command="/usr/local/bin/trajectory")
        s = provider.status()
        assert s["configured_command"] == "/usr/local/bin/trajectory"

