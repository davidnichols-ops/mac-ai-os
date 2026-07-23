"""Tests for app/guards_evidence.py — evidence objects and verification execution.

Covers:
- CommandEvidence: construction, serialization, success property
- ManualVerificationEvidence: validity, before/after distinction
- ReviewEvidence: specificity check
- capture_command: actual command execution, exit codes, git state
- evidence_to_json / load_evidence: round-trip serialization
- is_stale: staleness checking against current commit
"""

from __future__ import annotations

import json
import subprocess

import pytest

from app.guards_evidence import (
    CommandEvidence,
    ManualVerificationEvidence,
    ReviewEvidence,
    capture_command,
    evidence_to_json,
    is_stale,
    load_evidence,
)

# --- CommandEvidence ---------------------------------------------------------


class TestCommandEvidence:
    def test_success_true_when_exit_zero(self):
        ev = CommandEvidence(
            command="true", exit_code=0,
            stdout="", stderr="",
            duration_seconds=0.1, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp",
        )
        assert ev.success is True

    def test_success_false_when_nonzero(self):
        ev = CommandEvidence(
            command="false", exit_code=1,
            stdout="", stderr="error",
            duration_seconds=0.1, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp",
        )
        assert ev.success is False

    def test_to_dict_round_trip(self):
        ev = CommandEvidence(
            command="pytest", exit_code=0,
            stdout="5 passed", stderr="",
            duration_seconds=2.3, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
            dirty=True, env_keys=["FOO"],
        )
        d = ev.to_dict()
        assert d["type"] == "command"
        assert d["exit_code"] == 0
        assert d["success"] is True
        ev2 = CommandEvidence.from_dict(d)
        assert ev2.command == ev.command
        assert ev2.exit_code == ev.exit_code
        assert ev2.commit == ev.commit


# --- ManualVerificationEvidence ----------------------------------------------


class TestManualVerification:
    def test_valid_with_distinct_before_after(self):
        ev = ManualVerificationEvidence(
            action="ran the command",
            observed="it worked",
            before="error occurred",
            after="no error",
        )
        assert ev.valid is True

    def test_invalid_with_identical_before_after(self):
        ev = ManualVerificationEvidence(
            action="ran the command",
            observed="it worked",
            before="same",
            after="same",
        )
        assert ev.valid is False

    def test_invalid_with_empty_fields(self):
        ev = ManualVerificationEvidence(
            action="", observed="", before="", after="",
        )
        assert ev.valid is False

    def test_to_dict_round_trip(self):
        ev = ManualVerificationEvidence(
            action="ran X", observed="Y",
            before="old", after="new",
        )
        d = ev.to_dict()
        assert d["type"] == "manual"
        ev2 = ManualVerificationEvidence.from_dict(d)
        assert ev2.action == ev.action
        assert ev2.before == ev.before


# --- ReviewEvidence ----------------------------------------------------------


class TestReviewEvidence:
    def test_specific_with_findings(self):
        ev = ReviewEvidence(
            reviewer="sub-agent",
            findings=["risk: auth boundary changed"],
            risks_identified=["auth"],
        )
        assert ev.is_specific is True

    def test_not_specific_without_findings(self):
        ev = ReviewEvidence(
            reviewer="sub-agent",
            findings=[], risks_identified=[],
        )
        assert ev.is_specific is False


# --- capture_command ---------------------------------------------------------


class TestCaptureCommand:
    def test_successful_command(self, tmp_path):
        ev = capture_command("echo hello", cwd=str(tmp_path))
        assert ev.exit_code == 0
        assert "hello" in ev.stdout
        assert ev.success is True
        assert ev.duration_seconds > 0
        assert ev.working_dir == str(tmp_path.resolve())

    def test_failing_command(self, tmp_path):
        ev = capture_command("exit 1", cwd=str(tmp_path))
        assert ev.exit_code == 1
        assert ev.success is False

    def test_command_with_stderr(self, tmp_path):
        ev = capture_command("echo error >&2", cwd=str(tmp_path))
        assert ev.exit_code == 0
        assert "error" in ev.stderr

    def test_timeout_handling(self, tmp_path):
        ev = capture_command("sleep 10", cwd=str(tmp_path), timeout=1)
        assert ev.exit_code == -1
        assert "TIMEOUT" in ev.stderr

    def test_git_state_capture(self, tmp_path):
        # Create a git repo and make a commit.
        subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path, timeout=5)
        subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path, timeout=5)
        subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path, timeout=5)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=tmp_path, timeout=5)
        subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path, timeout=5)
        # Clean working tree.
        ev = capture_command("echo test", cwd=str(tmp_path))
        assert ev.commit  # should have a commit hash
        assert ev.dirty is False

    def test_git_dirty_detection(self, tmp_path):
        subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path, timeout=5)
        subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path, timeout=5)
        subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path, timeout=5)
        (tmp_path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=tmp_path, timeout=5)
        subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path, timeout=5)
        # Make working tree dirty.
        (tmp_path / "file.txt").write_text("changed")
        ev = capture_command("echo test", cwd=str(tmp_path))
        assert ev.dirty is True

    def test_nonexistent_command(self, tmp_path):
        ev = capture_command("this_command_does_not_exist_12345", cwd=str(tmp_path))
        assert ev.exit_code != 0
        assert ev.success is False


# --- Serialization -----------------------------------------------------------


class TestSerialization:
    def test_command_evidence_json_round_trip(self):
        ev = CommandEvidence(
            command="pytest", exit_code=0,
            stdout="5 passed", stderr="",
            duration_seconds=2.3, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        json_str = evidence_to_json(ev)
        ev2 = load_evidence(json_str)
        assert isinstance(ev2, CommandEvidence)
        assert ev2.command == ev.command
        assert ev2.exit_code == ev.exit_code

    def test_manual_evidence_json_round_trip(self):
        ev = ManualVerificationEvidence(
            action="ran X", observed="Y",
            before="old", after="new",
        )
        json_str = evidence_to_json(ev)
        ev2 = load_evidence(json_str)
        assert isinstance(ev2, ManualVerificationEvidence)
        assert ev2.action == ev.action

    def test_load_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown evidence type"):
            load_evidence(json.dumps({"type": "unknown"}))


# --- is_stale ----------------------------------------------------------------


class TestStaleness:
    def test_same_commit_not_stale(self):
        ev = CommandEvidence(
            command="test", exit_code=0,
            stdout="", stderr="",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        assert is_stale(ev, "abc123") is False

    def test_different_commit_is_stale(self):
        ev = CommandEvidence(
            command="test", exit_code=0,
            stdout="", stderr="",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        assert is_stale(ev, "def456") is True

    def test_empty_commit_not_stale(self):
        ev = CommandEvidence(
            command="test", exit_code=0,
            stdout="", stderr="",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="",
        )
        assert is_stale(ev, "abc123") is False
