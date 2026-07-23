"""Tests for app/guards_readiness.py — PR readiness state machine.

Covers:
- PRContent: from_markdown parsing, required sections
- validate_content: missing sections, bug fix requirements
- assess_readiness: state transitions, evidence gating, staleness
- ReadinessReport: state determination, blockers, next steps
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.guards_evidence import CommandEvidence, ManualVerificationEvidence, ReviewEvidence
from app.guards_readiness import (
    PRContent,
    ReadinessState,
    assess_readiness,
    validate_content,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a git repo with an initial commit."""
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path, timeout=5)
    subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path, timeout=5)
    subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path, timeout=5)
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], capture_output=True, cwd=tmp_path, timeout=5)
    subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path, timeout=5)
    return tmp_path


def _make_evidence(command: str = "pytest", exit_code: int = 0) -> CommandEvidence:
    return CommandEvidence(
        command=command, exit_code=exit_code,
        stdout="5 passed" if exit_code == 0 else "1 failed",
        stderr="" if exit_code == 0 else "error",
        duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
        working_dir="/tmp", commit="abc123",
    )


def _make_content(**kwargs) -> PRContent:
    defaults = {
        "summary": "Fix the bug",
        "why": "It was causing errors",
        "what_changed": "Changed function X",
    }
    defaults.update(kwargs)
    return PRContent(**defaults)


# --- PRContent ---------------------------------------------------------------


class TestPRContent:
    def test_from_markdown_parses_sections(self):
        md = """## Summary
Fix the bug

## Why
It was causing errors

## What Changed
Changed function X

## Root Cause
Off-by-one error

## Before
Returned 0

## After
Returns 1
"""
        content = PRContent.from_markdown(md)
        assert content.summary == "Fix the bug"
        assert content.why == "It was causing errors"
        assert content.what_changed == "Changed function X"
        assert content.root_cause == "Off-by-one error"
        assert content.before == "Returned 0"
        assert content.after == "Returns 1"

    def test_from_markdown_empty_sections(self):
        md = "## Summary\n\n## Why\n\n"
        content = PRContent.from_markdown(md)
        assert content.summary == ""
        assert content.why == ""

    def test_to_dict_round_trip(self):
        content = _make_content()
        d = content.to_dict()
        content2 = PRContent.from_dict(d)
        assert content2.summary == content.summary


# --- validate_content --------------------------------------------------------


class TestValidateContent:
    def test_complete_content_passes(self):
        content = _make_content()
        result = validate_content(content)
        assert result.verdict.value == "pass"

    def test_missing_summary_blocks(self):
        content = _make_content(summary="")
        result = validate_content(content)
        assert result.verdict.value == "block"
        assert "Summary" in result.reason

    def test_missing_why_blocks(self):
        content = _make_content(why="")
        result = validate_content(content)
        assert result.verdict.value == "block"
        assert "Why" in result.reason

    def test_bug_fix_requires_root_cause(self):
        content = _make_content()
        result = validate_content(content, is_bug_fix=True)
        assert result.verdict.value == "block"
        assert "Root Cause" in result.reason

    def test_bug_fix_requires_before_after(self):
        content = _make_content(root_cause="off-by-one")
        result = validate_content(content, is_bug_fix=True)
        assert result.verdict.value == "block"
        assert "Before" in result.reason or "After" in result.reason

    def test_complete_bug_fix_passes(self):
        content = _make_content(
            root_cause="off-by-one",
            regression_test="test_off_by_one",
            before="returned 0",
            after="returns 1",
        )
        result = validate_content(content, is_bug_fix=True)
        assert result.verdict.value == "pass"


# --- assess_readiness --------------------------------------------------------


class TestAssessReadiness:
    def test_no_content_is_draft(self, git_repo):
        report = assess_readiness(repo_dir=str(git_repo), content=None)
        assert report.state == ReadinessState.DRAFT
        assert "content not written" in " ".join(report.blockers).lower()

    def test_incomplete_content_is_draft(self, git_repo):
        content = PRContent(summary="", why="", what_changed="")
        report = assess_readiness(repo_dir=str(git_repo), content=content)
        assert report.state == ReadinessState.DRAFT

    def test_content_but_no_evidence_is_understood(self, git_repo):
        content = _make_content()
        report = assess_readiness(repo_dir=str(git_repo), content=content)
        assert report.state == ReadinessState.UNDERSTOOD
        assert any("test" in b.lower() for b in report.blockers)

    def test_content_and_tests_but_no_build_is_understood(self, git_repo):
        content = _make_content()
        test_ev = _make_evidence("pytest", 0)
        report = assess_readiness(
            repo_dir=str(git_repo), content=content,
            test_evidence=test_ev,
        )
        assert report.state == ReadinessState.UNDERSTOOD
        assert any("build" in b.lower() or "lint" in b.lower() for b in report.blockers)

    def test_failed_tests_block(self, git_repo):
        content = _make_content()
        test_ev = _make_evidence("pytest", 1)  # failed
        build_ev = _make_evidence("ruff check", 0)
        report = assess_readiness(
            repo_dir=str(git_repo), content=content,
            test_evidence=test_ev, build_evidence=build_ev,
        )
        assert report.state == ReadinessState.UNDERSTOOD
        assert any("failed" in b.lower() for b in report.blockers)

    def test_stale_evidence_blocks(self, git_repo):
        content = _make_content()
        test_ev = _make_evidence("pytest", 0)
        test_ev.commit = "oldcommit123"
        build_ev = _make_evidence("ruff check", 0)
        build_ev.commit = "oldcommit123"
        report = assess_readiness(
            repo_dir=str(git_repo), content=content,
            test_evidence=test_ev, build_evidence=build_ev,
            current_commit="newcommit456",
        )
        assert report.state == ReadinessState.UNDERSTOOD
        assert any("stale" in b.lower() for b in report.blockers)

    def test_bug_fix_without_manual_verification_blocks(self, git_repo):
        content = _make_content(
            root_cause="off-by-one",
            regression_test="test_fix",
            before="old",
            after="new",
        )
        test_ev = _make_evidence("pytest", 0)
        build_ev = _make_evidence("ruff check", 0)
        report = assess_readiness(
            repo_dir=str(git_repo), content=content, is_bug_fix=True,
            test_evidence=test_ev, build_evidence=build_ev,
        )
        assert report.state == ReadinessState.UNDERSTOOD
        assert any("manual" in b.lower() for b in report.blockers)

    def test_bug_fix_with_manual_verification_advances(self, git_repo):
        content = _make_content(
            root_cause="off-by-one",
            regression_test="test_fix",
            before="old",
            after="new",
        )
        test_ev = _make_evidence("pytest", 0)
        build_ev = _make_evidence("ruff check", 0)
        manual_ev = ManualVerificationEvidence(
            action="ran the command",
            observed="it worked",
            before="error",
            after="no error",
        )
        report = assess_readiness(
            repo_dir=str(git_repo), content=content, is_bug_fix=True,
            test_evidence=test_ev, build_evidence=build_ev,
            manual_evidence=manual_ev,
            current_commit="abc123",
        )
        # Should be VERIFIED or PR_READY (depending on diff)
        assert report.state in (ReadinessState.VERIFIED, ReadinessState.PR_READY)

    def test_all_evidence_no_diff_changes_is_pr_ready(self, git_repo):
        content = _make_content()
        test_ev = _make_evidence("pytest", 0)
        build_ev = _make_evidence("ruff check", 0)
        report = assess_readiness(
            repo_dir=str(git_repo), content=content,
            test_evidence=test_ev, build_evidence=build_ev,
            current_commit="abc123",
        )
        # No diff changes → diff passes → PR_READY
        assert report.state == ReadinessState.PR_READY
        assert report.is_pr_ready is True

    def test_review_evidence_informational(self, git_repo):
        content = _make_content()
        test_ev = _make_evidence("pytest", 0)
        build_ev = _make_evidence("ruff check", 0)
        review_ev = ReviewEvidence(
            reviewer="sub-agent",
            findings=["risk: minor"],
            risks_identified=["minor"],
        )
        report = assess_readiness(
            repo_dir=str(git_repo), content=content,
            test_evidence=test_ev, build_evidence=build_ev,
            review_evidence=review_ev,
            current_commit="abc123",
        )
        assert report.state == ReadinessState.PR_READY
        # Review should appear in checks
        review_check = [c for c in report.checks if c.name == "review"]
        assert len(review_check) == 1
        assert review_check[0].status == "pass"


# --- ReadinessReport ---------------------------------------------------------


class TestReadinessReport:
    def test_to_dict_structure(self, git_repo):
        report = assess_readiness(repo_dir=str(git_repo), content=None)
        d = report.to_dict()
        assert "state" in d
        assert "checks" in d
        assert "blockers" in d
        assert "next_steps" in d
        assert "is_pr_ready" in d

    def test_to_json(self, git_repo):
        report = assess_readiness(repo_dir=str(git_repo), content=None)
        json_str = report.to_json()
        import json
        d = json.loads(json_str)
        assert d["state"] == "draft"
