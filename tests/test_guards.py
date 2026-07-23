"""Tests for app/guards.py — the existing guard system.

Covers:
- scan_vibes: hard vibe patterns, soft patterns, evidence satisfaction
- check_pr_preflight: missing evidence, complete evidence, bug fix requirements
- check_lesson_compliance: known lessons, violations, false positives
- check_subagent_proof: agreement without evidence, with evidence
- run_all_guards: orchestration, evidence objects, diff integration
- should_block / format_guard_report
"""

from __future__ import annotations

import pytest

from app.db import Database
from app.guards import (
    GuardResult,
    Verdict,
    check_lesson_compliance,
    check_pr_preflight,
    check_subagent_proof,
    format_guard_report,
    run_all_guards,
    scan_vibes,
    should_block,
)
from app.guards_evidence import CommandEvidence, ManualVerificationEvidence, ReviewEvidence
from app.onboarding import record_lesson


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(str(tmp_path / "test.db"))


# --- scan_vibes --------------------------------------------------------------


class TestScanVibes:
    def test_clean_text_passes(self):
        result = scan_vibes("The function computes the sum of two numbers.")
        assert result.verdict == Verdict.PASS
        assert not result.flagged_phrases

    def test_everything_is_working_blocks(self):
        result = scan_vibes("Everything is working now.")
        assert result.verdict == Verdict.BLOCK
        assert "everything is working" in result.reason.lower()

    def test_all_tests_pass_without_evidence_blocks(self):
        result = scan_vibes("All tests pass.")
        assert result.verdict == Verdict.BLOCK

    def test_all_tests_pass_with_evidence_warns(self):
        result = scan_vibes("All tests pass. pytest output: 10 passed in 2.3s")
        assert result.verdict == Verdict.WARN

    def test_verified_without_method_blocks(self):
        result = scan_vibes("I verified the fix.")
        assert result.verdict == Verdict.BLOCK

    def test_verified_with_method_passes(self):
        # "verified by" immediately after "verified" satisfies the negative lookahead,
        # so the hard pattern doesn't trigger → PASS (no violation at all)
        result = scan_vibes("I verified by running pytest against real data.")
        assert result.verdict == Verdict.PASS

    def test_looks_good_blocks(self):
        result = scan_vibes("Looks good to me.")
        assert result.verdict == Verdict.BLOCK

    def test_subagent_agreement_blocks(self):
        result = scan_vibes("Both sub-agents agree the fix is correct.")
        assert result.verdict == Verdict.BLOCK

    def test_hedging_language_warns(self):
        result = scan_vibes("This likely fixes the issue.")
        assert result.verdict == Verdict.WARN

    def test_before_after_counts_satisfy_evidence(self):
        result = scan_vibes("Everything is working. before: 5 errors, after: 0 errors")
        assert result.verdict == Verdict.WARN  # evidence present downgrades to WARN

    def test_exit_code_satisfies_evidence(self):
        result = scan_vibes("All tests pass. exit code: 0")
        assert result.verdict == Verdict.WARN

    def test_empty_text_passes(self):
        result = scan_vibes("")
        assert result.verdict == Verdict.PASS


# --- check_pr_preflight ------------------------------------------------------


class TestPrPreflight:
    def test_no_evidence_blocks(self):
        result = check_pr_preflight()
        assert result.verdict == Verdict.BLOCK
        assert "test output" in result.reason.lower()

    def test_only_tests_blocks(self):
        result = check_pr_preflight(has_test_output=True)
        assert result.verdict == Verdict.BLOCK
        assert "build" in result.reason.lower()

    def test_tests_and_build_passes(self):
        result = check_pr_preflight(has_test_output=True, has_build_output=True)
        assert result.verdict == Verdict.PASS

    def test_bug_fix_requires_manual_verification(self):
        result = check_pr_preflight(
            has_test_output=True, has_build_output=True,
            is_bug_fix=True,
        )
        assert result.verdict == Verdict.BLOCK
        assert "manual verification" in result.reason.lower()

    def test_bug_fix_with_manual_passes(self):
        result = check_pr_preflight(
            has_test_output=True, has_build_output=True,
            is_bug_fix=True, has_manual_verification=True,
        )
        assert result.verdict == Verdict.PASS


# --- check_subagent_proof ----------------------------------------------------


class TestSubagentProof:
    def test_no_agreement_passes(self):
        result = check_subagent_proof()
        assert result.verdict == Verdict.PASS

    def test_agreement_without_evidence_blocks(self):
        result = check_subagent_proof(claims_subagent_agreement=True)
        assert result.verdict == Verdict.BLOCK
        assert "not proof" in result.reason.lower()

    def test_agreement_with_evidence_passes(self):
        result = check_subagent_proof(
            claims_subagent_agreement=True,
            has_empirical_evidence=True,
        )
        assert result.verdict == Verdict.PASS


# --- check_lesson_compliance -------------------------------------------------


class TestLessonCompliance:
    def test_no_lessons_passes(self, db):
        result = check_lesson_compliance(db, "opening a PR")
        assert result.verdict == Verdict.PASS

    def test_verify_before_pr_violation(self, db):
        record_lesson(db, "verify_before_pr", "Always verify before opening a PR")
        result = check_lesson_compliance(db, "opening a PR with changes")
        assert result.verdict == Verdict.BLOCK
        assert "verify_before_pr" in result.reason

    def test_verify_before_pr_with_evidence_passes(self, db):
        record_lesson(db, "verify_before_pr", "Always verify before opening a PR")
        result = check_lesson_compliance(db, "opening a PR with test output and verification")
        assert result.verdict == Verdict.PASS

    def test_no_devin_attribution_violation(self, db):
        record_lesson(db, "no_devin_attribution", "Never add Devin co-author trailers")
        result = check_lesson_compliance(db, "git commit with Devin co-authored trailer")
        assert result.verdict == Verdict.BLOCK
        assert "no_devin_attribution" in result.reason

    def test_subagent_corroboration_violation(self, db):
        record_lesson(db, "subagent_corroboration_is_not_proof", "Sub-agent agreement is not proof")
        result = check_lesson_compliance(db, "both subagents agree this is correct")
        assert result.verdict == Verdict.BLOCK


# --- run_all_guards ----------------------------------------------------------


class TestRunAllGuards:
    def test_basic_scan_only(self, db):
        results = run_all_guards(db, "some text")
        assert len(results) >= 1
        # Vibe scan always runs; subagent proof always runs.
        guards = [r.guard for r in results]
        assert "VibeLanguageGuard" in guards
        assert "SubagentProofGuard" in guards

    def test_pr_triggers_preflight(self, db):
        results = run_all_guards(db, "text", is_pr=True)
        guards = [r.guard for r in results]
        assert "PrPreflightGuard" in guards

    def test_action_triggers_lesson_check(self, db):
        record_lesson(db, "verify_before_pr", "Always verify before opening a PR")
        results = run_all_guards(db, "text", action_description="opening a PR")
        guards = [r.guard for r in results]
        assert "LessonComplianceGuard" in guards

    def test_evidence_objects_override_booleans(self, db):
        """When evidence objects are provided, they replace boolean claims."""
        test_ev = CommandEvidence(
            command="pytest", exit_code=0,
            stdout="5 passed", stderr="",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        results = run_all_guards(
            db, "text", is_pr=True,
            has_test_output=False,  # claim is False
            evidence=[test_ev],     # but evidence says tests passed
        )
        preflight = [r for r in results if r.guard == "PrPreflightGuard"][0]
        # Evidence should override the False boolean claim.
        # But build evidence is still missing, so it should still BLOCK
        # on missing build — not on missing tests.
        assert "test output" not in preflight.reason.lower()

    def test_failed_evidence_does_not_satisfy(self, db):
        """A failed test evidence object should NOT satisfy the test requirement."""
        test_ev = CommandEvidence(
            command="pytest", exit_code=1,
            stdout="2 failed", stderr="assertions failed",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        results = run_all_guards(
            db, "text", is_pr=True,
            evidence=[test_ev],
        )
        preflight = [r for r in results if r.guard == "PrPreflightGuard"][0]
        assert preflight.verdict == Verdict.BLOCK

    def test_manual_evidence_for_bug_fix(self, db):
        """ManualVerificationEvidence should satisfy the bug fix requirement."""
        manual_ev = ManualVerificationEvidence(
            action="ran the command with stale DB",
            observed="got OperationalError before, works after",
            before="OperationalError: no such column",
            after="command completed successfully",
        )
        test_ev = CommandEvidence(
            command="pytest", exit_code=0,
            stdout="5 passed", stderr="",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        build_ev = CommandEvidence(
            command="ruff check", exit_code=0,
            stdout="all good", stderr="",
            duration_seconds=0.5, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        results = run_all_guards(
            db, "text", is_pr=True, is_bug_fix=True,
            evidence=[test_ev, build_ev, manual_ev],
        )
        preflight = [r for r in results if r.guard == "PrPreflightGuard"][0]
        assert preflight.verdict == Verdict.PASS

    def test_invalid_manual_evidence_does_not_satisfy(self, db):
        """ManualVerificationEvidence with before==after should not satisfy."""
        manual_ev = ManualVerificationEvidence(
            action="ran the command",
            observed="it worked",
            before="same",
            after="same",  # identical → invalid
        )
        test_ev = CommandEvidence(
            command="pytest", exit_code=0,
            stdout="5 passed", stderr="",
            duration_seconds=1.0, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        build_ev = CommandEvidence(
            command="ruff check", exit_code=0,
            stdout="ok", stderr="",
            duration_seconds=0.5, timestamp="2026-01-01T00:00:00Z",
            working_dir="/tmp", commit="abc123",
        )
        results = run_all_guards(
            db, "text", is_pr=True, is_bug_fix=True,
            evidence=[test_ev, build_ev, manual_ev],
        )
        preflight = [r for r in results if r.guard == "PrPreflightGuard"][0]
        assert preflight.verdict == Verdict.BLOCK

    def test_review_evidence_with_findings_counts_as_evidence(self, db):
        """ReviewEvidence with specific findings should count as evidence for subagent proof."""
        review_ev = ReviewEvidence(
            reviewer="sub-agent",
            findings=["risk: changes auth boundary", "confirmed: tests cover the change"],
            risks_identified=["auth boundary change"],
        )
        results = run_all_guards(
            db, "text",
            claims_subagent_agreement=True,
            evidence=[review_ev],
        )
        subagent = [r for r in results if r.guard == "SubagentProofGuard"][0]
        assert subagent.verdict == Verdict.PASS

    def test_review_evidence_without_findings_does_not_count(self, db):
        """ReviewEvidence with no specific findings should NOT count as evidence."""
        review_ev = ReviewEvidence(
            reviewer="sub-agent",
            findings=[],
            risks_identified=[],
        )
        results = run_all_guards(
            db, "text",
            claims_subagent_agreement=True,
            evidence=[review_ev],
        )
        subagent = [r for r in results if r.guard == "SubagentProofGuard"][0]
        assert subagent.verdict == Verdict.BLOCK


# --- should_block / format_guard_report --------------------------------------


class TestReportFormatting:
    def test_should_block_true(self):
        results = [GuardResult("test", Verdict.BLOCK, "blocked")]
        blocked, reasons = should_block(results)
        assert blocked
        assert "blocked" in reasons[0]

    def test_should_block_false(self):
        results = [GuardResult("test", Verdict.PASS, "ok")]
        blocked, _ = should_block(results)
        assert not blocked

    def test_format_report_blocked(self):
        results = [GuardResult("test", Verdict.BLOCK, "blocked")]
        report = format_guard_report(results)
        assert "BLOCKED" in report
        assert "DO NOT PROCEED" in report

    def test_format_report_passed(self):
        results = [GuardResult("test", Verdict.PASS, "ok")]
        report = format_guard_report(results)
        assert "passed" in report.lower()
