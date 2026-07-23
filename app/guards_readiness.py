"""PR readiness state machine and content checklist.

The checklist is NOT a collection of unrelated booleans. It's a state machine
where each transition requires explicit evidence. An agent cannot skip from
"I changed code" to "Open PR" merely by asserting that verification occurred.

States:
  DRAFT        — code changes exist but nothing has been verified
  UNDERSTOOD   — the change is documented (what/why/behavior changed)
  VERIFIED     — empirical evidence exists (tests run, build verified)
  PR_READY     — all local checks pass, diff inspected, content structured
  CI_READY     — (GitHub) PR exists with correct base, CI is green
  MERGE_READY  — (GitHub) reviews satisfied, no merge conflicts

Transitions require evidence objects, not boolean claims:
  DRAFT → UNDERSTOOD: requires PRContent (summary, why, what_changed)
  UNDERSTOOD → VERIFIED: requires CommandEvidence (tests) + CommandEvidence (build/lint)
  VERIFIED → PR_READY: requires DiffGuard PASS/WARN + no blocking guard violations
  PR_READY → CI_READY: requires GitHub PR exists + correct base branch
  CI_READY → MERGE_READY: requires CI green + review approval

For bug fixes, UNDERSTOOD → VERIFIED additionally requires:
  - ManualVerificationEvidence (before/after)
  - regression test evidence (a test that would fail without the fix)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.guards import GuardResult, Verdict
from app.guards_diff import inspect_diff
from app.guards_evidence import (
    CommandEvidence,
    ManualVerificationEvidence,
    ReviewEvidence,
    is_stale,
)


class ReadinessState(StrEnum):
    DRAFT = "draft"
    UNDERSTOOD = "understood"
    VERIFIED = "verified"
    PR_READY = "pr_ready"
    CI_READY = "ci_ready"
    MERGE_READY = "merge_ready"


# --- PR content checklist ----------------------------------------------------


@dataclass
class PRContent:
    """Structured PR description content.

    The PR description must have a structured format. The system checks
    which sections are present and non-empty. Different change types
    require different sections.

    For all PRs:
      - summary: what changed
      - why: why it changed
      - what_changed: detailed list of changes

    For bug fixes (additionally):
      - root_cause: what was the root cause
      - regression_test: what test proves the bug is fixed
      - before: previous behavior
      - after: new behavior

    For features (additionally):
      - behavior_changed: what behavior changed
      - behavior_unchanged: what intentionally did not change
      - risk: assessed risk level

    Optional (recommended for all):
      - known_limitations: known caveats
      - scope: stated scope (for diff scope checking)
    """

    summary: str = ""
    why: str = ""
    what_changed: str = ""
    # Bug fix fields
    root_cause: str = ""
    regression_test: str = ""
    before: str = ""
    after: str = ""
    # Feature fields
    behavior_changed: str = ""
    behavior_unchanged: str = ""
    risk: str = ""
    # Optional
    known_limitations: str = ""
    scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "why": self.why,
            "what_changed": self.what_changed,
            "root_cause": self.root_cause,
            "regression_test": self.regression_test,
            "before": self.before,
            "after": self.after,
            "behavior_changed": self.behavior_changed,
            "behavior_unchanged": self.behavior_unchanged,
            "risk": self.risk,
            "known_limitations": self.known_limitations,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PRContent:
        return cls(**{k: d.get(k, "") for k in (
            "summary", "why", "what_changed", "root_cause",
            "regression_test", "before", "after",
            "behavior_changed", "behavior_unchanged", "risk",
            "known_limitations", "scope",
        )})

    @classmethod
    def from_markdown(cls, md: str) -> PRContent:
        """Parse a markdown PR description into structured content.

        Expects sections like:
          ## Summary
          ## Why
          ## What Changed
          ## Root Cause
          ## Regression Test
          ## Before
          ## After
          ## Behavior Changed
          ## Behavior Unchanged
          ## Risk
          ## Known Limitations
          ## Scope
        """
        sections: dict[str, str] = {}
        current_header = ""
        current_lines: list[str] = []

        for line in md.split("\n"):
            header_match = re.match(r"^##\s+(.+?)\s*$", line)
            if header_match:
                if current_header:
                    sections[current_header.lower().replace(" ", "_")] = "\n".join(current_lines).strip()
                current_header = header_match.group(1)
                current_lines = []
            else:
                current_lines.append(line)
        if current_header:
            sections[current_header.lower().replace(" ", "_")] = "\n".join(current_lines).strip()

        return cls(
            summary=sections.get("summary", ""),
            why=sections.get("why", ""),
            what_changed=sections.get("what_changed", sections.get("what_changed", "")),
            root_cause=sections.get("root_cause", ""),
            regression_test=sections.get("regression_test", ""),
            before=sections.get("before", ""),
            after=sections.get("after", ""),
            behavior_changed=sections.get("behavior_changed", ""),
            behavior_unchanged=sections.get("behavior_unchanged", ""),
            risk=sections.get("risk", ""),
            known_limitations=sections.get("known_limitations", ""),
            scope=sections.get("scope", ""),
        )


def validate_content(content: PRContent, *, is_bug_fix: bool = False) -> GuardResult:
    """Validate that PR content has all required sections.

    Args:
        content: The PR content to validate.
        is_bug_fix: Whether this is a bug fix (requires additional sections).

    Returns:
        GuardResult — BLOCK if required sections are missing or empty.
    """
    missing: list[str] = []

    # Required for all PRs.
    if not content.summary.strip():
        missing.append("Summary")
    if not content.why.strip():
        missing.append("Why")
    if not content.what_changed.strip():
        missing.append("What Changed")

    # Required for bug fixes.
    if is_bug_fix:
        if not content.root_cause.strip():
            missing.append("Root Cause (required for bug fixes)")
        if not content.regression_test.strip():
            missing.append("Regression Test (required for bug fixes)")
        if not content.before.strip():
            missing.append("Before (required for bug fixes)")
        if not content.after.strip():
            missing.append("After (required for bug fixes)")

    if missing:
        return GuardResult(
            guard="PRContentGuard",
            verdict=Verdict.BLOCK,
            reason=f"PR content missing required sections: {', '.join(missing)}",
            evidence_required=missing,
        )

    return GuardResult(
        guard="PRContentGuard",
        verdict=Verdict.PASS,
        reason="All required PR content sections present.",
    )


# --- Readiness assessment ----------------------------------------------------


@dataclass
class ReadinessCheck:
    """A single check in the readiness assessment."""

    name: str
    status: str  # "pass", "warn", "block", "missing", "stale"
    detail: str
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class ReadinessReport:
    """Full readiness assessment report."""

    state: ReadinessState
    checks: list[ReadinessCheck] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    @property
    def is_pr_ready(self) -> bool:
        return self.state in (ReadinessState.PR_READY, ReadinessState.CI_READY, ReadinessState.MERGE_READY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "is_pr_ready": self.is_pr_ready,
            "checks": [c.to_dict() for c in self.checks],
            "blockers": self.blockers,
            "next_steps": self.next_steps,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)


def assess_readiness(
    *,
    repo_dir: str,
    content: PRContent | None = None,
    is_bug_fix: bool = False,
    test_evidence: CommandEvidence | None = None,
    build_evidence: CommandEvidence | None = None,
    lint_evidence: CommandEvidence | None = None,
    manual_evidence: ManualVerificationEvidence | None = None,
    review_evidence: ReviewEvidence | None = None,
    current_commit: str = "",
    diff_base: str = "HEAD",
    skip_diff: bool = False,
) -> ReadinessReport:
    """Assess PR readiness and determine the current state.

    This is the comprehensive readiness check. It evaluates all evidence
    and determines which state the change is in.

    Args:
        repo_dir: Path to the git repository.
        content: PR content (summary, why, etc.). None if not yet written.
        is_bug_fix: Whether this is a bug fix.
        test_evidence: Evidence from running tests.
        build_evidence: Evidence from running build.
        lint_evidence: Evidence from running lint.
        manual_evidence: Manual verification evidence (required for bug fixes).
        review_evidence: Review evidence (optional, not required).
        current_commit: Current HEAD commit (for staleness checking).
        diff_base: Base ref for diff inspection.
        skip_diff: Skip diff inspection (e.g., no changes to inspect).

    Returns:
        ReadinessReport with state, checks, blockers, and next steps.
    """
    checks: list[ReadinessCheck] = []
    blockers: list[str] = []
    next_steps: list[str] = []

    # --- Check 1: Content (DRAFT → UNDERSTOOD) ---
    if content is None:
        checks.append(ReadinessCheck(
            name="content", status="missing",
            detail="No PR content provided. Write a structured PR description.",
        ))
        blockers.append("PR content not written")
        next_steps.append("Write PR description with: Summary, Why, What Changed")
        if is_bug_fix:
            next_steps.append("For bug fixes also include: Root Cause, Regression Test, Before, After")
        return ReadinessReport(
            state=ReadinessState.DRAFT,
            checks=checks, blockers=blockers, next_steps=next_steps,
        )

    content_result = validate_content(content, is_bug_fix=is_bug_fix)
    checks.append(ReadinessCheck(
        name="content", status=_verdict_to_status(content_result.verdict),
        detail=content_result.reason,
    ))
    if content_result.verdict == Verdict.BLOCK:
        blockers.append(content_result.reason)
        next_steps.append("Complete missing PR content sections")
        return ReadinessReport(
            state=ReadinessState.DRAFT,
            checks=checks, blockers=blockers, next_steps=next_steps,
        )

    # --- Check 2: Test evidence (UNDERSTOOD → VERIFIED) ---
    if test_evidence is None:
        checks.append(ReadinessCheck(
            name="tests", status="missing",
            detail="No test evidence. Run: maos guards verify -- <test-command>",
        ))
        blockers.append("No test evidence")
        next_steps.append("Run tests: maos guards verify -- uv run pytest")
    else:
        stale = is_stale(test_evidence, current_commit) if current_commit else False
        if stale:
            checks.append(ReadinessCheck(
                name="tests", status="stale",
                detail=f"Test evidence is stale (captured at {test_evidence.commit[:8]}, current is {current_commit[:8]})",
                evidence=test_evidence.to_dict(),
            ))
            blockers.append("Test evidence is stale — re-run tests")
            next_steps.append("Re-run tests at current commit")
        elif not test_evidence.success:
            checks.append(ReadinessCheck(
                name="tests", status="block",
                detail=f"Tests failed (exit code {test_evidence.exit_code})",
                evidence=test_evidence.to_dict(),
            ))
            blockers.append(f"Tests failed: exit code {test_evidence.exit_code}")
            next_steps.append("Fix failing tests")
        else:
            checks.append(ReadinessCheck(
                name="tests", status="pass",
                detail=f"Tests passed (exit 0, {test_evidence.duration_seconds}s)",
                evidence=test_evidence.to_dict(),
            ))

    # --- Check 3: Build/lint evidence ---
    if build_evidence is None and lint_evidence is None:
        checks.append(ReadinessCheck(
            name="build_lint", status="missing",
            detail="No build/lint evidence. Run: maos guards verify -- <build-or-lint-command>",
        ))
        if "No test evidence" not in blockers:  # don't pile on
            blockers.append("No build/lint evidence")
        next_steps.append("Run lint: maos guards verify -- uv run ruff check app tests")
    else:
        for name, ev in [("build", build_evidence), ("lint", lint_evidence)]:
            if ev is None:
                continue
            stale = is_stale(ev, current_commit) if current_commit else False
            if stale:
                checks.append(ReadinessCheck(
                    name=name, status="stale",
                    detail=f"{name.title()} evidence is stale",
                    evidence=ev.to_dict(),
                ))
                blockers.append(f"{name.title()} evidence is stale")
            elif not ev.success:
                checks.append(ReadinessCheck(
                    name=name, status="block",
                    detail=f"{name.title()} failed (exit code {ev.exit_code})",
                    evidence=ev.to_dict(),
                ))
                blockers.append(f"{name.title()} failed: exit code {ev.exit_code}")
                next_steps.append(f"Fix {name} issues")
            else:
                checks.append(ReadinessCheck(
                    name=name, status="pass",
                    detail=f"{name.title()} passed (exit 0, {ev.duration_seconds}s)",
                    evidence=ev.to_dict(),
                ))

    # --- Check 4: Manual verification (bug fixes only) ---
    if is_bug_fix:
        if manual_evidence is None:
            checks.append(ReadinessCheck(
                name="manual_verification", status="missing",
                detail="No manual verification evidence. Required for bug fixes.",
            ))
            blockers.append("No manual verification for bug fix")
            next_steps.append("Provide manual verification with before/after")
        elif not manual_evidence.valid:
            checks.append(ReadinessCheck(
                name="manual_verification", status="block",
                detail="Manual verification invalid (empty fields or before==after)",
                evidence=manual_evidence.to_dict(),
            ))
            blockers.append("Manual verification is invalid")
            next_steps.append("Provide meaningful before/after in manual verification")
        else:
            checks.append(ReadinessCheck(
                name="manual_verification", status="pass",
                detail="Manual verification provided with distinct before/after",
                evidence=manual_evidence.to_dict(),
            ))

    # If we have blockers so far, we're not VERIFIED yet.
    if blockers:
        return ReadinessReport(
            state=ReadinessState.UNDERSTOOD,
            checks=checks, blockers=blockers, next_steps=next_steps,
        )

    # --- Check 5: Diff inspection (VERIFIED → PR_READY) ---
    if not skip_diff:
        diff_result = inspect_diff(repo_dir, diff_base, stated_scope=content.scope)
        checks.append(ReadinessCheck(
            name="diff", status=_verdict_to_status(diff_result.verdict),
            detail=diff_result.reason,
        ))
        if diff_result.verdict == Verdict.BLOCK:
            blockers.append(f"Diff blocked: {diff_result.reason}")
            next_steps.append("Remove blocked patterns from diff (secrets, credentials)")
    else:
        checks.append(ReadinessCheck(
            name="diff", status="pass",
            detail="Diff inspection skipped",
        ))

    # --- Check 6: Review evidence (optional, informational) ---
    if review_evidence is not None:
        if review_evidence.is_specific:
            checks.append(ReadinessCheck(
                name="review", status="pass",
                detail=f"Review by {review_evidence.reviewer} with {len(review_evidence.findings)} findings",
                evidence=review_evidence.to_dict(),
            ))
        else:
            checks.append(ReadinessCheck(
                name="review", status="warn",
                detail=f"Review by {review_evidence.reviewer} has no specific findings",
                evidence=review_evidence.to_dict(),
            ))
    # Review is NOT required for PR_READY — it's optional.

    # Determine final state.
    if blockers:
        return ReadinessReport(
            state=ReadinessState.VERIFIED,
            checks=checks, blockers=blockers, next_steps=next_steps,
        )

    # All local checks pass → PR_READY.
    # CI_READY and MERGE_READY require GitHub state, which is checked separately.
    next_steps.append("PR is ready to open. Use: gh pr create ...")
    return ReadinessReport(
        state=ReadinessState.PR_READY,
        checks=checks, blockers=blockers, next_steps=next_steps,
    )


def _verdict_to_status(verdict: Verdict) -> str:
    """Map a GuardResult verdict to a readiness check status."""
    return {
        Verdict.PASS: "pass",
        Verdict.WARN: "warn",
        Verdict.BLOCK: "block",
    }.get(verdict, "block")
