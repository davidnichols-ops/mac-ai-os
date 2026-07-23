"""Guards — hard filters against vibe overload and known failure patterns.

The problem: agents say "it works" without evidence, open PRs with
unverified fixes, and use sub-agent agreement as proof. This module
codifies hard rules that catch these patterns. Better a too-strong
filter that false-positives than a weak one that lets bad vibes through.

Each guard returns a GuardResult with a verdict:
  - PASS: no issues found, proceed
  - WARN: suspicious pattern detected, proceed but flag it
  - BLOCK: hard violation, refuse to proceed and escalate

Guards are intentionally aggressive. False positives are acceptable;
false negatives are not. The user explicitly prefers over-filtering.

Guard types:
  1. VibeLanguageGuard — scans text for unsupported success claims
  2. ClaimEvidenceGuard — requires evidence for "verified/works/passes" claims
  3. PrPreflightGuard — hard checks before PR creation
  4. LessonComplianceGuard — checks if an action violates recorded lessons
  5. SubagentProofGuard — flags sub-agent agreement used as sole evidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.db import Database
from app.log import get_logger
from app.onboarding import list_lessons

log = get_logger(__name__)


class Verdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardResult:
    """Result of a guard check."""
    guard: str
    verdict: Verdict
    reason: str
    evidence_required: list[str] = field(default_factory=list)
    flagged_phrases: list[str] = field(default_factory=list)
    lesson_violations: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard": self.guard,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence_required": self.evidence_required,
            "flagged_phrases": self.flagged_phrases,
            "lesson_violations": self.lesson_violations,
        }


# --- Vibe language patterns ------------------------------------------------
# These are phrases that indicate unverified confidence. When detected,
# the guard flags them. Some are hard BLOCKs (claims of verification
# with no evidence), others are WARNs (hedging language).

# Hard blocks: claiming success/verification without evidence.
HARD_VIBE_PATTERNS = [
    # Claiming verification that can't be backed by text alone
    (r"\bverified\b(?!\s+(?:by|with|against|via))", "claims 'verified' without specifying how"),
    (r"\bconfirmed\s+(?:it\s+)?works\b", "claims 'confirmed it works' without evidence"),
    (r"\ball\s+tests?\s+pass(?:ing|ed)?\b(?!\s+(?:with|showing|output))", "claims 'all tests pass' without showing output"),
    (r"\beverything\s+(?:is\s+)?(?:working|works|good|fine)\b", "'everything is working' — classic vibe overload"),
    (r"\bit\s+should\s+work\b", "'it should work' — hedging a success claim"),
    (r"\blooks\s+(?:good|fine|correct)\b", "'looks good' — visual inspection is not verification"),
    (r"\bI\s+(?:believe|think)\s+(?:it|this)\s+(?:works|is\s+(?:correct|fine))\b", "belief is not evidence"),
    (r"\bappears\s+to\s+work\b", "'appears to work' — hedging"),
    (r"\bseems\s+(?:fine|correct|to\s+work)\b", "'seems fine' — hedging"),
    (r"\bprobably\s+(?:works|fine|correct)\b", "'probably works' — hedging"),
    (r"\bno\s+issues?\s+(?:found|detected|seen)\b(?!\s+(?:by|with|via))", "claims 'no issues found' without specifying method"),
    (r"\bsub-?agents?\s+(?:both\s+|all\s+)?(?:agree|confirm|corroborate)\b", "sub-agent agreement used as evidence — this is NOT proof"),
]

# Soft warnings: hedging language that doesn't claim success but is suspicious.
SOFT_VIBE_PATTERNS = [
    (r"\blikely\b", "hedging: 'likely'"),
    (r"\bprobably\b", "hedging: 'probably'"),
    (r"\bshould\s+be\s+(?:fine|ok|correct)\b", "hedging: 'should be fine'"),
    (r"\bI\s+(?:think|believe)\b", "hedging: 'I think'"),
    (r"\bappears\s+to\b", "hedging: 'appears to'"),
    (r"\bseems\s+to\b", "hedging: 'seems to'"),
    (r"\bmy\s+understanding\s+is\b", "hedging: 'my understanding is'"),
]

# Evidence phrases that SATISFY a claim (if present, the claim is OK).
EVIDENCE_PATTERNS = [
    r"\btest\s+output\b",
    r"\bcommand\s+output\b",
    r"\bexit\s+code\s*[:=]\s*\d+",
    r"\b\d+\s+tests?\s+pass(?:ed|ing)?\b",
    r"\b\d+\s+tests?\s+fail(?:ed|ing)?\b",
    r"\bpytest\b.*\b\d+\s+pass",
    r"\bcargo\s+test\b",
    r"\brun\s+.*\boutput\b",
    r"\bverified\s+(?:by|with|against|via)\b",
    r"\bconfirmed\s+(?:by|with|via)\b",
    r"\bbefore:\s*\d+.*after:\s*\d+",  # before/after counts
    r"\bcount\s*[:=]\s*\d+",
    r"\bresult\s*[:=]",
    r"\boutput\s*[:=]",
    r"\bconsole\s+output\b",
    r"\bstdout\b",
    r"\bstderr\b",
]


def _has_evidence(text: str) -> bool:
    """Check if text contains evidence phrases that back up a claim."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in EVIDENCE_PATTERNS)


def scan_vibes(text: str) -> GuardResult:
    """Scan text for vibe language — unsupported success claims.

    This is the primary anti-vibe-overload filter. It catches:
    - "everything is working" without evidence
    - "verified" without specifying how
    - "all tests pass" without showing output
    - "looks good" as a substitute for verification
    - Sub-agent agreement used as proof

    If evidence phrases are present (test output, command output, before/after
    counts, etc.), claims are downgraded from BLOCK to PASS.

    Returns:
        GuardResult with verdict and flagged phrases.
    """
    text_lower = text.lower()
    flagged: list[str] = []
    hard_violations: list[str] = []
    evidence = _has_evidence(text)

    # Check hard patterns.
    for pattern, description in HARD_VIBE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            flagged.append(description)
            hard_violations.append(description)

    # Check soft patterns.
    soft_flagged: list[str] = []
    for pattern, description in SOFT_VIBE_PATTERNS:
        if re.search(pattern, text_lower):
            soft_flagged.append(description)
            flagged.append(description)

    # Determine verdict.
    if hard_violations and not evidence:
        return GuardResult(
            guard="VibeLanguageGuard",
            verdict=Verdict.BLOCK,
            reason=f"Hard vibe claims without evidence: {'; '.join(hard_violations)}. "
                   f"Provide test output, command output, or before/after counts.",
            flagged_phrases=flagged,
            evidence_required=["test output", "command output", "before/after counts"],
        )

    if hard_violations and evidence:
        # Claims exist but evidence is also present — warn but don't block.
        return GuardResult(
            guard="VibeLanguageGuard",
            verdict=Verdict.WARN,
            reason=f"Vibe claims detected but evidence present: {'; '.join(hard_violations)}. "
                   f"Verify the evidence actually supports the claims.",
            flagged_phrases=flagged,
        )

    if soft_flagged:
        return GuardResult(
            guard="VibeLanguageGuard",
            verdict=Verdict.WARN,
            reason=f"Hedging language detected: {'; '.join(soft_flagged)}. "
                   f"Be declarative or provide evidence.",
            flagged_phrases=flagged,
        )

    return GuardResult(
        guard="VibeLanguageGuard",
        verdict=Verdict.PASS,
        reason="No vibe language detected.",
    )


def check_pr_preflight(
    *,
    has_test_output: bool = False,
    has_build_output: bool = False,
    has_manual_verification: bool = False,
    is_bug_fix: bool = False,
    test_command: str = "",
) -> GuardResult:
    """Hard pre-flight checks before opening a PR.

    Requirements:
    - Test output MUST be provided (not just "tests pass")
    - Build MUST be verified
    - If it's a bug fix, manual verification against real data is REQUIRED

    This guard BLOCKS if any required evidence is missing. No exceptions.
    """
    missing: list[str] = []

    if not has_test_output:
        missing.append("test output (run the test suite, include the output)")
    if not has_build_output:
        missing.append("build output (run the build/linter, include the output)")
    if is_bug_fix and not has_manual_verification:
        missing.append(
            "manual verification against real data (run the actual command "
            "against the actual input, show before/after counts)"
        )

    if missing:
        return GuardResult(
            guard="PrPreflightGuard",
            verdict=Verdict.BLOCK,
            reason=f"PR pre-flight failed. Missing evidence: {'; '.join(missing)}. "
                   f"Do NOT open the PR until this evidence is provided.",
            evidence_required=missing,
        )

    return GuardResult(
        guard="PrPreflightGuard",
        verdict=Verdict.PASS,
        reason="All pre-flight checks passed.",
    )


def check_lesson_compliance(db: Database, action_description: str) -> GuardResult:
    """Check if an action description violates any recorded lesson.

    This guard reads all lessons from the user_model and checks if the
    action description contains patterns that violate them. It's intentionally
    aggressive — if there's any overlap, it flags it.
    """
    lessons = list_lessons(db)
    violations: list[str] = []

    action_lower = action_description.lower()

    for lesson in lessons:
        slug = lesson["slug"]
        text = lesson["text"].lower()

        # Extract key phrases from the lesson (nouns/verbs that indicate
        # what the lesson is about).
        if slug == "verify_before_pr":
            # Check for PR-related actions without verification
            if any(w in action_lower for w in ["pr", "pull request", "merge"]):
                if not any(w in action_lower for w in ["verify", "verified", "test", "evidence", "output", "before", "after"]):
                    violations.append(
                        "verify_before_pr: opening a PR without verification evidence "
                        "(no 'verify', 'test output', or 'before/after' in description)"
                    )

        elif slug == "subagent_corroboration_is_not_proof":
            # Check for sub-agent agreement as evidence
            if any(w in action_lower for w in ["sub-agent", "subagent", "both agree", "agents agree", "corroborat"]):
                if not any(w in action_lower for w in ["empirical", "actual", "real data", "ran", "executed", "output"]):
                    violations.append(
                        "subagent_corroboration_is_not_proof: using sub-agent agreement "
                        "as evidence without empirical verification"
                    )

        elif slug == "no_devin_attribution":
            # Check for Devin attribution in commit-related actions
            if any(w in action_lower for w in ["commit", "git push", "pr"]):
                if any(w in action_lower for w in ["devin", "co-authored", "generated with", "co_author"]):
                    violations.append(
                        "no_devin_attribution: adding Devin attribution to a commit/PR"
                    )

        else:
            # Generic check: if the action description contains key words from
            # the lesson text, flag it as a potential violation.
            # Extract significant words from the lesson.
            lesson_words = {w for w in text.split() if len(w) > 4 and w.isalpha()}
            action_words = set(action_lower.split())
            overlap = lesson_words & action_words
            # Only flag if there's significant overlap (more than 3 words).
            if len(overlap) > 3:
                violations.append(
                    f"{slug}: action description overlaps significantly with lesson "
                    f"(shared: {', '.join(list(overlap)[:5])})"
                )

    if violations:
        return GuardResult(
            guard="LessonComplianceGuard",
            verdict=Verdict.BLOCK,
            reason=f"Action may violate recorded lessons: {'; '.join(violations)}",
            lesson_violations=violations,
        )

    return GuardResult(
        guard="LessonComplianceGuard",
        verdict=Verdict.PASS,
        reason="No lesson violations detected.",
    )


def check_subagent_proof(
    *,
    claims_subagent_agreement: bool = False,
    has_empirical_evidence: bool = False,
) -> GuardResult:
    """Hard check: sub-agent agreement is not proof.

    If the agent claims sub-agent agreement as evidence AND has no empirical
    evidence (actual command output, test results, etc.), this is a BLOCK.

    This was the exact failure mode in the apple-photos-export incident:
    two sub-agents agreed on a diagnosis, but both shared the same blind spot.
    """
    if claims_subagent_agreement and not has_empirical_evidence:
        return GuardResult(
            guard="SubagentProofGuard",
            verdict=Verdict.BLOCK,
            reason="Sub-agent agreement used as sole evidence. This is NOT proof. "
                   "Sub-agents read the same code and can share the same blind spot. "
                   "Run an empirical test against the actual system.",
            evidence_required=["empirical test output", "command output", "before/after counts"],
        )

    if claims_subagent_agreement and has_empirical_evidence:
        return GuardResult(
            guard="SubagentProofGuard",
            verdict=Verdict.PASS,
            reason="Sub-agent agreement backed by empirical evidence.",
        )

    return GuardResult(
        guard="SubagentProofGuard",
        verdict=Verdict.PASS,
        reason="No sub-agent agreement claims detected.",
    )


def run_all_guards(
    db: Database,
    text: str,
    *,
    action_description: str = "",
    is_pr: bool = False,
    is_bug_fix: bool = False,
    has_test_output: bool = False,
    has_build_output: bool = False,
    has_manual_verification: bool = False,
    claims_subagent_agreement: bool = False,
    has_empirical_evidence: bool = False,
    evidence: list | None = None,
    repo_dir: str = "",
    diff_base: str = "HEAD",
) -> list[GuardResult]:
    """Run all guards and return results.

    This is the comprehensive check. Run it before any significant action
    (opening a PR, committing a fix, declaring success).

    Args:
        db: MAOS database (for lesson compliance check).
        text: The text to scan for vibes (agent output, PR description, etc.).
        action_description: What the agent is about to do (for lesson compliance).
        is_pr: Whether this is a PR-related action (triggers pre-flight).
        is_bug_fix: Whether this is a bug fix (requires manual verification).
        has_test_output: Whether test output is provided (LEGACY: prefer evidence).
        has_build_output: Whether build output is provided (LEGACY: prefer evidence).
        has_manual_verification: Whether manual verification was done (LEGACY: prefer evidence).
        claims_subagent_agreement: Whether sub-agent agreement is claimed as evidence.
        has_empirical_evidence: Whether empirical evidence is provided.
        evidence: List of evidence objects (CommandEvidence, ManualVerificationEvidence,
                  ReviewEvidence). When provided, these SUPPLEMENT or REPLACE the
                  boolean flags — the system trusts evidence objects over boolean claims.
        repo_dir: Repository directory for diff inspection (if is_pr and repo_dir set).
        diff_base: Base ref for diff inspection.

    Returns:
        List of GuardResults. If any is BLOCK, the action should not proceed.
    """
    results: list[GuardResult] = []

    # When evidence objects are provided, derive the boolean flags from them
    # rather than trusting the agent's self-attested claims. Evidence objects
    # were produced by the system executing commands; booleans are just claims.
    ev_test = False
    ev_build = False
    ev_manual = False
    ev_empirical = False
    has_review_evidence = False
    if evidence:
        from app.guards_evidence import CommandEvidence, ManualVerificationEvidence, ReviewEvidence
        for ev in evidence:
            if isinstance(ev, CommandEvidence):
                if ev.success:
                    # Distinguish test vs build by heuristic: command contains
                    # "test"/"pytest"/"cargo test" → test evidence; otherwise build.
                    cmd_lower = ev.command.lower()
                    if any(w in cmd_lower for w in ["test", "pytest", "cargo test", "npm test"]):
                        ev_test = True
                    else:
                        ev_build = True
                    ev_empirical = True
            elif isinstance(ev, ManualVerificationEvidence):
                ev_manual = ev.valid
                ev_empirical = True
            elif isinstance(ev, ReviewEvidence):
                has_review_evidence = ev.is_specific

    # Use evidence-derived flags when evidence is provided; fall back to
    # the explicit boolean flags only when no evidence object covers that check.
    effective_has_test = ev_test if evidence else has_test_output
    effective_has_build = ev_build if evidence else has_build_output
    effective_has_manual = ev_manual if evidence else has_manual_verification
    effective_has_empirical = ev_empirical if evidence else has_empirical_evidence

    # 1. Vibe language scan.
    results.append(scan_vibes(text))

    # 2. PR pre-flight (if applicable).
    if is_pr:
        results.append(check_pr_preflight(
            has_test_output=effective_has_test,
            has_build_output=effective_has_build,
            has_manual_verification=effective_has_manual,
            is_bug_fix=is_bug_fix,
        ))

    # 3. Diff inspection (if PR and repo_dir provided).
    if is_pr and repo_dir:
        from app.guards_diff import inspect_diff
        results.append(inspect_diff(repo_dir, diff_base))

    # 4. Lesson compliance (if action description provided).
    if action_description:
        results.append(check_lesson_compliance(db, action_description))

    # 5. Sub-agent proof check.
    # If we have review evidence with specific findings, that's not "sole
    # evidence" — it's supplementary. But if the agent claims sub-agent
    # agreement AND has no empirical evidence AND no specific review, BLOCK.
    results.append(check_subagent_proof(
        claims_subagent_agreement=claims_subagent_agreement,
        has_empirical_evidence=effective_has_empirical or has_review_evidence,
    ))

    return results


def should_block(results: list[GuardResult]) -> tuple[bool, list[str]]:
    """Check if any guard result is a BLOCK.

    Returns:
        (should_block, list of reasons)
    """
    reasons = [r.reason for r in results if r.blocked]
    return bool(reasons), reasons


def format_guard_report(results: list[GuardResult]) -> str:
    """Format guard results as a readable report."""
    lines: list[str] = []
    blocked, reasons = should_block(results)

    if blocked:
        lines.append("=== GUARD REPORT: BLOCKED ===")
        for r in results:
            if r.blocked:
                lines.append(f"  [{r.guard}] BLOCK: {r.reason}")
                if r.evidence_required:
                    lines.append(f"    Evidence required: {', '.join(r.evidence_required)}")
                if r.flagged_phrases:
                    lines.append(f"    Flagged: {', '.join(r.flagged_phrases)}")
                if r.lesson_violations:
                    lines.append(f"    Lesson violations: {', '.join(r.lesson_violations)}")
        lines.append("")
        lines.append("DO NOT PROCEED. Fix the issues above and re-run guards.")
    else:
        warnings = [r for r in results if r.verdict == Verdict.WARN]
        passes = [r for r in results if r.passed]
        lines.append(f"=== GUARD REPORT: {len(passes)} passed, {len(warnings)} warnings ===")
        for r in warnings:
            lines.append(f"  [{r.guard}] WARN: {r.reason}")
            if r.flagged_phrases:
                lines.append(f"    Flagged: {', '.join(r.flagged_phrases)}")
        if not warnings:
            lines.append("  All guards passed clean.")

    return "\n".join(lines)
