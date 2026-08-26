"""Bounded source validation campaigns for priority EIS targets."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from app.eis_adapters import AdapterSnapshot, analyze_repository
from app.eis_security.finding_engine import Finding, FindingStatus
from app.engineering_intelligence import ValidationEvidence


@dataclass(frozen=True)
class CampaignResult:
    target: str
    snapshot: AdapterSnapshot
    evidence: list[ValidationEvidence]
    findings: list[Finding]


def run_command_evidence(target: str, repository: Path, command: list[str], timeout_seconds: int = 120) -> ValidationEvidence:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=repository, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        verdict = "passed" if result.returncode == 0 else "failed"
        exit_code = result.returncode
        stdout, stderr = result.stdout[-12_000:], result.stderr[-12_000:]
        limitation = []
    except subprocess.TimeoutExpired as exc:
        verdict, exit_code = "error", 124
        stdout = (exc.stdout or "")[-12_000:] if isinstance(exc.stdout, str) else ""
        stderr = "command timed out"
        limitation = [f"timeout after {timeout_seconds}s"]
    return ValidationEvidence(
        subject=target,
        commit="unknown",
        command=" ".join(command),
        verdict=verdict,
        exit_code=exit_code,
        duration_seconds=round(time.monotonic() - started, 3),
        artifacts=[],
        limitations=limitation,
        environment={"cwd": str(repository), "stdout_tail": stdout, "stderr_tail": stderr},
    )


def classify_snapshot(snapshot: AdapterSnapshot) -> list[Finding]:
    findings: list[Finding] = []
    if snapshot.source_signals:
        findings.append(Finding(
            finding_id=f"{snapshot.name}:source-signals",
            title="Source-level security signals require runtime confirmation",
            status=FindingStatus.SUSPECTED,
            severity=3,
            probability=3,
            leverage=4,
            confidence=2,
            cost=2,
            hypothesis="A source pattern may represent an exploitable trust-boundary weakness.",
            test="Review each signal and execute a bounded regression or negative test against the relevant path.",
            observation=f"{len(snapshot.source_signals)} conservative source signal(s) detected.",
            impact="Potential command execution, data exposure, unsafe parsing, or path-boundary failure depending on context.",
            affected_systems=[snapshot.name],
            evidence=[signal.path for signal in snapshot.source_signals[:20]],
        ))
    findings.append(Finding(
        finding_id=f"{snapshot.name}:runtime-coverage",
        title="Runtime adversarial coverage is incomplete",
        status=FindingStatus.MISSING,
        severity=4,
        probability=4,
        leverage=5,
        confidence=5,
        cost=3,
        hypothesis="Static inventory does not exercise hostile runtime behavior.",
        test="Run target-specific parser, authorization, replay, failure-injection, or differential campaigns.",
        observation="The source campaign generated hypotheses but did not prove runtime behavior.",
        impact="Production failures may remain undiscovered despite a clean static scan.",
        affected_systems=[snapshot.name],
        evidence=["SECURITY_VALIDATION_REPORT.md"],
    ))
    return findings


def run_source_campaign(target: str, repository: Path, command: list[str] | None = None) -> CampaignResult:
    snapshot = analyze_repository(repository, target)
    evidence = [] if command is None else [run_command_evidence(target, repository, command)]
    return CampaignResult(target, snapshot, evidence, classify_snapshot(snapshot))
