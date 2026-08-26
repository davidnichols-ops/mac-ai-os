"""Adapter-based observe/model/validate/challenge/decide orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.engineering_intelligence import EngineeringFinding, ValidationEvidence, rank_findings


class EISAdapter(Protocol):
    name: str

    def observe(self) -> dict[str, object]: ...

    def model(self, observation: dict[str, object]) -> dict[str, object]: ...

    def validate(self, model: dict[str, object]) -> list[ValidationEvidence]: ...

    def challenge(self, model: dict[str, object], evidence: list[ValidationEvidence]) -> list[EngineeringFinding]: ...


@dataclass
class EISRun:
    adapter: str
    observation: dict[str, object]
    model: dict[str, object]
    evidence: list[ValidationEvidence] = field(default_factory=list)
    findings: list[EngineeringFinding] = field(default_factory=list)

    @property
    def decisions(self) -> list[EngineeringFinding]:
        return rank_findings(self.findings)


class EngineeringIntelligenceLoop:
    """Run a repository adapter without letting narrative substitute for evidence."""

    def run(self, adapter: EISAdapter) -> EISRun:
        observation = adapter.observe()
        model = adapter.model(observation)
        evidence = adapter.validate(model)
        findings = adapter.challenge(model, evidence)
        return EISRun(adapter.name, observation, model, evidence, findings)
