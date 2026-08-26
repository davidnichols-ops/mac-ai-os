"""Normalize security observations without overstating scanner output."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class FindingStatus(StrEnum):
    CONFIRMED = "confirmed_vulnerability"
    SUSPECTED = "suspected_weakness"
    MISSING = "missing_validation"
    FUTURE = "future_risk"


@dataclass(frozen=True)
class Finding:
    finding_id: str
    title: str
    status: FindingStatus
    severity: int
    probability: int
    leverage: int
    confidence: int
    cost: int
    hypothesis: str
    test: str
    observation: str
    impact: str
    affected_systems: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def priority(self) -> float:
        return round(self.severity * self.probability * self.leverage * self.confidence / self.cost, 2)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "status": self.status.value, "priority": self.priority}


def write_findings(findings: list[Finding], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([finding.to_dict() for finding in findings], indent=2, sort_keys=True) + "\n")
