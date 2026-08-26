"""Shared evidence and decision primitives for ecosystem validation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationEvidence:
    """An auditable result from a validation command or adapter."""

    subject: str
    commit: str
    command: str
    verdict: str
    exit_code: int
    duration_seconds: float
    environment: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.verdict not in {"passed", "failed", "skipped", "error"}:
            raise ValueError("verdict must be passed, failed, skipped, or error")
        if self.exit_code < 0:
            raise ValueError("exit_code cannot be negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        """Write one evidence record as stable, machine-readable JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class EngineeringFinding:
    """A normalized risk/opportunity finding with transparent prioritization."""

    finding_id: str
    title: str
    severity: int
    probability: int
    leverage: int
    confidence: int
    implementation_cost: int
    affected_systems: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("severity", "probability", "leverage", "confidence", "implementation_cost"):
            value = getattr(self, name)
            if not 1 <= value <= 5:
                raise ValueError(f"{name} must be between 1 and 5")

    @property
    def priority(self) -> float:
        """Rank impact × probability × leverage, adjusted by confidence and cost."""
        return round((self.severity * self.probability * self.leverage * self.confidence) / self.implementation_cost, 2)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "priority": self.priority}


def rank_findings(findings: list[EngineeringFinding]) -> list[EngineeringFinding]:
    """Return findings from highest to lowest expected engineering leverage."""
    return sorted(findings, key=lambda finding: finding.priority, reverse=True)
