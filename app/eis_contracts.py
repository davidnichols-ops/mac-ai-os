"""Canonical EIS contracts shared by repository adapters and research campaigns."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ClaimStatus(StrEnum):
    OBSERVED = "observed"
    SUPPORTED = "supported"
    CHALLENGED = "challenged"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    subject: str
    statement: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    evidence_refs: list[str] = field(default_factory=list)
    challenge_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status.value}


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    claim_id: str
    hypothesis: str
    test: str
    expected_observation: str
    scope: str
    status: str = "planned"
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    decision_id: str
    title: str
    action: str
    rationale: str
    priority: float
    affected_systems: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
