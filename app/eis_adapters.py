"""Repository adapter registry for weaving existing ecosystem capabilities into EIS."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.eis_contracts import Challenge, Claim
from app.eis_security.attack_surface import AttackSurface, discover_attack_surface
from app.eis_security.dependency_audit import DependencySignal, audit_dependencies
from app.eis_security.source_analyzer import SourceSignal, analyze_source
from app.eis_security.threat_model import Threat, generate_threats


@dataclass(frozen=True)
class AdapterSnapshot:
    name: str
    repository: str
    attack_surface: AttackSurface
    source_signals: list[SourceSignal]
    dependency_signals: list[DependencySignal]
    threats: list[Threat]
    claims: list[Claim]
    challenges: list[Challenge]


@dataclass(frozen=True)
class RepositoryAdapter:
    name: str
    capability: str
    analyzer: Callable[[Path], AdapterSnapshot]


def analyze_repository(path: Path, name: str | None = None) -> AdapterSnapshot:
    repository_name = name or path.name
    surface = discover_attack_surface(path)
    source_signals = analyze_source(path)
    dependency_signals = audit_dependencies(path)
    threats = generate_threats(surface)
    claim = Claim(
        claim_id=f"{repository_name}:surface",
        subject=repository_name,
        statement="The bounded EIS source inventory identifies the repository's high-signal attack surface.",
        evidence_refs=["attack_surface.py", "source_analyzer.py", "dependency_audit.py"],
    )
    challenge = Challenge(
        challenge_id=f"{repository_name}:coverage",
        claim_id=claim.claim_id,
        hypothesis="Important runtime or indirect attack surfaces are absent from static path heuristics.",
        test="Run repository-native tests, dependency advisories, and targeted runtime probes.",
        expected_observation="Coverage report identifies analyzed and skipped boundaries.",
        scope=repository_name,
    )
    return AdapterSnapshot(repository_name, str(path), surface, source_signals, dependency_signals, threats, [claim], [challenge])


def default_adapters() -> list[RepositoryAdapter]:
    return [
        RepositoryAdapter("trustcard", "MCP provenance and authorization", analyze_repository),
        RepositoryAdapter("maos", "agent execution and permission boundaries", analyze_repository),
        RepositoryAdapter("aafp", "protocol, identity, and replay validation", analyze_repository),
        RepositoryAdapter("infernet", "distributed trust and execution", analyze_repository),
        RepositoryAdapter("cvconform", "ML differential correctness", analyze_repository),
    ]
