"""Small STRIDE-style threat model generator grounded in attack-surface evidence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.eis_security.attack_surface import AttackSurface


@dataclass(frozen=True)
class Threat:
    category: str
    asset: str
    threat: str
    validation: str
    status: str = "unverified_hypothesis"


def generate_threats(surface: AttackSurface) -> list[Threat]:
    threats: list[Threat] = []
    for boundary in surface.trust_boundaries:
        threats.extend([
            Threat("Spoofing", boundary, "attacker presents an untrusted identity", "exercise authentication and signature verification with invalid and replayed identities"),
            Threat("Tampering", boundary, "attacker alters policy, manifest, or serialized input", "mutate input and assert rejection plus stable audit evidence"),
            Threat("Elevation of privilege", boundary, "low-privilege input reaches a privileged action", "invoke every sensitive action with an incomplete or downgraded capability"),
        ])
    for entry in surface.entry_points:
        threats.append(Threat("Denial of service", entry, "hostile input causes unbounded work or resource exhaustion", "run bounded mutation/property tests with time and memory limits"))
    for path in surface.sensitive_paths:
        threats.append(Threat("Information disclosure", path, "credentials or sensitive data cross a logging, indexing, or response boundary", "scan fixtures and inspect redacted outputs under hostile content"))
    return threats


def write_threats(threats: list[Threat], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(threat) for threat in threats], indent=2, sort_keys=True) + "\n")
