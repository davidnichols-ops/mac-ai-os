"""Offline dependency-risk inventory; vulnerability claims require advisory data."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DependencySignal:
    manifest: str
    dependency: str
    signal: str
    status: str = "missing_validation"


def audit_dependencies(repository: Path) -> list[DependencySignal]:
    signals: list[DependencySignal] = []
    for name in ("pyproject.toml", "package.json", "Cargo.toml", "go.mod", "requirements.txt", "uv.lock", "package-lock.json", "Cargo.lock", "go.sum"):
        manifest = repository / name
        if not manifest.exists():
            continue
        text = manifest.read_text(errors="ignore")
        if name in {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "requirements.txt"}:
            signals.append(DependencySignal(name, "*", "declared dependencies require advisory scan"))
        if name.endswith("lock") or name.endswith(".sum"):
            signals.append(DependencySignal(name, "*", "lock/provenance material present"))
        if re.search(r"(?i)(git\+|http://|file:|path\s*=)", text):
            signals.append(DependencySignal(name, "*", "non-registry or local dependency source requires provenance review", "suspected_weakness"))
    return signals
