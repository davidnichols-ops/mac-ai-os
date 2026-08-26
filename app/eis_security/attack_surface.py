"""Generate a conservative attack-surface inventory from repository structure."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttackSurface:
    repository: str
    entry_points: list[str]
    trust_boundaries: list[str]
    sensitive_paths: list[str]
    deployment_paths: list[str]
    evidence: list[str]


def discover_attack_surface(repository: Path) -> AttackSurface:
    files = [p for p in repository.rglob("*") if p.is_file() and ".git" not in p.parts]
    def rel(path: Path) -> str:
        return str(path.relative_to(repository))
    entry_points = [rel(p) for p in files if p.name in {"main.py", "__main__.py", "server.py", "app.py", "index.js", "main.rs"} or "bin" in p.parts]
    trust_terms = ("auth", "permission", "policy", "trust", "signature", "manifest", "mcp", "token", "credential")
    trust_boundaries = [rel(p) for p in files if any(term in p.name.lower() for term in trust_terms)]
    sensitive_paths = [rel(p) for p in files if any(term in str(p).lower() for term in ("secret", ".env", "oauth", "private", "security"))]
    deployment_paths = [rel(p) for p in files if p.name in {"Dockerfile", "docker-compose.yml", "Procfile"} or any(part in p.parts for part in (".github", "deployment", "infrastructure", "packaging"))]
    return AttackSurface(str(repository), sorted(set(entry_points)), sorted(set(trust_boundaries)), sorted(set(sensitive_paths)), sorted(set(deployment_paths)), ["filesystem structure only; no claim of completeness"])


def write_attack_surface(surface: AttackSurface, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(surface), indent=2, sort_keys=True) + "\n")
