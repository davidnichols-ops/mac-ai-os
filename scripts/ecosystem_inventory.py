#!/usr/bin/env python3
"""Deterministic, local-first ecosystem inventory and engineering intelligence seed."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ROOTS = [
    Path.home() / "mac-ai-os",
    Path.home() / "Projects",
]
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "target", "dist", "build", "__pycache__", ".pytest_cache"}
MANIFESTS = {"pyproject.toml", "Cargo.toml", "package.json", "go.mod", "Package.swift", "Dockerfile", "docker-compose.yml"}

@dataclass
class Repository:
    id: str
    path: str
    remote: str | None
    branch: str | None
    last_commit: str | None
    file_count: int
    languages: dict[str, int]
    manifests: list[str]
    test_paths: list[str]
    documentation: list[str]
    ci_paths: list[str]
    deployment_paths: list[str]
    research_paths: list[str]
    todo_hits: int
    dependencies: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)


def run_git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def files_under(path: Path) -> list[Path]:
    result: list[Path] = []
    for item in path.rglob("*"):
        if item.is_dir() and item.name in SKIP_DIRS:
            continue
        if item.is_file() and not any(part in SKIP_DIRS for part in item.relative_to(path).parts):
            result.append(item)
    return result


def classify(path: Path) -> str:
    return {".py": "python", ".rs": "rust", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".go": "go", ".swift": "swift", ".c": "c", ".cpp": "cpp", ".md": "markdown", ".yaml": "yaml", ".yml": "yaml"}.get(path.suffix.lower(), "other")


def parse_dependencies(path: Path, manifest: str) -> list[str]:
    text = (path / manifest).read_text(errors="ignore")
    if manifest == "pyproject.toml":
        return sorted(set(re.findall(r"^[ \t]*[\"']([A-Za-z0-9][A-Za-z0-9_.-]+)(?:[<>=!~]|[\"'])", text, re.MULTILINE)))[:80]
    if manifest == "package.json":
        return sorted(set(re.findall(r'"([@A-Za-z0-9][^"/]+(?:/[^" ]+)?)"\s*:', text)))[:80]
    if manifest == "Cargo.toml":
        return sorted(set(re.findall(r"^([A-Za-z][A-Za-z0-9_-]*)\s*=", text, re.MULTILINE)))[:80]
    if manifest == "go.mod":
        return sorted(set(re.findall(r"^\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+[^ ]*)", text, re.MULTILINE)))[:80]
    return []


def scan(path: Path) -> Repository:
    all_files = files_under(path)
    def rel(item: Path) -> str:
        return str(item.relative_to(path))
    manifests = sorted(rel(p) for p in all_files if p.name in MANIFESTS)
    docs = sorted(rel(p) for p in all_files if p.name.lower().startswith(("readme", "architecture", "security", "roadmap", "design", "adr", "agents", "claude")) or "docs" in p.parts)[:120]
    tests = sorted(rel(p) for p in all_files if p.name.startswith(("test_", "tests", "*_test")) or "test" in p.parts or "tests" in p.parts)[:80]
    ci = sorted(rel(p) for p in all_files if ".github/workflows" in str(p) or p.name in {"Makefile", "justfile", "Taskfile.yml"})[:80]
    deploy = sorted(rel(p) for p in all_files if p.name in {"Dockerfile", "docker-compose.yml", "Procfile"} or any(x in p.parts for x in ("deployment", "packaging", "infrastructure")))[:80]
    research = sorted(rel(p) for p in all_files if any(token in p.name.lower() for token in ("research", "paper", "benchmark", "experiment", "finding", "audit")))[:100]
    counts: dict[str, int] = {}
    todo_hits = 0
    for file in all_files:
        language = classify(file)
        counts[language] = counts.get(language, 0) + 1
        if language in {"python", "rust", "typescript", "javascript", "go", "swift"}:
            try:
                todo_hits += len(re.findall(r"\b(?:TODO|FIXME|HACK|XXX)\b|not implemented", file.read_text(errors="ignore"), re.IGNORECASE))
            except OSError:
                pass
    dependencies = sorted({dep for manifest in manifests for dep in parse_dependencies(path, manifest)})
    signals = []
    if not tests:
        signals.append("no_detected_tests")
    if not ci:
        signals.append("no_detected_ci")
    if not any(p.endswith("lock") or "lock" in p for p in manifests + [str(x) for x in all_files]):
        signals.append("lockfile_not_detected")
    if todo_hits:
        signals.append(f"unfinished_markers:{todo_hits}")
    return Repository(
        id=path.name,
        path=str(path),
        remote=run_git(path, "remote", "get-url", "origin"),
        branch=run_git(path, "branch", "--show-current"),
        last_commit=run_git(path, "log", "-1", "--format=%h %ad %s", "--date=short"),
        file_count=len(all_files), languages=counts, manifests=manifests, test_paths=tests,
        documentation=docs, ci_paths=ci, deployment_paths=deploy, research_paths=research,
        todo_hits=todo_hits, dependencies=dependencies, signals=signals,
    )


def discover(roots: list[Path]) -> list[Repository]:
    paths: set[Path] = set()
    for root in roots:
        if (root / ".git").is_dir() and root.name != "Projects":
            paths.add(root)
        if root.is_dir():
            paths.update(p for p in root.iterdir() if (p / ".git").is_dir())
    return [scan(path) for path in sorted(paths)]


def write_artifacts(repos: list[Repository], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "engineering-intelligence/v1", "generated_at": datetime.now(UTC).isoformat(), "repositories": [asdict(r) for r in repos]}
    (out / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    edges = []
    for repo in repos:
        edges.append({"source": "ecosystem", "target": repo.id, "type": "contains", "evidence": repo.path})
        for dep in repo.dependencies:
            edges.append({"source": repo.id, "target": dep, "type": "depends_on", "evidence": "manifest"})
    graph = {"schema": "engineering-knowledge-graph/v1", "generated_at": payload["generated_at"], "nodes": [{"id": "ecosystem", "kind": "ecosystem"}] + [{"id": r.id, "kind": "repository", "path": r.path} for r in repos], "edges": edges}
    (out / "ARCHITECTURE_GRAPH.json").write_text(json.dumps(graph, indent=2) + "\n")
    rows = []
    for r in repos:
        rows.append(f"| {r.id} | {r.languages.get('python', 0)} py / {r.languages.get('rust', 0)} rs / {r.languages.get('typescript', 0)} ts | {len(r.test_paths)} | {len(r.ci_paths)} | {len(r.dependencies)} | {', '.join(r.signals) or 'none'} |")
    (out / "SYSTEM_MAP.md").write_text("# Ecosystem System Map\n\nGenerated by `scripts/ecosystem_inventory.py`. This is an evidence-backed inventory, not a claim that every repository is production-ready.\n\n## Scope\n\nLocal repositories under `~/mac-ai-os` and `~/Projects`, plus GitHub ownership metadata collected separately.\n\n## Repository inventory\n\n| Repository | Stack signal | Test paths | CI paths | Dependencies | Signals |\n|---|---:|---:|---:|---:|---|\n" + "\n".join(rows) + "\n\n## Dominant system clusters\n\n- **Agent operating systems:** MAOS, trajectory-os, GrokHack, jarword, local-operator.\n- **Trust and adversarial validation:** trustcard, heartland-factory, sentinel-graph, repo-archaeologist.\n- **Distributed protocols:** AAFP, AAFP-research, aafp-go, infernet, petals.\n- **ML correctness and deployment:** cvconform, apple quality recognition, ONNX Runtime, CoreML tools, Ultralytics, Transformers.\n- **Native macOS intelligence:** X-MaC and related utilities.\n\n## Evidence gaps\n\nThe inventory currently proves repository presence, manifests, test/CI/documentation signals, dependency declarations, and recent git activity. It does not yet prove test quality, deployment success, runtime compatibility, ownership, or security posture. Those become validation jobs.\n")
    (out / "ENGINEERING_RISKS.md").write_text("# Engineering Risks\n\n## Highest-leverage risks\n\n| ID | Risk | Probability | Impact | Confidence | Affected systems |\n|---|---|---:|---:|---:|---|\n| R1 | Validation is fragmented across bespoke CI, benchmark reports, and local commands; there is no ecosystem-wide evidence schema. | High | Critical | High | All active repositories |\n| R2 | Multiple projects independently implement memory, graphs, orchestration, trust, and repository analysis. Interfaces are not standardized. | High | High | High | MAOS, trajectory-os, repo-archaeologist, trustcard, GrokHack |\n| R3 | ML/runtime claims can drift from deployed behavior; model format, provider fallback, and hardware assumptions are not continuously differential-tested. | High | Critical | High | local-operator, cvconform, X-MaC, ML forks |\n| R4 | Security controls are strong in isolated projects but lack a shared authorization boundary and cross-project threat model. | Medium | Critical | Medium | trustcard, MAOS, GrokHack, heartland-factory |\n| R5 | Research and roadmap artifacts are abundant, but completion and freshness are not machine-enforced. | High | Medium | High | AAFP, Infernet, X-MaC, trajectory-os |\n| R6 | Local-first deployments rely on Apple Silicon, disk, credentials, and external services with uneven reproducibility. | Medium | High | High | MAOS, local-operator, X-MaC, jarword |\n\n## Required validation\n\nEvery finding must carry evidence command, commit, timestamp, severity, confidence, affected components, and remediation cost. A green test command without captured evidence is not a green system.\n")
    (out / "VALIDATION_ROADMAP.md").write_text("# Validation Roadmap\n\n## Phase 0 — Evidence substrate (now)\n\n1. Run deterministic inventory on every repository and commit `inventory.json` plus graph snapshots.\n2. Define a common `ValidationEvidence` JSON schema: subject, commit, command, environment, exit code, duration, artifacts, verdict, limitations.\n3. Add freshness checks so stale reports cannot masquerade as current proof.\n\n## Phase 1 — Cross-project gates\n\n- Test/lint/type/security/dependency/deployment checks behind adapters.\n- Reproducible command discovery from project instructions.\n- Diff impact analysis: changed files → owning modules → validation jobs.\n- Secret and dangerous-sink scanning with allowlisted test fixtures.\n\n## Phase 2 — Adversarial and semantic validation\n\n- Hypothesis/property tests for protocol, trust, graph, and policy invariants.\n- Differential tests for AAFP Rust/Go, CV runtimes, model providers, and serialization.\n- Fuzz harnesses for parsers, MCP manifests, protocol frames, and file/indexing boundaries.\n- Failure injection and chaos tests for distributed nodes, model fallback, disk pressure, and credential expiry.\n\n## Phase 3 — Decision loop\n\nRank findings by `impact × probability × leverage × confidence`, penalized by remediation cost and evidence age. Feed outcomes into MAOS memory and repository pull-request checks.\n\n## Exit criteria\n\nA project is ecosystem-ready when its primary paths have repeatable evidence, dependency provenance, rollback instructions, a threat model, and at least one adversarial or differential validation path.\n")
    (out / "DECISION_LOG.md").write_text("# Decision Log\n\n## 2026-08-06 — Make MAOS the substrate host\n\n**Decision:** Implement the first reusable layer in MAOS rather than creating another standalone platform.\n\n**Rationale:** MAOS already owns SQLite, events, indexing, graph storage, workflows, permissions, evidence-producing guards, and MCP. A parallel platform would duplicate the highest-value infrastructure.\n\n**Trade-off:** MAOS becomes a broader control plane and must preserve stable interfaces.\n\n## 2026-08-06 — Start deterministic, then add agents\n\n**Decision:** Inventory and evidence collection are deterministic; LLMs may summarize or prioritize only after raw evidence is stored.\n\n**Rationale:** Current research shows coding agents improve when grounded in execution feedback, while model agreement is not proof. Determinism makes findings reproducible and auditable.\n\n## 2026-08-06 — Use adapters, not a monorepo\n\n**Decision:** Keep repositories independent and connect them through evidence, graph, and validation adapter contracts.\n\n**Rationale:** The ecosystem includes Rust, Python, TypeScript, Swift, Go, upstream forks, and deployed artifacts. A monorepo would increase coupling without solving runtime validation.\n")
    (out / "OPPORTUNITY_MATRIX.md").write_text("# Opportunity Matrix\n\nScoring uses strategic leverage, not novelty.\n\n| Opportunity | Impact | Probability | Leverage | Cost | Priority | Why now |\n|---|---:|---:|---:|---:|---:|---|\n| Shared evidence schema + adapter runner | 5 | 5 | 5 | 2 | 62.5 | Converts scattered tests into comparable proof |\n| Repository graph and change-impact engine | 5 | 4 | 5 | 3 | 33.3 | Reuses repo-archaeologist and MAOS graph capabilities |\n| Cross-runtime differential validation | 5 | 4 | 5 | 4 | 25.0 | Directly targets AAFP, CV, and ML drift |\n| Authorized adversarial validation packs | 5 | 4 | 4 | 3 | 26.7 | Reuses trustcard, fuzzing, and security research |\n| Reproducible deployment probes | 4 | 4 | 4 | 3 | 21.3 | Finds environment-only failures before users do |\n| Unified ownership/dependency policy | 4 | 3 | 4 | 2 | 24.0 | Reduces duplicate effort and unclear accountability |\n\n## Duplicate efforts to consolidate\n\n- Repository analysis: repo-archaeologist + MAOS workspace/index/graph.\n- Trust/policy/evidence: trustcard + MAOS guards + heartland trust architecture.\n- Agent memory/context: MAOS + trajectory-os + jarword + GrokHack integration.\n- Differential correctness: cvconform + AAFP interop tests + ML export parity work.\n- Distributed execution: AAFP + Infernet + Petals.\n\n## Real-world applicability\n\nThe substrate is directly useful for local AI agents, MCP supply-chain trust, Apple Silicon ML deployment, distributed inference, and inherited repository onboarding. The common product is not another agent. It is verifiable engineering state.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path.home() / "mac-ai-os" / "engineering-intelligence")
    parser.add_argument("roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    args = parser.parse_args()
    repos = discover(args.roots)
    write_artifacts(repos, args.out)
    print(f"indexed {len(repos)} repositories -> {args.out}")

if __name__ == "__main__":
    main()
