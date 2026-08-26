# Validation Roadmap

## Phase 0 — Evidence substrate (now)

1. Run deterministic inventory on every repository and commit `inventory.json` plus graph snapshots.
2. Define a common `ValidationEvidence` JSON schema: subject, commit, command, environment, exit code, duration, artifacts, verdict, limitations.
3. Add freshness checks so stale reports cannot masquerade as current proof.

## Phase 1 — Cross-project gates

- Test/lint/type/security/dependency/deployment checks behind adapters.
- Reproducible command discovery from project instructions.
- Diff impact analysis: changed files → owning modules → validation jobs.
- Secret and dangerous-sink scanning with allowlisted test fixtures.

## Phase 2 — Adversarial and semantic validation

- Hypothesis/property tests for protocol, trust, graph, and policy invariants.
- Differential tests for AAFP Rust/Go, CV runtimes, model providers, and serialization.
- Fuzz harnesses for parsers, MCP manifests, protocol frames, and file/indexing boundaries.
- Failure injection and chaos tests for distributed nodes, model fallback, disk pressure, and credential expiry.

## Phase 3 — Decision loop

Rank findings by `impact × probability × leverage × confidence`, penalized by remediation cost and evidence age. Feed outcomes into MAOS memory and repository pull-request checks.

## Exit criteria

A project is ecosystem-ready when its primary paths have repeatable evidence, dependency provenance, rollback instructions, a threat model, and at least one adversarial or differential validation path.
