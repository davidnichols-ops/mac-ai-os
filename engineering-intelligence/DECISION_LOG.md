# Decision Log

## 2026-08-06 — Make MAOS the substrate host

**Decision:** Implement the first reusable layer in MAOS rather than creating another standalone platform.

**Rationale:** MAOS already owns SQLite, events, indexing, graph storage, workflows, permissions, evidence-producing guards, and MCP. A parallel platform would duplicate the highest-value infrastructure.

**Trade-off:** MAOS becomes a broader control plane and must preserve stable interfaces.

## 2026-08-06 — Start deterministic, then add agents

**Decision:** Inventory and evidence collection are deterministic; LLMs may summarize or prioritize only after raw evidence is stored.

**Rationale:** Current research shows coding agents improve when grounded in execution feedback, while model agreement is not proof. Determinism makes findings reproducible and auditable.

## 2026-08-06 — Use adapters, not a monorepo

**Decision:** Keep repositories independent and connect them through evidence, graph, and validation adapter contracts.

**Rationale:** The ecosystem includes Rust, Python, TypeScript, Swift, Go, upstream forks, and deployed artifacts. A monorepo would increase coupling without solving runtime validation.
