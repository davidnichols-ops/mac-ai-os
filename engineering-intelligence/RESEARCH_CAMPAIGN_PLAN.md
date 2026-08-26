# Research Campaign Plan

## Operating boundary

This plan is limited to owned local repositories and offline or isolated test execution. It does not authorize production probing or third-party testing.

## Priority model

```text
priority = impact × exploitability × confidence × leverage / cost
```

Evidence status is separate from priority. A high-priority hypothesis is not a vulnerability until reproduced.

## Campaign 1 — Trustcard MCP boundary

**Hypotheses**

1. Manifest identity, schema, and authorization may become inconsistent after tool drift.
2. Duplicate/replayed JSON-RPC identifiers may confuse pending-request or approval state.
3. `--auto-update` may turn untrusted runtime discovery into an unreviewed policy mutation.
4. Child-process environment inheritance may expose credentials across the proxy boundary.
5. Non-tool JSON-RPC methods may bypass the same policy checks as `tools/call`.

**Validation methods**

- Run repository-native adversarial/auth/proxy/evidence tests.
- Inspect manifest canonicalization, identity binding, and drift transitions.
- Use local fixture MCP servers only.
- Generate bounded duplicate-ID, malformed-message, and tool-schema mutation cases.
- Verify policy decisions and receipts against exact manifest/schema hashes.
- Do not use real credentials or external MCP endpoints.

**Expected evidence**

- Exact command and commit.
- Fixture server source and manifest digest.
- Input JSON-RPC message or mutation seed.
- Proxy decision, response, receipt, and exit status.
- Before/after manifest state for auto-update tests.

## Campaign 2 — MAOS execution boundary

**Hypotheses**

- Tool authorization may diverge between planning, routing, and execution.
- Prompt-injected content may influence tool selection or memory persistence.
- Sensitive context may cross an MCP/tool response boundary.

**Validation methods**

- Run core safety, tool executor, MCP, action, and approval tests.
- Use synthetic hostile tool descriptions and context documents.
- Verify permission checks occur at execution, not only planning.
- Assert secrets are absent from logs, context packets, and evidence.

## Campaign 3 — AAFP protocol boundary

**Hypotheses**

- Malformed/truncated frames may cause parser crashes or unbounded allocation.
- Version negotiation may accept incompatible capabilities.
- Replay and revocation state may become inconsistent across sessions.

**Validation methods**

- Run workspace, conformance, crypto, identity, and fuzz targets.
- Use bounded mutation cases against local decoders.
- Record parser verdict, allocation/time limit, and minimized input.

## Campaign 4 — cvconform differential boundary

**Hypotheses**

- Positional output fallback may hide semantic output reordering.
- Runtime exceptions may be conflated with model divergence.
- Input generators may not cover adversarial shapes/dtypes/ranges.

**Validation methods**

- Run comparison, integration, fault-injection, discovery, and reduction tests.
- Build a fixture with same-shaped outputs but different semantic names/order.
- Record runtime versions, artifact digest, seeds, tolerance policy, and alignment mode.

## Campaign 5 — Infernet distributed execution

**Hypotheses**

- Discovery metadata may be trusted before pin/descriptor validation.
- Provider selection may route work to stale or incompatible nodes.
- Remote artifact paths and sync operations may cross trust boundaries.

**Validation methods**

- Offline unit and subprocess tests only.
- No live peer discovery or external network traffic.
- Inject stale descriptors, malformed messages, provider failures, and invalid paths into fixtures.

## Campaign 6 — Repository infrastructure

**Hypotheses**

- Source/archive path handling may allow traversal or symlink escapes.
- CI workflows may have excessive token/write permissions.
- Dependency declarations may lack lock/provenance/advisory coverage.

**Validation methods**

- Use temporary directories and synthetic archives.
- Parse workflow permissions without dispatching workflows.
- Run dependency tools only against local manifests and lockfiles.

## Evidence rule

Every candidate must contain:

```text
hypothesis + test + observation + evidence + impact
```

Static signals remain `suspected_weakness` or `missing_validation` until a reproducible test demonstrates security impact.
