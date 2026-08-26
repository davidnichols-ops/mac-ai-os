# Authorized Target Queue

Generated 2026-08-06 from local ownership evidence. This queue contains only repositories owned by `davidnichols-ops` or the local MAOS workspace. No third-party production systems are included.

## Authorization policy

- Authorization basis: local filesystem access plus repository remotes owned by `davidnichols-ops`.
- Allowed activity: offline source analysis, local unit tests, local fixtures, bounded fuzz/property generation, and isolated test processes.
- Not authorized by this artifact: probing production URLs, scanning third-party infrastructure, credential use, destructive actions, or testing systems not listed below.
- Any live program requires explicit written scope, exclusions, rate limits, test accounts, and disclosure channel before activation.

## Queue

| Rank | Program / repository | Scope | Rationale | Expected attack surface | Authorization state |
|---:|---|---|---|---|---|
| 1 | trustcard | `~/Projects/trustcard` | Highest concentration of MCP trust, identity, authorization, manifest, and provenance logic. | MCP proxy, JSON-RPC interception, manifests, auth, signatures, receipts, policy, child-process boundary. | Owned local repository; offline validation authorized. |
| 2 | mac-ai-os / MAOS | `~/mac-ai-os` | Agent execution and permission boundary; substrate host. | Tool router, action executor, shell/filesystem tools, MCP, memory/context ingestion, approvals. | Owned local workspace; offline validation authorized. |
| 3 | AAFP | `~/Projects/aafp` | Protocol and cryptographic state-machine surface with fuzz fixtures. | CBOR/frame/RPC parsing, handshake, replay cache, identity, discovery, relay. | Owned local repository; offline validation authorized. |
| 4 | cvconform | `~/Projects/cvconform` | Differential ML correctness and untrusted artifact loading. | Model loaders, runtime adapters, output alignment, tolerance policies, reducer. | Owned local repository; offline validation authorized. |
| 5 | Infernet | `~/Projects/infernet` | Distributed discovery, trust, routing, and remote execution. | Node server, discovery, transport, capability, provider selection, artifact/model execution. | Local repository; offline validation only; no live network campaign. |
| 6 | repo-archaeologist | `~/Projects/repo-archaeologist` | Repository infrastructure and source-analysis boundary. | Path handling, archive/URL resolution, source scanning, report generation. | Owned local repository; offline validation authorized. |

## Deferred targets

- Public bug-bounty programs: deferred. No explicit program scope, exclusions, test accounts, or disclosure authorization is present in the workspace.
- External SaaS, cloud, developer infrastructure, AI/ML platforms, and production APIs: deferred until written authorization is recorded.

## First campaign

Trustcard offline validation is selected first. It has the highest expected combination of impact, feasibility, and reusable evidence value without requiring live access.
