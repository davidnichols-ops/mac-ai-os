# Security Validation Report

Generated from bounded source analysis and threat-model planning. This report does not claim ecosystem security.

## 1. Current attack surface

The first pass covers repository entry points, authentication/permission/trust/manifest/MCP paths, sensitive-path signals, deployment files, and dependency manifests across the mapped ecosystem.

## 2. Systems analyzed

Priority systems: trustcard, MAOS, AAFP, Infernet, cvconform, repository infrastructure.

## 3. Validation performed

- Deterministic repository inventory across 38 local repositories.
- Attack-surface structure extraction.
- Conservative source-pattern analysis.
- STRIDE-style threat hypothesis generation.
- Dependency manifest and provenance signal extraction.
- Bounded mutation-case generation primitives.

## 4. Confirmed findings

**None established by this phase.** Source signals and threat hypotheses are not vulnerabilities until a reproducible test demonstrates impact.

## 5. Unverified hypotheses

- MCP tool identity or manifest changes may cross a trust boundary if provenance is not bound to the exact tool schema.
- Agent execution paths may permit capability confusion if authorization is checked before normalization or after dispatch.
- Protocol parsers may mishandle malformed, truncated, duplicated, or version-incompatible frames.
- ML validation may miss numerical drift outside tested fixtures or input distributions.
- Distributed node discovery and routing may rely on unauthenticated or stale node state.

## 6. Missing validation coverage

- Cross-project authorization contract tests.
- Replay and signature-binding tests for trustcard.
- Property/fuzz tests for AAFP and MCP parsers.
- Failure-injection tests for Infernet.
- Adversarial input and metamorphic tests for cvconform.
- Advisory-backed dependency scans with freshness evidence.
- CI permission and secret-boundary verification.

## 7. Highest-risk areas

1. Trust identity and authorization across MCP boundaries.
2. Agent-controlled execution and prompt/tool injection.
3. Protocol parsing and resource exhaustion.
4. Cross-runtime ML correctness claims.
5. Deployment and dependency provenance.

## 8. Recommended remediation order

1. Establish evidence and authorization contracts.
2. Add parser/property/fuzz harnesses.
3. Add cross-runtime differential gates.
4. Add dependency and CI provenance validation.
5. Add controlled runtime probes and failure injection.

## 9. Campaign execution status

The activated source campaigns analyzed six bounded targets: Trustcard, MAOS, AAFP, Infernet, cvconform, and repository infrastructure. They produced 12 normalized findings:

- 6 `missing_validation`
- 6 `suspected_weakness`
- 0 `confirmed_vulnerability`

This result means the system successfully generated challengeable hypotheses and coverage gaps. It does not mean the targets are secure.

Machine-readable campaign output is stored in `security_campaigns.json`. Phase 4 campaign proposals are stored in `RESEARCH_CAMPAIGNS.json` and remain approval-required.

## 10. Authorized Trustcard campaign

The first authorized campaign ran against the owned local Trustcard repository only. No production endpoint, third-party system, or real credential was contacted.

Evidence captured under `engagement/trustcard/evidence/`:

- full local Node test suite: 513 passed, 1 failed
- targeted security/auth/proxy suite: 92 passed, 1 failed
- `npm run lint`: unavailable because no lint script is declared
- direct expiry reproduction output

The one failing test concerns manifest expiry duration semantics. The implementation uses calendar-day arithmetic while the test expects a fixed elapsed 90-day interval. This is classified as a low-impact suspected weakness/test-contract mismatch, not a confirmed vulnerability.

No unauthorized access, privilege escalation, manifest bypass, replay acceptance, signature failure, or tool authorization failure was demonstrated by this campaign.
