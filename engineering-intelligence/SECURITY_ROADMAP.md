# Security Roadmap

## Immediate fixes

- Make every security finding carry hypothesis, test, observation, evidence, impact, and status.
- Treat source scanners as suspected weaknesses, never confirmed vulnerabilities.
- Add lockfile/advisory freshness checks to repository adapters.
- Define a shared capability and authorization vocabulary for MAOS, trustcard, and GrokHack.

## Architectural improvements

- Bind MCP provenance to canonical tool schemas and invocation policy.
- Add a common evidence store and event stream in MAOS.
- Add change-impact routing from repository graph to validation adapters.
- Use trajectory-os decision models for prioritization, not for proof.
- Keep cvconform and AAFP differentials as reusable challenge adapters.

## Future validation capabilities

- Stateful protocol fuzzing.
- Replay, downgrade, and identity-confusion campaigns.
- Sandboxed agent permission tests.
- Distributed chaos and stale-node experiments.
- Metamorphic and numerical robustness testing for ML.
- Continuous external advisory ingestion with signed evidence.

## Phase 4 — Autonomous research operator boundary

The first planning layer now ranks findings and emits approval-required campaign proposals. It deliberately does not self-authorize hostile runtime actions.

Next capabilities:

- change-aware campaign selection from graph diffs
- signed external vulnerability intelligence ingestion
- repository-native test command discovery
- isolated parser and authorization campaigns
- evidence freshness and regression scheduling
- MAOS workflow/approval integration
- trajectory-os opportunity and risk scoring adapter

## Trustcard campaign actions

1. Decide whether manifest expiry means calendar days or fixed elapsed duration.
2. Add a timezone/DST regression test.
3. Add a deterministic lint/static-analysis command and CI gate.
4. Execute the local duplicate-ID, environment-boundary, and non-tool-method fixture tests.
5. Persist Trustcard receipts and policy decisions as EIS evidence records.
