# Project Decisions — 2026-08-03

## Three-State Project Policy: SHIP / MAINTAIN / ARCHIVE

### Decisions

| Project | State | Rationale |
|---------|-------|-----------|
| **trustcard** | **SHIP** | v3.0.0 ready to publish. 514 tests, bruce-lee reference server, evidence substrate. Commit untracked work, update README, publish. |
| **AAFP** | **MAINTAIN** | Protocol is coherent, 2869 tests, 2 implementations. Maintain with fixed biweekly time allocation. Publish architecture paper. |
| **Infernet** | **SHIP** | v0.1 ready after gitignore fix + clean commit. 3-node demo works. Architecture docs excellent. |
| **X-MaC** | **SHIP** | v1 scope locked (3 capabilities). Fix broken tests (done), implement safety, benchmark, sign, notarize. |
| **Petals fork** | **ARCHIVE** | Superseded by Infernet. Trust layer to be ported to Infernet. Tag final release, mark as research input. |
| **Elonwatch** | **SHIP** | Finished experiment. DMG built, README polished. No further development needed. |
| **apple-quality-recognition-engine** | **MAINTAIN** | Production evaluation needed. Lock class schema, build repeatable pipeline, measure on real hardware. |
| **Suburban Scan** | **ARCHIVE** | No clear user/business path. Baseline experiment not started. Archive after confirming no strategic relation to apple-recognition. |
| **NVIDIA certification** | **DECISION: DROP** | No evidence of strategic value for current work. Formally drop instead of leaving "in progress." |
| **Website factory** | **MAINTAIN** | Measure revenue per hour. If not meaningful after defined experiment, shut down. |
| **MAOS (mac-ai-os)** | **MAINTAIN** | Active development. 841 tests passing. Continue test coverage + agent loop improvements. |

### Idea Quarantine Policy

Every new idea goes into a document first. Cannot become a repository until an
existing project is shipped, archived, or explicitly paused.

Current active projects: trustcard, AAFP, Infernet, X-MaC, apple-quality-recognition-engine, MAOS.
That's 6 — already at the limit. No new projects until one is shipped or archived.

### Weekly Finish List (this week)

1. **Publish trustcard v3.0.0** — commit untracked work, update README, npm publish
2. **Clean and push Infernet** — fix gitignore (done), clean working tree, create public repo
3. **Merge or split X-MaC PR #9** — the single biggest open loop

### Archive List

Projects to archive this week:
- Petals fork → tag v0.3-final, mark as superseded by Infernet
- Suburban Scan → archive, no baseline experiment started
- cloud-to-butt / cloud-to-butt-revolutionized → archive (jokes)
- jane-street-md5-puzzle → archive (one-off)
- mif_env_mapper-v1.0.0 → archive (stale)
- book-inventory → make private (personal)

### NVIDIA Certification Decision

**Decision: DROP.** The certification track is not strategically useful for the
current work (trusted agents, distributed inference, edge AI on Apple Silicon).
Formally drop it instead of leaving it permanently "in progress." If GPU
profiling skills become needed, learn them through actual projects (Infernet
benchmarks, AAFP performance validation) rather than certification study.
