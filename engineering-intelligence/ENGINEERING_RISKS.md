# Engineering Risks

## Highest-leverage risks

| ID | Risk | Probability | Impact | Confidence | Affected systems |
|---|---|---:|---:|---:|---|
| R1 | Validation is fragmented across bespoke CI, benchmark reports, and local commands; there is no ecosystem-wide evidence schema. | High | Critical | High | All active repositories |
| R2 | Multiple projects independently implement memory, graphs, orchestration, trust, and repository analysis. Interfaces are not standardized. | High | High | High | MAOS, trajectory-os, repo-archaeologist, trustcard, GrokHack |
| R3 | ML/runtime claims can drift from deployed behavior; model format, provider fallback, and hardware assumptions are not continuously differential-tested. | High | Critical | High | local-operator, cvconform, X-MaC, ML forks |
| R4 | Security controls are strong in isolated projects but lack a shared authorization boundary and cross-project threat model. | Medium | Critical | Medium | trustcard, MAOS, GrokHack, heartland-factory |
| R5 | Research and roadmap artifacts are abundant, but completion and freshness are not machine-enforced. | High | Medium | High | AAFP, Infernet, X-MaC, trajectory-os |
| R6 | Local-first deployments rely on Apple Silicon, disk, credentials, and external services with uneven reproducibility. | Medium | High | High | MAOS, local-operator, X-MaC, jarword |

## Required validation

Every finding must carry evidence command, commit, timestamp, severity, confidence, affected components, and remediation cost. A green test command without captured evidence is not a green system.
