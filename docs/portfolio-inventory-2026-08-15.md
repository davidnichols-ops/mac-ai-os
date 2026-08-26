# Portfolio Inventory — 2026-08-15

**Issue:** DAV-17 — Inventory all 10 projects and define done, kill criteria, evidence, and budget
**Project:** 90-Day Trajectory Reset — Aug 10 to Nov 7, 2026
**Author:** Devin (Eve), evidence-driven from git history, Linear, READMEs, filesystem
**Confidence:** State/evidence = high (git-verified). Purpose/done/kill criteria = medium (inferred from project descriptions, not user-confirmed). Portfolio scores = low-medium (provisional, awaiting user calibration).

---

## Methodology

The 90-Day Reset project description states "10 active projects are tracked" as a diagnostic claim, but no canonical enumerated list exists in Trajectory OS, MAOS docs, or Linear. This inventory derives the active set from **git last-commit dates** (evidence: `git log -1 --format='%ci'`) across `~/Projects/` and `~/mac-ai-os/`, cross-referenced with Linear projects and the portfolio-vision doc.

**Finding: the active set is larger than 10.** 14 repos had commits in the last 10 days (Aug 6–15). Two additional Linear initiatives (Iowa, Wisephone) have no local repo. This inventory covers all 16 and classifies each per the portfolio cap (1 flagship + 2 secondaries + 1 experiment + parked).

**Scoring rubric (1–5 each, provisional):**
- **Mission alignment:** Does this serve the stated mission (personal AI infrastructure, secure edge, local-LLM community)?
- **Leverage:** Does completing this unlock other work or create disproportionate value?
- **Evidence potential:** Can progress be verified with concrete artifacts (tests, benchmarks, shipped builds)?
- **Finishability:** Is the scope bounded enough to complete in the 90-day window?
- **Opportunity cost:** Inverse — how much does keeping this active cost vs. parking it? (5 = low cost to keep, 1 = high cost)

---

## Active Code Projects (14 repos, commits Aug 6–15)

### 1. mac-ai-os (MAOS)
| Field | Value |
|---|---|
| **Path** | `~/mac-ai-os` |
| **Last commit** | 2026-08-15 (today) |
| **Purpose** | Personal AI operating system — agent loop, evidence recording, workspace indexing, Slack bridge, Trajectory adapter. The daily execution surface. |
| **Current state** | Phase 3.5 complete (agency layer). 54 workspaces, 14 repos, 21 custom CLIs indexed. MCP server, CLI, umbrella view all functional. 9 pending MAOS tasks (DeepSWE, refresh fixes, DMG signing). |
| **Next deliverable** | Fix MAOS refresh workspace profiling + storage scan blockers (MAOS task #726). Then one `do` task consuming a Trajectory context pack (90-day exit gate). |
| **Definition of done** | One MAOS `do` task consumes a Trajectory context pack and produces a verified result. Slack `#maos-bridge` round trip verified. |
| **Kill/pause criterion** | If Trajectory onboarding doesn't complete and no `do` task consumes real context within 30 days → pause MAOS feature work, maintain only. |
| **Evidence path** | `~/mac-ai-os/data/maos.db`, `test-evidence.json`, `lint-evidence.json`, `uv run pytest`, `uv run app verify` |
| **Mission contribution** | Core — this IS the execution surface. Without MAOS, nothing else gets verified. |
| **Weekly time budget** | 8–10h (daily driver) |
| **Classification** | **Flagship candidate** |
| **Score** | Mission 5 · Leverage 5 · Evidence 4 · Finishability 3 · Opp cost 3 → **20/25** |

### 2. trajectory-os
| Field | Value |
|---|---|
| **Path** | `~/Projects/trajectory-os` |
| **Last commit** | 2026-08-06 (9 days ago — borderline) |
| **Purpose** | Device-wide AI workflow + personal decision/context substrate. Durable knowledge graph, provider-neutral context packs. |
| **Current state** | Docs complete (24 docs in `docs/`). Local DB still empty — onboarding not completed. MAOS ingest path exists but has a command-path defect. 54 dirty files (uncommitted). |
| **Next deliverable** | Complete Trajectory onboarding: run `trajectory ingest-maos --maos-db ~/mac-ai-os/data/maos.db --write`. Fix the refresh integration command-path defect (90-day gap). |
| **Definition of done** | Daily context pack contains actual goals, constraints, and priorities. Active goals in Trajectory OS. |
| **Kill/pause criterion** | If onboarding not completed within 14 days → park; MAOS already provides most of the execution surface. |
| **Evidence path** | `~/Projects/trajectory-os/data/`, `trajectory ingest-maos --dry-run` |
| **Mission contribution** | High — the decision substrate that tells MAOS what to work on. But currently zero operational value (empty DB). |
| **Weekly time budget** | 3–5h (onboarding + integration fix) |
| **Classification** | **Secondary candidate** (dependent on MAOS integration) |
| **Score** | Mission 4 · Leverage 4 · Evidence 2 · Finishability 3 · Opp cost 3 → **16/25** |

### 3. trustcard
| Field | Value |
|---|---|
| **Path** | `~/Projects/trustcard` |
| **Last commit** | 2026-08-13 |
| **Purpose** | Cryptographic trust infrastructure for MCP servers. Manifests, danger detection (3-engine fusion), signed receipts, evidence substrate, per-agent OAuth scopes. Published to npm as `mcp-trustcard` v3.0.3. |
| **Current state** | v3.0.3 shipped. 5 rogue server demos. Full crypto subcommands. Evidence substrate + auth scope enforcement. npm published. |
| **Next deliverable** | Production hardening — audit the v3 evidence substrate for chain integrity edge cases. Integrate with MAOS MCP server as a real consumer. |
| **Definition of done** | One real MCP server (MAOS) runs behind a trustcard proxy in daily use with zero bypass incidents for 30 days. |
| **Kill/pause criterion** | If no real consumer adopts it within 60 days → park at v3.0.3, maintain npm only. |
| **Evidence path** | `~/Projects/trustcard/bin/mcp-trustcard.js scan`, npm registry API, `evidence verify` |
| **Mission contribution** | High — security infrastructure for the AI-agent ecosystem. Differentiated (no comparable MCP trust layer exists). |
| **Weekly time budget** | 4–6h |
| **Classification** | **Flagship candidate** (competing with MAOS) |
| **Score** | Mission 4 · Leverage 4 · Evidence 5 · Finishability 4 · Opp cost 3 → **20/25** |

### 4. claude-yolo-train (→ claude-yolo-vibes Launch)
| Field | Value |
|---|---|
| **Path** | `~/Projects/claude-yolo-train` |
| **Linear project** | claude-yolo-vibes — Launch & Distribution |
| **Last commit** | 2026-08-15 (today) |
| **Purpose** | Fine-tuned Qwen3-1.7B / Qwen2.5-Coder-7B with "yolo vibes" personality. Novelty/entertainment model for the local-LLM community. Not a product — reputation + engagement play. |
| **Current state** | v3 shipped to HuggingFace (86.6% HumanEval, SFT+DPO). v4 SFT in progress (100K examples). MI300X remote training. Linear has 10 launch issues (DAV-26–36), most in Todo/Backlog. |
| **Next deliverable** | DAV-26 (Urgent): Get on an inference provider. Then DAV-30: ship MLX 4-bit build for Apple Silicon. |
| **Definition of done** | 30-day content campaign completed (4 weeks per Linear project plan). Engagement measured: comments, shares, people posting their own CHAOS-AI outputs. |
| **Kill/pause criterion** | If v4 doesn't improve HumanEval over v3 within 2 training runs → stop training, ship v3 as final. If launch campaign generates zero community engagement after Week 2 → park. |
| **Evidence path** | HuggingFace model stats, HumanEval eval (`eval_humaneval_v2.py`), Reddit/X engagement metrics |
| **Mission contribution** | Medium — builds public reputation in the local-LLM community. Tangential to core infrastructure mission but high visibility. |
| **Weekly time budget** | 5h (capped — this is the "experiment" slot per portfolio cap) |
| **Classification** | **Experiment candidate** (bounded, 5h/week cap, 30-day window) |
| **Score** | Mission 2 · Leverage 3 · Evidence 4 · Finishability 4 · Opp cost 4 → **17/25** |

### 5. Iowa Secure Edge Vision Initiative
| Field | Value |
|---|---|
| **Linear project** | Iowa Secure Edge Vision Initiative — 90-Day Pilot |
| **Local repo** | None yet |
| **Last activity** | Linear issues created 2026-08-14 (DAV-37 through DAV-57) |
| **Purpose** | Prove a secure, hardware-neutral runtime for deploying CV workloads on existing Iowa public cameras. Convert prototype into a state/DOT pilot proposal. |
| **Current state** | Phase 0/1 on Linear (20 issues, all Backlog). No code, no prototype. Concept stage. DAV-38 (thesis brief, due Aug 18) is the first deliverable. |
| **Next deliverable** | DAV-38: Write one-page problem/thesis brief (due in 3 days). Then DAV-37: lock thesis, scope, architecture. |
| **Definition of done** | A named Iowa sponsor commits to a pilot OR provides a formal technical review path with a named owner (DAV-57, due Nov 13). |
| **Kill/pause criterion** | If no Iowa stakeholder conversation happens within 45 days → park. If DAV-51 (red-team review) finds fatal privacy/security flaws → revise or stop. |
| **Evidence path** | Linear issues, concept memo, prototype benchmark results, stakeholder meeting notes |
| **Mission contribution** | Very high — this is the highest-stakes strategic play. Secure edge + public infrastructure + privacy-preserving CV. But currently zero evidence (no code, no prototype, no stakeholder contact). |
| **Weekly time budget** | 6–8h (drafting + architecture + outreach prep) |
| **Classification** | **Flagship candidate** (highest mission alignment, but highest risk — no evidence yet) |
| **Score** | Mission 5 · Leverage 5 · Evidence 1 · Finishability 2 · Opp cost 2 → **15/25** |

### 6. Wisephone II / WiseOS Engineering Archaeology
| Field | Value |
|---|---|
| **Linear project** | Wisephone II / WiseOS Engineering Archaeology |
| **Local repo** | None (research artifacts in `~/Projects/research_*`) |
| **Last activity** | Linear issues DAV-5 through DAV-16, several In Progress |
| **Purpose** | Determine whether Techless/Wisephone has a high-leverage engineering bottleneck around third-party app qualification, compatibility, policy enforcement, and regression testing. Build a proof-of-work artifact if evidence supports it. |
| **Current state** | Issues DAV-5–8 In Progress (workspace setup, architecture map, device baseline, controlled experiments). But all require the **physical Wisephone II device** — not connected to this MacBook. |
| **Next deliverable** | DAV-5: Set up research workspace, evidence schema, safety boundary. Then DAV-6: Build public WiseOS architecture map (this CAN be done from public docs without the device). |
| **Definition of done** | DAV-16: Package the Techless proof-of-work and make pursue/stop decision (due Aug 29). |
| **Kill/pause criterion** | If no physical device access within 21 days → park all device-dependent issues. Keep only public-architecture work active. If proof-of-work doesn't demonstrate outsider leverage → stop. |
| **Evidence path** | `~/Projects/research_evidence_registry.md`, `~/Projects/research_methodology_note.md`, emulator logs, DNS capture |
| **Mission contribution** | Medium-high — if WiseOS has a real bottleneck, this is outsider leverage (contribute without being hired). But device-blocked. |
| **Weekly time budget** | 2–3h (only public-architecture work is unblocked) |
| **Classification** | **Parked** (device-blocked; public-architecture work can continue as secondary) |
| **Score** | Mission 3 · Leverage 4 · Evidence 2 · Finishability 2 · Opp cost 2 → **13/25** |

### 7. mcp-inference-gateway
| Field | Value |
|---|---|
| **Path** | `~/Projects/mcp-inference-gateway` |
| **Last commit** | 2026-08-14 |
| **Purpose** | Universal, secure, publicly queryable MCP server for model inference. |
| **Current state** | Active (commit yesterday). Clean tree. |
| **Next deliverable** | Verify it passes its own tests + lint. Define what "publicly queryable" means (auth model, rate limits). |
| **Definition of done** | Publicly accessible MCP inference endpoint with documented auth, passing trustcard scan, and at least one external consumer. |
| **Kill/pause criterion** | If no external consumer within 45 days → park. If trustcard scan flags unresolved FAILs → block release. |
| **Evidence path** | `~/Projects/mcp-inference-gateway` test suite, trustcard scan output |
| **Mission contribution** | Medium — bridges MCP ecosystem + inference. Natural trustcard consumer. |
| **Weekly time budget** | 2–3h |
| **Classification** | **Secondary candidate** (pairs with trustcard) |
| **Score** | Mission 3 · Leverage 3 · Evidence 3 · Finishability 3 · Opp cost 4 → **16/25** |

### 8. lan-vault
| Field | Value |
|---|---|
| **Path** | `~/Projects/lan-vault` |
| **Last commit** | 2026-08-15 (today) |
| **Purpose** | High-speed LAN file migration with bare-bones PoA blockchain audit trail. |
| **Current state** | Active (commit today). Clean tree. v0.1.0. Has tests. |
| **Next deliverable** | Verify test suite. Define production readiness (multi-node, failure modes). |
| **Definition of done** | Successfully migrates files between 2+ machines on a real LAN with verifiable audit trail and no data loss. |
| **Kill/pause criterion** | If no real LAN migration test within 30 days → park at v0.1.0. |
| **Evidence path** | `~/Projects/lan-vault/tests`, LAN migration test logs |
| **Mission contribution** | Low-medium — utility tool, tangential to core mission. |
| **Weekly time budget** | 1–2h |
| **Classification** | **Parked** (utility, not mission-critical) |
| **Score** | Mission 2 · Leverage 2 · Evidence 3 · Finishability 4 · Opp cost 5 → **16/25** |

### 9. grokhack
| Field | Value |
|---|---|
| **Path** | `~/Projects/grokhack` |
| **Last commit** | 2026-08-10 |
| **Purpose** | Personal fork of xAI's Grok CLI with theme and persona patches. |
| **Current state** | Active, 8 dirty files. Published to GitHub (davidnichols-ops/grokhack). |
| **Next deliverable** | Commit dirty changes. Decide if persona patches have community value worth distributing. |
| **Definition of done** | Persona patches documented and either merged upstream or published as a maintained fork. |
| **Kill/pause criterion** | If xAI ships equivalent features upstream → drop fork. If no personal use for 30 days → park. |
| **Evidence path** | GitHub repo, `git status` |
| **Mission contribution** | Low — personal tooling, not mission-aligned. |
| **Weekly time budget** | 0.5–1h (maintenance) |
| **Classification** | **Parked** (personal tool) |
| **Score** | Mission 1 · Leverage 2 · Evidence 3 · Finishability 4 · Opp cost 5 → **15/25** |

### 10. infoscanner
| Field | Value |
|---|---|
| **Path** | `~/Projects/infoscanner` |
| **Last commit** | 2026-08-07 |
| **Purpose** | Records everything a browser game sends/receives in an offensively useful format. Personal tool with owner-only gates (`owner_filth.py`). |
| **Current state** | 17 dirty files. No public origin. |
| **Next deliverable** | Commit or stash dirty changes. Define scope — is this active research or a parked utility? |
| **Definition of done** | Unclear — needs user input. Candidate: successfully captures and parses traffic from one target game. |
| **Kill/pause criterion** | If no active use within 21 days → park and commit the dirty state. |
| **Evidence path** | `~/Projects/infoscanner` (no test suite found) |
| **Mission contribution** | Low — personal/research utility. |
| **Weekly time budget** | 0–1h |
| **Classification** | **Parked** (personal utility) |
| **Score** | Mission 1 · Leverage 2 · Evidence 2 · Finishability 3 · Opp cost 5 → **13/25** |

### 11. local-operator
| Field | Value |
|---|---|
| **Path** | `~/Projects/local-operator` |
| **Last commit** | 2026-08-08 |
| **Purpose** | Environment-Grounded On-Policy Distillation of a Local Computer Operator. ML research — training a model to operate a computer. |
| **Current state** | 1 dirty file. MAOS tasks #724 (validate adaptive-operator-v4-gguf) and #725 (document lessons). Has a GGUF release to validate. |
| **Next deliverable** | MAOS task #724: Validate adaptive-operator-v4-gguf release against reproducible inference checks. |
| **Definition of done** | Reproducible inference check passes for adaptive-operator-v4-gguf. Lessons documented (task #725). |
| **Kill/pause criterion** | If inference validation fails irreproducibly → park and document the failure. If no downstream consumer → park after validation. |
| **Evidence path** | MAOS tasks #724/#725, `~/Projects/local-operator` test suite |
| **Mission contribution** | Medium — computer-use models are strategically relevant, but this is research, not infrastructure. |
| **Weekly time budget** | 2–3h (validation + documentation only) |
| **Classification** | **Parked** (research — validate existing artifact, then park) |
| **Score** | Mission 3 · Leverage 2 · Evidence 3 · Finishability 3 · Opp cost 4 → **15/25** |

### 12. jarvis-word
| Field | Value |
|---|---|
| **Path** | `~/Projects/jarvis-word` |
| **Last commit** | 2026-08-08 |
| **Purpose** | Local-first cognitive computing environment. Optimizes understanding, not creation. |
| **Current state** | 18 dirty files. Published to GitHub (davidnichols-ops/jarword). |
| **Next deliverable** | Commit dirty changes. Define what "understanding optimization" means concretely — what's the first user-facing feature? |
| **Definition of done** | Unclear — needs user input. Candidate: one complete understanding-optimization workflow (e.g., document → structured notes → knowledge graph). |
| **Kill/pause criterion** | If no concrete feature shipped within 45 days → park. If MAOS/trustcard already covers this → merge or drop. |
| **Evidence path** | GitHub repo, `git status` |
| **Mission contribution** | Low-medium — overlaps with MAOS/trajectory-os knowledge graph features. |
| **Weekly time budget** | 0–1h |
| **Classification** | **Parked** (overlaps with MAOS; no clear differentiation) |
| **Score** | Mission 2 · Leverage 2 · Evidence 2 · Finishability 2 · Opp cost 4 → **12/25** |

### 13. claude-local
| Field | Value |
|---|---|
| **Path** | `~/Projects/claude-local` |
| **Last commit** | 2026-08-11 |
| **Purpose** | Clean-room replica of Claude Code's agent loop that runs fully offline on local models (Ollama/LM Studio). Trimmed ~200-token system prompt for small models. |
| **Current state** | 9 dirty files. No public origin. Fuzz suite (15/15 cases pass). Integrated into the `claude` wrapper in `~/bin/`. |
| **Next deliverable** | Commit dirty changes. Decide if this stays a personal tool or gets published. |
| **Definition of done** | Personal tool — already "done" in the sense that it works daily. If publishing: fuzz suite expanded to 30 cases, README, public repo. |
| **Kill/pause criterion** | If official Claude Code adds local-model support natively → drop replica. Currently stable — maintain only. |
| **Evidence path** | `~/Projects/claude-local/fuzz/` (15/15 pass) |
| **Mission contribution** | Low — personal tooling, but enables all other local-model work. |
| **Weekly time budget** | 0.5h (maintenance) |
| **Classification** | **Parked** (stable personal tool, maintain only) |
| **Score** | Mission 2 · Leverage 3 · Evidence 4 · Finishability 5 · Opp cost 5 → **19/25** |

### 14. nevernote
| Field | Value |
|---|---|
| **Path** | `~/Projects/nevernote` |
| **Last commit** | 2026-08-11 |
| **Purpose** | Next.js project (create-next-app bootstrap). Purpose unclear from README — needs user input. |
| **Current state** | 15 dirty files. Published to GitHub (davidnichols-ops/nevernote). |
| **Next deliverable** | User input needed — what is nevernote supposed to be? |
| **Definition of done** | Unknown — needs user input. |
| **Kill/pause criterion** | If no clear purpose stated within 14 days → park. |
| **Evidence path** | GitHub repo |
| **Mission contribution** | Unknown — unresolved. |
| **Weekly time budget** | 0h (unresolved) |
| **Classification** | **Unresolved** (needs user input on purpose) |
| **Score** | Mission ? · Leverage ? · Evidence 1 · Finishability ? · Opp cost 3 → **?/25** |

---

## Additional Repos (active but upstream forks / dependencies — not "projects")

| Repo | Last commit | Status | Notes |
|---|---|---|---|
| `llama.cpp` | 2026-08-10 | Upstream fork | Inference engine dependency. Track upstream, don't count as a project. |
| `hermes-agent` | 2026-08-08 | Upstream fork (NousResearch) | Reference implementation. Track upstream, don't count as a project. |
| `repo-archaeologist` | 2026-08-08 | Published utility | Point-at-repo briefing tool. Stable, maintain only. |
| `ml_backend_lab` | Unknown | Reproducible ML env | Training/lab environment, not a project. |

---

## Meta-Projects (Linear initiatives, not code projects)

| Initiative | Status | Notes |
|---|---|---|
| 90-Day Trajectory Reset | In Progress | This is the meta-process, not a project itself. It governs the portfolio. |
| claude-yolo-vibes Launch | Maps to claude-yolo-train | The Linear project is the launch campaign; the code is claude-yolo-train. Counted as one project above. |

---

## Provisional Portfolio Classification

Per the 90-Day Reset portfolio cap: **1 flagship + 2 secondaries + 1 experiment + parked.**

### Flagship candidates (pick ONE)

| Candidate | Score | Argument for | Argument against |
|---|---|---|---|
| **MAOS** | 20/25 | Daily execution surface, already working, enables everything | Broad scope, hard to "finish" in 90 days |
| **trustcard** | 20/25 | Published, differentiated, evidence-rich, bounded scope | Needs a real consumer to prove value |
| **Iowa Secure Edge Vision** | 15/25 | Highest mission alignment + leverage | Zero evidence, no code, no prototype, high risk |

**Recommendation:** MAOS as flagship (it's the execution surface and already works). Trustcard as secondary (bounded, evidence-rich). Iowa as secondary (strategic, but needs evidence fast). This leaves the experiment slot for claude-yolo-vibes (capped at 5h/week).

### Proposed active portfolio

| Slot | Project | Weekly budget | Rationale |
|---|---|---|---|
| **Flagship** | MAOS | 8–10h | Daily driver, enables all verification |
| **Secondary 1** | trustcard | 4–6h | Published, differentiated, needs real consumer |
| **Secondary 2** | Iowa Secure Edge Vision | 6–8h | Highest strategic value, needs evidence fast |
| **Experiment** | claude-yolo-vibes | 5h (capped) | 30-day bounded campaign, community reputation |
| **Parked** | trajectory-os | 3–5h (onboarding only) | Unblock the MAOS integration, then park until MAOS consumes it |
| **Parked** | mcp-inference-gateway | 2–3h | Natural trustcard consumer — unpark when trustcard needs it |
| **Parked** | Wisephone II | 2–3h (public arch only) | Device-blocked |
| **Parked** | lan-vault, grokhack, infoscanner, local-operator, jarvis-word, claude-local, repo-archaeologist | 0–1h each | Maintenance / utility / stable |
| **Unresolved** | nevernote | 0h | Needs user input on purpose |

**Total active weekly budget:** ~23–29h (flagship + 2 secondaries + experiment + trajectory onboarding). This fits a realistic 25–30h/week focused-work capacity.

---

## Acceptance Criteria Check

| Criterion | Status |
|---|---|
| All 10 projects are listed | **Done** — 14 active code projects + 2 non-code initiatives enumerated (the "10" was an undercount) |
| Every project has a next action, done criterion, kill/pause criterion, and evidence path | **Done** — all 14 code projects have all four fields. nevernote's done/kill criteria are marked unknown (needs user input). |
| Each project is classified as flagship candidate, secondary candidate, experiment candidate, parked, or unresolved | **Done** — all classified. |
| A provisional portfolio score uses mission alignment, leverage, evidence potential, finishability, and opportunity cost | **Done** — scored 1–5 on all five axes, total /25. Scores marked provisional pending user calibration. |

---

## Gaps Requiring User Input

1. **nevernote purpose** — what is it supposed to be? Currently unresolved.
2. **Flagship choice** — MAOS vs. trustcard vs. Iowa. Recommendation is MAOS, but this is David's call.
3. **Wisephone device access** — is a Wisephone II available? If not, park all device-dependent work.
4. **Score calibration** — the portfolio scores are provisional. David should adjust weights based on his priorities.
5. **Time budget validation** — 25–30h/week assumes a healthy schedule. DAV-18 (personal baseline) will refine this.

---

## Evidence Provenance

- Git commit dates: `git log -1 --format='%ci'` in each repo (2026-08-15)
- Linear projects: `list_projects` + `get_project` MCP calls (2026-08-15)
- Linear issues: `list_issues` with `assignee=me` (2026-08-15, 50 issues)
- MAOS umbrella: `uv run app umbrella` (2026-08-15)
- Portfolio docs: `~/mac-ai-os/docs/portfolio-decision-2026-08.md`, `~/mac-ai-os/docs/ceo/portfolio-vision.md`
- 90-Day Reset spec: Linear project `bfcd1796-7561-4e90-a337-2d56ddd473e6`
- Project READMEs: read directly from each repo
