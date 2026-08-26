# X-MaC v1 Scope Decision — 2026-08-03

## Decision: Three User-Visible Capabilities for X-MaC v1

After surveying the codebase (630 operations, 13 engines, GNN, reasoning layer —
most behind `#[allow(dead_code)]`), the product needs to be reduced to three
coherent, user-visible capabilities that form a complete product story.

### The Three Capabilities

| # | Capability | What the user sees | What it replaces |
|---|-----------|-------------------|-----------------|
| 1 | **System Intelligence** | "Your Mac explained" — a living model of what's on your disk, what changed, what's expensive, what's redundant. Not a snapshot — a model that updates. | The digital twin + storage intelligence + GNN scoring, reduced to what produces user-visible explanations. |
| 2 | **Safe Storage Cleanup** | "Here's exactly what we'll delete, here's why, here's how to undo it." Every destructive action has a preview, a confirmation, and a rollback. | The cleanup engine + dedup + safety system, reduced to one workflow done right. |
| 3 | **Digital-Twin Explanation** | "Show me the evidence." When the system says something is important/expensive/redundant/risky, the user can drill down to the actual files, relationships, and metrics behind the recommendation. | The graph-to-user explanation path + evidence mode, reduced to a transparent reasoning layer. |

### What These Three Capabilities Form

A coherent product: **understand → act → verify**.

1. System Intelligence = **understand** (what's on my Mac?)
2. Safe Storage Cleanup = **act** (reclaim space safely)
3. Digital-Twin Explanation = **verify** (why should I trust this?)

### What Gets Cut, Deferred, or Archived

| Component | Status | Rationale |
|-----------|--------|-----------|
| GNN inference via CoreML | **v2** | Must outperform rules engine in measurable user-relevant way first (task #137-139). Currently produces no measurable outcome. |
| Reasoning engine (`Vec<String>` outputs) | **v2** | Outputs are not user-visible. Needs a closed loop: AI improves outcomes users can measure. |
| 630 operations across 13 engines | **Reduce to ~30** | Most are `#[allow(dead_code)]` and never seen by users. Pick 3 things and make them best-in-class. |
| GUI intelligence (split runs across disk/GPU/RAM) | **v2** | Premature optimization. First prove the 3 capabilities work. |
| Perceptual image hashing (DCT-based pHash) | **v1** (part of dedup) | Needed for capability #2 — but only exact duplicates + basic similarity, not full GNN scoring. |
| Causal analysis, optimization simulation, regression detection | **v2** | These are reasoning-layer features that don't produce user-visible outcomes yet. |

### Product Truth Table

| Capability | Input | Computation | Output | User Benefit | Measurable Outcome | Failure Mode |
|-----------|-------|-------------|--------|-------------|-------------------|-------------|
| System Intelligence | Filesystem scan | Walk dirs, hash files, build graph, categorize | "Dev artifacts: 23 GB (18%). Caches: 8 GB. Duplicates: 4 GB." | User knows what's on their disk without manual investigation | Time to answer "what's taking up space?" < 30s on 500K files | Scan timeout, incomplete results, wrong categorization |
| Safe Storage Cleanup | User selects categories to reclaim | Preview changes, snapshot state, execute deletions, verify results | "Deleted 812 files, reclaimed 4.2 GB. Snapshot saved. Undo available for 7 days." | User can safely reclaim space without fear of breaking something | Reclaimed space matches predicted space within 5% | Failed deletion, partial cleanup, snapshot corruption, false positive |
| Digital-Twin Explanation | User clicks "why?" on a recommendation | Traverse graph, collect evidence, format explanation | "These 3 caches account for 83% of reclaimable storage. Largest: Xcode DerivedData (3.1 GB), last modified 2 days ago." | User trusts the recommendation because they can see the evidence | User can answer "why does the system think this?" without reading code | Missing evidence, stale graph, broken drill-down |

### Definition of Done — X-MaC v1

| Criterion | Status | Evidence Required |
|-----------|--------|------------------|
| Installation | Pending | User can install via DMG or Homebrew without fighting Gatekeeper |
| Scan | Pending | Completes a 500K+ file scan in < 60s, with progress reporting |
| Measurable result | Pending | User sees categorized storage breakdown with actionable recommendations |
| Safe preview | Pending | Every destructive recommendation shows exactly what will change |
| Rollback | Pending | At least one destructive workflow is genuinely reversible (snapshot → execute → verify → undo) |
| Benchmark | Pending | Baseline performance numbers on real hardware (not synthetic) |
| Signing | Pending | App is signed with Developer ID + hardened runtime |
| Notarization | Pending | App is notarized — normal user can install without Gatekeeper warning |
| Documentation | Pending | README explains problem, safety model, installation, first scan, expected results |

### What "Done" Means for Each Capability

**Capability 1 (System Intelligence) is done when:**
- A 500K+ file scan completes in < 60s
- Results are categorized (dev artifacts, caches, media, archives, apps, backups, duplicates, unknown)
- User can answer "what's taking up space?" without manual investigation
- Incremental updates work (second scan is faster than first)

**Capability 2 (Safe Storage Cleanup) is done when:**
- Every destructive recommendation has a preview
- At least one workflow has real rollback (snapshot → execute → verify → undo)
- Post-action verification confirms expected files were removed
- Actual reclaimed space is measured and reported
- False-positive cleanup recommendations are tracked

**Capability 3 (Digital-Twin Explanation) is done when:**
- Every AI-generated recommendation has a "why?" explanation
- Explanations cite actual filesystem or graph evidence
- User can drill down from recommendation → evidence → files
- A "show me the evidence" mode exists

### Immediate Next Steps

1. **Merge or split PR #9** (36-commit digital twin integration) — this is the single biggest open loop
2. **Remove dead code** — survey the 274 `#[allow(dead_code)]` annotations, keep what serves the 3 capabilities, remove the rest
3. **Build the benchmark harness** — 500K+ file scan, measure throughput, identify bottlenecks
4. **Implement the first real rollback workflow** — pick one narrow destructive operation and make it genuinely reversible
5. **Sign and notarize** — without this, no normal user can install
