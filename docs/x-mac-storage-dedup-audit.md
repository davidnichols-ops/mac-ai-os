# X-MaC Storage & Dedup Audit — 2026-08-03

## Storage Intelligence (Capability #1)

### Current State
- Disk engine walks directories with WalkDir — works but no incremental scanning
- Category breakdown implemented (caches, dev artifacts, media, archives, apps, backups, duplicates, unknown)
- `--explain` and `--group-by` flags implemented and tested
- No benchmark suite, no profiling, no evidence of 500K+ file performance

### Gaps to Close for v1

| Gap | Priority | Effort | Task # |
|-----|----------|--------|--------|
| Build 500K+ file benchmark suite | HIGH | 2 days | 146 |
| Implement incremental filesystem updates | HIGH | 3 days | 148 |
| Benchmark repeated scans (second scan faster) | MEDIUM | 1 day | 147 |
| Add progress reporting (real, not fake %) | MEDIUM | 1 day | 150 |
| Add cancellation support | MEDIUM | 1 day | 151 |
| Add resumable scanning | LOW | 2 days | 152 |
| Category-level storage accounting | DONE | — | 154 |
| Storage explanations (not just rankings) | DONE | — | 153 |
| Reclaimable-space confidence score | MEDIUM | 1 day | 155 |
| Measure actual reclaimed storage | MEDIUM | 1 day | 159 |
| Track false-positive cleanup recommendations | MEDIUM | 1 day | 160 |
| Safe exclusions (system-critical paths) | HIGH | 1 day | 161 |
| "Never touch without explicit confirmation" policy | HIGH | 1 day | 162 |

### Benchmark Plan
1. Create synthetic 500K file tree (mix of small files, large files, caches, dev artifacts)
2. Measure: startup time, scan time, memory usage, graph construction time
3. Measure on real hardware (M4 Mac, not synthetic)
4. Second scan should exploit prior state (incremental)
5. Identify the largest bottleneck and optimize it

## Duplicate Detection (Capability #2 component)

### Current State
- BLAKE3 exact-duplicate engine: implemented, needs completion
- Perceptual image hashing: not implemented (needs DCT-based pHash)
- DuplicateEngine exists but not wired into CLI as `xmac dedup`
- No false-positive evaluation
- No configurable similarity thresholds
- No duplicate review UI
- No deterministic canonical-file policy

### Gaps to Close for v1

| Gap | Priority | Effort | Task # |
|-----|----------|--------|--------|
| Complete BLAKE3 exact-duplicate engine | HIGH | 1 day | 163 |
| Wire DuplicateEngine into CLI (`xmac dedup`) | HIGH | 1 day | 11, 172 |
| Separate exact vs perceptual duplicates | HIGH | 1 day | 165 |
| Add false-positive evaluation for perceptual | MEDIUM | 1 day | 166 |
| Configurable similarity thresholds | MEDIUM | 1 day | 167 |
| Deterministic canonical-file policy | MEDIUM | 1 day | 169 |
| "Keep all" option (never force deletion) | HIGH | 0.5 day | 170 |
| Benchmark on large photo libraries | LOW | 1 day | 171 |
| Perceptual image hashing (DCT pHash) | v2 | 3 days | 164, 13 |
| Duplicate review UI | v2 | 5 days | 168 |

### Dedup Implementation Plan
1. Complete BLAKE3 exact-duplicate engine (fast, deterministic, independently testable)
2. Wire into CLI as `xmac dedup` — make it independently usable
3. Add "keep all" option — system should never make user fight to preserve data
4. Add deterministic canonical-file policy (explain why one copy is recommended)
5. Add configurable similarity thresholds with visible tradeoff explanation
6. Benchmark on real photo library (10K+ images)
7. v2: Add perceptual image hashing (DCT-based pHash) for "similar photos"
