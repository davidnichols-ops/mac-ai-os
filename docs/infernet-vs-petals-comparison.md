# Infernet vs Petals — Architecture Comparison

## Decision Required

Two distributed inference architectures are being maintained. This document
compares them to determine which receives active development and which is
archived or maintained.

## Summary

| Dimension | Infernet | Petals Fork |
|-----------|---------|-------------|
| **Origin** | Clean-slate (5 commits, Jul 2026) | Fork of bigscience-workshop/petals (19 commits ahead) |
| **Language** | Python | Python |
| **Model partitioning** | Block-range pipeline parallelism | Tensor parallelism + pipeline parallelism |
| **Transport** | Custom TCP protocol + OCP transport (new) | libp2p (hivemind) |
| **Hardware** | Heterogeneous (Mac MLX + Windows PyTorch + llama.cpp) | GPU-only (NVIDIA via bitsandbytes) |
| **Trust** | Trust verification (keys.py) | Full trust layer (Ed25519 descriptors, TOFU, Gate 1, anomaly detection) |
| **KV cache** | Session-based, 10x decode speedup claimed | Standard hivemind caching |
| **Tests** | 4 test files, limited coverage | 208 tests + 17 adversarial |
| **Docs** | README + ARCHITECTURE.md (374 lines) + SECURITY.md + ROADMAP.md | README (16KB) + 10 docs + ROADMAP + adversarial report |
| **Maturity** | Early MVP (Phase 1) | 3 frozen releases (v0.1-v0.3) |
| **Differentiation** | Apple Silicon + heterogeneous hardware + quantization preservation | Trust layer for decentralized inference |

## Detailed Comparison

### 1. Topology Assumptions

**Petals** assumes:
- Volunteer GPUs with similar capability
- Relatively stable network (hivemind DHT for peer discovery)
- Tensor parallelism across peers (requires low latency)
- NVIDIA GPU + bitsandbytes quantization

**Infernet** assumes:
- Heterogeneous hardware (Mac MLX, Windows PyTorch, llama.cpp)
- LAN-scale networks (192.168.x.x)
- Pipeline parallelism (block-range forward — each node handles a range of layers)
- Quantization preservation (each node runs its own backend at its own quantization)

**Verdict:** Infernet's topology assumptions are more modern (Apple Silicon,
heterogeneous hardware, MoE models). Petals' assumptions were valid in 2022-2023
but are increasingly obsolete with RTX 5090, M4 Ultra, and MoE architectures.

### 2. Model Execution Granularity

**Petals:** Tensor parallelism — splits individual layers across GPUs. Requires
high-bandwidth, low-latency interconnect. Each peer handles a slice of every
forward pass.

**Infernet:** Block-range pipeline parallelism — each node handles a contiguous
range of transformer blocks. Node 1 does blocks 0-15, Node 2 does blocks 16-31,
etc. This is coarser-grained, more tolerant of network latency, and works with
heterogeneous backends.

**Verdict:** Infernet's approach is better suited for heterogeneous hardware and
higher-latency networks. Petals' approach is better for homogeneous GPU clusters
with fast interconnects.

### 3. Trust and Identity

**Petals fork:** Has a full trust layer — Ed25519 signed capability descriptors,
TOFU pinning, Gate 1 verification (UNKNOWN/KNOWN/CHANGED/REJECTED), trust-aware
routing, PeerID binding, anomaly detection. 208 tests + 17 adversarial. This is
the most mature part of the fork.

**Infernet:** Has `trust/keys.py` — basic key loading. No descriptor system, no
TOFU, no Gate 1, no anomaly detection. Trust is minimal.

**Verdict:** Petals fork has a significantly more mature trust layer. Infernet
should adopt or adapt this rather than rebuilding from scratch.

### 4. Failure Recovery

**Petals:** Hivemind handles peer departure via DHT re-routing. The trust fork
adds anomaly detection (NaN/Inf/gross magnitude → ban + retry). Well-tested with
17 adversarial scenarios.

**Infernet:** No failure injection test suite. No documented recovery behavior.
This is a gap (task #108).

**Verdict:** Petals fork is far more mature in failure handling.

### 5. Hardware Heterogeneity

**Petals:** GPU-only. Requires NVIDIA GPU + bitsandbytes. Cannot run on Apple
Silicon. This is a fundamental limitation of the hivemind/torch architecture.

**Infernet:** Explicitly designed for heterogeneous hardware. MLX backend for
Mac, PyTorch backend for Windows/Linux, llama.cpp as fallback. This is its
primary differentiator.

**Verdict:** Infernet wins decisively on hardware heterogeneity. This is the
reason it exists.

### 6. Performance Claims

**Petals:** No formal benchmarks in the fork. Upstream Petals has benchmarks but
they're for GPU-only configurations.

**Infernet:** Claims "10x decode speedup" from session-based KV cache and
"working end-to-end" 3-node GPT-OSS 20B. No benchmark code to substantiate
these claims (task #106).

**Verdict:** Neither has formal benchmarks. Both need this before publishing.

## Recommendation

### Decision: **Infernet receives active development. Petals fork is archived as research input.**

**Rationale:**

1. **Infernet's hardware heterogeneity is the differentiator that matters.**
   Apple Silicon + Windows + llama.cpp support is something Petals cannot offer.
   This is the reason Infernet exists and it's a real problem worth solving.

2. **Petals' trust layer should be ported to Infernet.** The trust layer
   (descriptors, TOFU, Gate 1, anomaly detection) is the most valuable part of
   the Petals fork and is not coupled to hivemind's tensor parallelism. It should
   be adapted to Infernet's pipeline parallelism architecture.

3. **Petals' topology assumptions are obsolete.** The `REVIVAL-ARCHITECTURE.md`
   document in the Petals fork honestly assesses this. Volunteer GPU networks
   with tensor parallelism don't fit the 2026 hardware landscape.

4. **Maintaining two distributed inference architectures is unsustainable.**
   Task #305 explicitly calls this out. Pick one.

### Action Items

1. **Archive Petals fork** — tag final release, mark as "research project,
   superseded by Infernet". Keep the repo public as a reference.
2. **Port Petals trust layer to Infernet** — the trust modules (`canon.py`,
   `descriptor.py`, `pinstore.py`, `gate1.py`, `anomaly.py`) are self-contained
   and don't depend on hivemind.
3. **Write the "why not just use Petals?" section** for Infernet's README
   (task #121) — answer: heterogeneous hardware, pipeline parallelism, Apple
   Silicon support.
4. **Use Petals' adversarial test patterns** as a template for Infernet's
   failure injection test suite (task #108).
