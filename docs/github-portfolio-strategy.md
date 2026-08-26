# GitHub Portfolio Strategy — 2026-08-03

## Coherent Technical Narrative

**"AI infrastructure engineer building trusted, distributed, edge-deployed AI systems."**

Five flagship projects that tell this story:

| # | Project | What it proves | Pinned? |
|---|---------|---------------|---------|
| 1 | **trustcard** | Cryptographic trust infrastructure for MCP servers — Ed25519 signed manifests, TOFU pinning, two-gate enforcement, 514 tests | ✅ Pin |
| 2 | **AAFP** | Post-quantum agent networking protocol — ML-DSA-65 signatures, QUIC transport, 2869 tests, Rust+Go interop | ✅ Pin |
| 3 | **Infernet** | Distributed inference for heterogeneous hardware — Mac MLX + Windows PyTorch, pipeline parallelism, 3-node GPT-OSS 20B | ✅ Pin |
| 4 | **X-MaC** | macOS system intelligence — digital twin, filesystem analysis, safe cleanup with rollback, on-device GNN | ✅ Pin |
| 5 | **apple-quality-recognition-engine** | Edge CV on Apple Silicon — CoreML deployment, production grading, real orchard use case | ✅ Pin |

## What to Remove/Archive from Public Profile

- **cloud-to-butt** / **cloud-to-butt-revolutionized** — jokes, archive
- **jane-street-md5-puzzle** — one-off, archive
- **mif_env_mapper-v1.0.0** — stale, archive
- **geo-flight** — unclear, review
- **book-inventory** — personal, archive or make private
- Any repos with no README, no commits in 6+ months, and no strategic value

## README Rewrites for Flagship Repos

Each flagship repo's README should answer in 10 seconds:
1. What problem does this solve?
2. Why should I care?
3. How do I try it?

### trustcard
- **Problem**: MCP servers can change their tools at any time, potentially introducing destructive capabilities. No trust infrastructure exists.
- **Solution**: Content-addressed tool identity + Ed25519 signed manifests + TOFU pinning + two-gate enforcement.
- **Try it**: `npx mcp-trustcard scan <server-spec>`

### AAFP
- **Problem**: Agent-to-agent communication relies on classical cryptography (Noise+Ed25519) that's vulnerable to quantum attacks.
- **Solution**: Post-quantum protocol with ML-DSA-65 signatures, QUIC transport, capability-based DHT.
- **Try it**: `cargo test --workspace` (2869 tests)

### Infernet
- **Problem**: Large models can't run on a single machine, but existing distributed inference assumes homogeneous NVIDIA GPUs.
- **Solution**: Pipeline parallelism across heterogeneous hardware (Mac MLX, Windows PyTorch, llama.cpp).
- **Try it**: `python scripts/orchestrate_3node.py`

### X-MaC
- **Problem**: macOS lacks a trustworthy system intelligence tool that explains what's on your disk and helps you clean safely.
- **Solution**: Digital twin + filesystem graph + safe cleanup with rollback.
- **Try it**: `cargo run -- disk --explain`

### apple-quality-recognition-engine
- **Problem**: Apple sorting on small orchards is manual and inconsistent.
- **Solution**: Edge CV on Apple Silicon — CoreML model grades apples by quality and defect.
- **Try it**: See the live demo video

## Architecture Diagrams to Add

- **trustcard**: manifest identity → capability identity → policy evaluation → agent identity → authorization → invocation → outcome
- **AAFP**: node roles, model partitioning, block-range execution, transport, KV-cache, session lifecycle
- **Infernet**: 3-node pipeline parallelism diagram, KV cache flow, trust verification
- **X-MaC**: digital twin architecture, safety state machine, scan → explain → cleanup → verify flow

## Upstream Contributions to Feature

Create a `CONTRIBUTIONS.md` in the portfolio repo listing:
- ultralytics: DDP callback serialization fix (#25125)
- coremltools: numerically stable log_softmax/logcumsumexp (#2763), softplus decomposition (#2756)
- roboflow-python: YOLO data.yaml fix (#506), upload result fix (#502), opencv pin relax (#501), class ordering (#493)
- onnxruntime: EP fallback detection docs (#29921)
- trippy: privacy settings in report output (#1844)
- nano-spawn: stdin string input (#116)

## Action Items

1. Pin the 5 flagship repos on GitHub profile
2. Rewrite READMEs for all 5 flagships (10-second test)
3. Add architecture diagrams to trustcard, AAFP, Infernet, X-MaC
4. Create CONTRIBUTIONS.md with merged PR links
5. Archive 5-10 low-value repos
6. Add short READMEs to any public repo missing one
