# Infernet Architecture Review — 2026-08-03

## Architecture Summary

Infernet is a distributed inference substrate for heterogeneous hardware. It
splits large models across multiple machines using pipeline parallelism
(block-range forward), where each node runs its own backend (MLX, PyTorch,
llama.cpp) at its own quantization level.

## Node Roles

| Role | Description | Current Implementation |
|------|-------------|----------------------|
| **Coordinator** | Receives inference request, splits into block ranges, dispatches to nodes | `cli/run_client.py` |
| **Worker** | Executes a range of transformer blocks, returns activations | `node/server.py` |
| **Discovery** | mDNS-based node discovery on LAN | Implemented |
| **Trust** | Verifies node identity and model identity | `trust/keys.py` (minimal) |

## Model Partitioning

Block-range pipeline parallelism:
- Node 1: blocks 0-15 (MLX on Mac)
- Node 2: blocks 16-31 (PyTorch on Windows)
- Node 3: blocks 32-47 (llama.cpp on Linux)
- Activations flow: Node 1 → Node 2 → Node 3 → output

Each node runs its own backend at its own quantization. The coordinator
preserves quantization — it doesn't force all nodes to the same precision.

## Control Plane vs Execution Plane

**Current state:** Not separated. The coordinator both decides where work goes
and manages execution. This needs to be separated (task #110).

**Recommended:**
- **Control plane:** Decides which nodes handle which block ranges, based on
  capability, load, and trust state
- **Execution plane:** Executes the assigned block range and returns activations

## Model-Artifact Identity

**Current state:** Minimal. The cluster doesn't verify that all nodes are
running the same model, quantization, revision, and tokenizer.

**Needed (tasks #111-112):**
- Model identity descriptor: {model_name, revision, tokenizer, quantization, backend}
- Compatibility check before session starts
- Refuse to form cluster when nodes disagree

## KV Cache

Session-based KV cache provides 10x decode speedup (claimed, not benchmarked).
The cache is per-session and per-node — each node caches its own block range's
KV state.

## Failure Modes

**Not tested.** No failure injection test suite exists (task #108).

**Needed:**
- Kill node during prefill → verify clean failure or recovery
- Kill node during decode → verify session recovery
- Introduce latency → measure throughput degradation
- Drop packets → verify retry behavior
- Restart node → verify session resumption

## Telemetry

**Not implemented.** No structured telemetry for inference sessions (task #113).

**Needed:**
- Per-session: node, stage, block range, latency, bytes transferred, tokens
  produced, cache state, errors
- Per-node: load, memory, throughput, uptime

## Minimum Viable Public Infernet (v0.1)

1. Reliably demonstrate distributed inference on 3 heterogeneous machines
2. Reproducible benchmarks (not just "it works")
3. 3-node demo script that anyone can run
4. Architecture README explaining what it is and why
5. Security: .env gitignored, no secrets in code (DONE)

## What to Port from Petals

1. **Trust layer** — descriptor system, TOFU pinning, Gate 1 verification
2. **Anomaly detection** — NaN/Inf/gross magnitude detection in activations
3. **Adversarial test patterns** — template for failure injection tests
4. **Trust-aware routing** — weight nodes by trust state

## What NOT to Port from Petals

1. **Hivemind DHT** — Infernet uses mDNS, which is simpler and LAN-appropriate
2. **Tensor parallelism** — Infernet uses pipeline parallelism
3. **bitsandbytes** — Infernet supports multiple backends, not just NVIDIA
