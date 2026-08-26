# Portfolio decision — 2026-08-06

## Decision

Use MAOS as the daily execution surface, Trajectory OS as the personal
decision/context substrate, and Infernet as a bounded infrastructure project.
Do not make cloud GPU spend or a new distributed-compute deployment a
prerequisite for the personal system.

## Evidence

- MAOS has a working agent loop, safety gate, trajectory recording, Slack
  bridge, and an optional Trajectory context adapter.
- Trajectory OS has the durable personal knowledge graph and provider-neutral
  context packs, but the local database is still empty until onboarding.
- Security validation has produced hypotheses and coverage gaps, not confirmed
  vulnerabilities. The highest-value next security work is authorization,
  parser/property, and agent-permission testing.
- Infernet's full test collection currently aborts during the local PyTorch /
  OpenMP import path. It needs reproducible environment and failure-injection
  evidence before being treated as production infrastructure.
- GitHub access is operational. X-MaC PR #9 remains open, but its current
  individual checks are passing; the aggregate PR state is stale/unstable.
  Infernet is still private, so it is not yet a public portfolio artifact.

## Compute policy

1. Use the Apple Silicon/MLX path for MAOS development and private context.
2. Rent GPU capacity only for a measured experiment with a defined output:
   benchmark, training run, or distributed-inference demonstration.
3. Prefer short-lived on-demand or serverless capacity; do not reserve or buy
   hardware until utilization and throughput are measured.
4. Record cost per useful output (for example, cost per million generated
   tokens or completed benchmark), not hourly price alone.

## Exit gates

- Trajectory onboarding completed and a daily context pack contains actual
  goals, constraints, and priorities.
- One MAOS `do` task consumes that context and produces a verified result.
- Slack `#maos-bridge` command round trip verified; move to `#general` only
  after the bot is explicitly invited there.
- Infernet passes a reproducible environment check and a 3-node benchmark.
- At least one authorized security campaign demonstrates a concrete result,
  rather than only a source-level hypothesis.

## GitHub execution order

1. Reconcile X-MaC PR #9's aggregate status and merge it only after reviewing
   the current check run, rather than treating the stale aggregate as a new
   engineering project.
2. Make Infernet reproducible locally, then decide whether its README and
   benchmark are ready for public visibility.
3. Keep MAOS and Trajectory OS focused on the integrated daily-use vertical
   slice; do not open another broad feature branch.
