# MAOS CEO Operating System

## Purpose

MAOS is the operating layer for David's project portfolio. It observes the machine and repositories, maintains durable context, delegates bounded work to agents, verifies outcomes, and gives David the decision brief.

The CEO layer is not a chatbot persona. It is a repeatable control loop:

```text
observe → form thesis → choose focus → delegate → verify → brief David → learn
```

## Roles

| Role | Responsibility | Must produce |
| --- | --- | --- |
| CEO office | Portfolio sequencing and trade-offs | Thesis, focus, delegations, insight |
| Architecture agent | Turn strategy into a bounded plan | Milestone plan with dependencies and non-goals |
| Implementation agent | Ship one focused slice | Commit, tests, handoff |
| Verification agent | Challenge claims with real evidence | Test/lint/security/readiness evidence |
| Research agent | Reduce external uncertainty | Source-backed finding and recommendation |
| David | Final priority and irreversible decision gate | Proceed, reprioritize, or park decision |

## Delegation contract

Every assignment uses these fields:

- `owner`: named agent or `david`.
- `why`: the strategic reason now.
- `deliverable`: a concrete artifact.
- `verification`: an executable check or review procedure.
- `priority`: relative sequencing value.
- `status`: pending, active, blocked, verified, or parked.

Do not delegate vague goals such as “make it better.” Do not treat two agents agreeing as verification. MAOS must capture evidence from the actual repository or service.

## Portfolio rule

One milestone is active at a time. Other projects are classified as:

1. **Active:** receives implementation bandwidth now.
2. **Dependency:** directly unblocks the active milestone.
3. **Research:** reduces a named uncertainty with a time-box.
4. **Parked:** recorded, but not allowed to compete for attention.

This prevents the portfolio from becoming a collection of impressive unfinished tabs.

## Human insight policy

MAOS should tell David:

- what it observed;
- what it believes and why;
- what is uncertain;
- what it delegated;
- what evidence is required;
- what decision only David can make.

The system may create tasks and recommendations. High-impact or irreversible actions remain gated by the existing permission and approval layers.

## Standard workflow

1. Run `uv run app workspace scan ~/Projects` when inventory is stale.
2. Run `uv run app workspace profile --force` when project understanding is stale.
3. Run `uv run app office vision` to generate `docs/ceo/portfolio-vision.md`.
4. Run `uv run app office delegate` to materialize the queue as MAOS tasks.
5. Agents execute only their assigned bounded deliverable.
6. Run the real repository tests, lint, security checks, and readiness guards.
7. Record the outcome and update the vision.
8. Post the concise vision and blockers to the configured MAOS Slack channel.

## Agent continuation pattern

An agent joining later should read, in order:

1. `AGENTS.md` for repository rules.
2. This document for the operating contract.
3. `docs/ceo/portfolio-vision.md` for the current thesis and queue.
4. `uv run app onboard --json` for durable context and lessons.
5. The relevant repository's status, tests, and recent decisions.

Then it should continue the first unverified delegation. It must not create a competing strategy without recording the trade-off.
