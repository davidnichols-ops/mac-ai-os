# Slack Umbrella Control Plane

## Goal

Slack is the human-facing control plane for the computer and the project portfolio. MAOS remains the local authority: it owns permissions, approvals, audit logs, evidence, memory, workflows, and MCP exposure.

Slack is not a second operating system and not an unrestricted shell. It is the console over MAOS.

```text
Slack
  ↕ Socket Mode
MAOS Slack Bridge
  ├── read-only umbrella snapshot
  ├── portfolio vision + delegation
  ├── repository / GitHub evidence
  ├── custom CLI and MCP inventory
  ├── telemetry, network, tasks, workflows
  └── approved actions through MAOS permissions
```

## Commands

| Slack command | Purpose | Mutability |
| --- | --- | --- |
| `umbrella` | Computer, portfolio, repositories, CLIs, MCP configs, docs | Read-only |
| `portfolio` | Alias for `umbrella` | Read-only |
| `status` | System, network, tasks, active workspace | Read-only |
| `tasks` | Pending delegation queue | Read-only |
| `digest` | Current operational digest | Read-only |
| `context` | Full bounded MAOS context packet | Read-only |
| `ask <question>` | Local knowledge/RAG query | Read-only |
| `do <goal>` | Bounded agentic work | Risk-limited |
| `run <command>` | Shell action | Approval-gated by risk |
| `task done <id>` | Complete a tracked task | Audited action |

## Umbrella snapshot

`maos umbrella` is intentionally evidence-backed and deterministic. It reports:

- registered MAOS workspaces;
- pending tasks;
- local `~/Projects` repositories, branches, origins, and dirty state;
- custom executables under `~/bin` and `~/.grok/bin`;
- MCP/config surfaces under the supported local config roots;
- MAOS documentation inventory;
- current portfolio focus when recorded.

It does not expose secrets, execute commands, or infer repository health from filenames. GitHub freshness and tests are separate validation jobs.

## Notification policy

The bridge posts only operationally useful events by default:

- network interruption/degradation;
- workflow crashes;
- denied actions;
- bridge startup/shutdown;
- explicit CEO briefs and milestone digests.

High-volume file/index/model events stay in local MAOS storage unless explicitly promoted. Slack should remain readable enough that David can see the decision signal.

## Agent handoff pattern

Agents must read:

1. `AGENTS.md`;
2. `docs/ceo-operating-system.md`;
3. `docs/ceo/portfolio-vision.md`;
4. this document;
5. `uv run app onboard --json`;
6. `umbrella` output and repository evidence.

A handoff must state:

- owner;
- why now;
- deliverable;
- verification command;
- evidence artifact;
- blockers and uncertainty;
- next decision owner.

## Security boundary

The Slack allowlist, MAOS safety gate, ActionExecutor, approval queue, decision log, and evidence-producing guards remain mandatory. Do not add a Slack shortcut that bypasses them. The target is a powerful console with a narrow authority boundary, not a remote root prompt.
