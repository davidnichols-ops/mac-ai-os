# MAOS Agent Package Contract

This directory implements MAOS's internal agents and external-agent handoff.
The authoritative device workflow is `~/AGENTS_WORKFLOW.md`; the
repository contract is the top-level `AGENTS.md`.

## Start every external run

```bash
cd ~/mac-ai-os
uv run app agent doctor --provider <devin|grokhack|claude-code|codex|local|openrouter> --json
uv run app agent brief --provider <provider> --task "<task>"
uv run app agent start --provider <provider> --role operator --task "<task>"
uv run app onboard --json
uv run app refresh --json
```

Close the run only after verification:

```bash
uv run app agent end <agent-id> --summary "<what changed>" \
  --result "<tests and live evidence>" --workspace "<workspace>"
```

## Secret boundary

- Raw credentials never belong in this directory, tracked configuration,
  prompts, logs, task descriptions, or handoff records.
- MAOS runtime credentials live in the repo's `.env` file with mode
  `600`; only variable names and readiness are exposed by `agent doctor`.
- Devin and Claude use their CLI-managed authentication.
- GrokHack and OpenRouter use `OPENROUTER_API_KEY` from the protected MAOS
  environment.
- Slack uses `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_USER_TOKEN`, and
  `SLACK_TEAM_ID` from the same protected environment.
- Together and DeepInfra training keys remain in a protected `.env` file
  and are not duplicated into tracked configuration.

## Invariants

- Refresh context before planning.
- Existing MAOS tasks outrank newly invented work unless evidence justifies a
  change.
- Preserve unrelated dirty worktree changes.
- Use permissioned tools and approval gates for consequential actions.
- Verification requires executed evidence, not model agreement.
- Record a concise completion summary and evidence before ending the session.
