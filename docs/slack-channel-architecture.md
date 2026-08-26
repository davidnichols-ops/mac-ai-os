# Slack Channel Architecture — `mac ai os` Workspace

Status: **Proposal — review required.** No Slack APIs called, no channels
created, no tokens changed, no GitHub modified. This is a design document.

Workspace: `E0BMNNC3T6J` (`mac ai os`). Bot: `maos_local`.
Allowlist: single user (`U0BMUCS6USG`). All channels are effectively
single-recipient today; the privacy column below is the policy for when
the workspace grows.

---

## Design principles

1. **Slack is the CEO console, not a root shell.** Commands flow through
   MAOS permissions, approval, and audit. The channel layout enforces that
   mentally — a command channel is not an alert feed is not an audit log.
2. **Separate signal from noise.** Alerts, daily briefs, command responses,
   and audit records have different volumes and different readers. Mixing
   them in `#general` buries the decision signal.
3. **One command channel.** David sends commands in exactly one place. The
   bot responds there. Everything else is push-only.
4. **Project channels route by workspace.** Alerts and context for a
   specific repo go to its project channel, not the command console.
5. **`#general` is for humans.** Demote it from the command console back to
   workspace-wide announcements. The bot does not take commands there.

---

## Channel inventory

### Tier 1 — Command & Control

| Channel | ID | Action | Purpose |
| --- | --- | --- | --- |
| `#maos` | — | **Create** (or rename `#maos-bridge`) | Primary bidirectional command console. David sends `status`, `brief`, `do`, `run`, `ask`; bot responds. The CEO seat. |
| `#maos-alerts` | — | **Create** | Push-only operational alerts: `NETWORK_INTERRUPTED`, `NETWORK_DEGRADED`, `WORKFLOW_CRASHED`, `ACTION_DENIED`. Bot posts, David reads. No commands. |
| `#maos-daily` | `C0BNPFAB4MP` | **Reuse** `#daily-brief` (rename) | Scheduled daily brief + on-demand digest. Read-only. One post per morning, plus `digest` on demand. |
| `#maos-audit` | `C0BMSQGTCJV` | **Reuse** `#maos-ops` (rename) | Action executions, approval decisions, denied actions, bridge start/stop. Append-only audit trail. Read-only. |

### Tier 2 — Project channels (workspace-routed)

| Channel | ID | Action | Routed workspaces | Business unit |
| --- | --- | --- | --- | --- |
| `#trajectory-os` | `C0BNKG96KLN` | Reuse | `mac-ai-os`, `dnichols-ops-` | Platform / Trajectory OS |
| `#x-mac` | `C0BNFA7KWBD` | Reuse | `X-MaC`, `X-MaC-digital-twin` | Product Engineering |
| `#aafp` | `C0BN669C4S3` | Reuse | `aafp`, `aafp-go`, `AAFP-research` | R&D / Networking |
| `#engineering` | `C0BNPDJEJUR` | Reuse | `trustcard`, `jarword`, `repo-archaeologist`, `word_proc` | Security / Internal Tools |
| `#research` | `C0BPFRSEDL0` | Reuse | `cvconform`, `apple-quality-recognition-engine`, `Gargantua` | R&D / CV / Graphics |
| `#ops` | `C0BNPFBH64R` | Reuse | `linkedin-dev-pipeline`, `elonwatch`, `elonwatch-mac` | Ops / Monitoring / Growth |

### Tier 3 — External / workspace-wide

| Channel | ID | Action | Purpose |
| --- | --- | --- | --- |
| `#github-prs` | `C0BNPDPFYAV` | Reuse | GitHub PR/issue tracking and review notifications. |
| `#general` | `C0BMSKMUTKK` | **Demote** | Workspace-wide human announcements. Bot posts bridge start/stop here only. No commands. |
| `#random` | `C0BMMB3S11B` | Reuse | Off-topic. Bot does not post. |

### Deprecated

| Channel | ID | Action | Reason |
| --- | --- | --- | --- |
| `#maos-bridge` | `C0BMJKXL38T` | **Rename → `#maos`** or archive | Original bridge channel. Superseded by the dedicated command console. |

---

## Posting rules

| Channel | Who posts | What | Volume |
| --- | --- | --- | --- |
| `#maos` | David + bot | Commands and responses. Bot posts bridge online/offline. | Medium — only when David is active |
| `#maos-alerts` | Bot only | Alertable events from `alert_types` config. No command responses. | Low — only on incidents |
| `#maos-daily` | Bot only | Scheduled brief at `digest_time` + on-demand `digest`/`brief`. | 1/day + on-demand |
| `#maos-audit` | Bot only | `ACTION_EXECUTED`, `ACTION_DENIED`, approval outcomes, bridge lifecycle. | Low-medium |
| `#trajectory-os` | Bot only | Workspace-routed alerts + `project`/`trajectory` briefs for MAOS work. | Low |
| `#x-mac` / `#aafp` / `#engineering` / `#research` / `#ops` | Bot only | Workspace-routed alerts for that business unit. | Low per channel |
| `#github-prs` | Bot / GitHub app | PR and issue notifications. | Variable |
| `#general` | Humans + bot (lifecycle only) | Announcements. Bot posts only bridge start/stop. | Low |
| `#random` | Humans only | Off-topic. | — |

**Routing rule:** the notifier's `WORKSPACE_CHANNELS` map already routes
alerts by workspace. Alerts for unmapped workspaces fall back to
`#maos-alerts`, not `#general`.

---

## Ownership & privacy

| Channel | Owner | Privacy | Sensitivity |
| --- | --- | --- | --- |
| `#maos` | David | Private (invite-only) | High — contains command history and agent responses |
| `#maos-alerts` | David | Private | Medium — operational state, no secrets |
| `#maos-daily` | David | Private | Low — summaries, no raw data |
| `#maos-audit` | David | Private | High — action log with user IDs and command text |
| Project channels | David | Private | Medium — repo state and task context |
| `#github-prs` | David | Private (default) | Medium — public repo metadata |
| `#general` | David | Workspace-wide | Low |
| `#random` | David | Workspace-wide | Low |

**Secrets policy:** no tokens, API keys, file paths under `~/.ssh`, or
`.env` contents are ever posted to any channel. The guards system flags
Slack tokens in diffs. The notifier formats events and discards message
content — only action results are logged locally.

---

## Migration plan: `#maos-bridge` / `#general` → new architecture

The portfolio decision (2026-08-06) set the exit gate: "Slack `#maos-bridge`
command round trip verified; move to `#general` only after the bot is
explicitly invited there." The bot is now in `#general` and
`SLACK_CHANNEL=C0BMSKMUTKK`. This proposal replaces that interim state with
a purpose-built layout.

### Phase 1 — Create the command console (no code change)

1. In Slack, create `#maos` (or rename `#maos-bridge` → `#maos`).
2. Invite `@maos_local` to `#maos`.
3. Create `#maos-alerts`. Invite `@maos_local`.
4. Post a pinned message in `#maos` with the command reference.

### Phase 2 — Repoint the default channel (config only)

5. Update `~/mac-ai-os/.env`:
   ```dotenv
   SLACK_CHANNEL=C0BNXXXXXXX   # the new #maos channel ID
   ```
6. Verify: `uv run app slack status` shows the new channel.
7. Verify: `uv run app slack test` posts to `#maos`.

### Phase 3 — Split alerts from commands (code change)

8. Add `alerts_channel` to `SlackConfig` (default: same as `channel`).
9. Update `SlackNotifier.post_alert` to post to `alerts_channel` instead of
   the workspace-routed or default channel for unmapped workspaces.
10. Update `WORKSPACE_CHANNELS` fallback from `self.channel` to
    `self.alerts_channel`.
11. Set `SLACK_ALERTS_CHANNEL` in `.env` to the `#maos-alerts` ID.
12. Verify: trigger a test alert (e.g. `maos slack send "test alert"` to the
    alerts channel) and confirm it lands in `#maos-alerts`, not `#maos`.

### Phase 4 — Rename existing channels (Slack UI)

13. Rename `#daily-brief` → `#maos-daily`. Update `CHANNEL_IDS` in
    `app/slack/notifier.py`.
14. Rename `#maos-ops` → `#maos-audit`. Update `CHANNEL_IDS`.
15. Archive `#maos-bridge` if it was not renamed in Phase 1.

### Phase 5 — Demote `#general`

16. Remove `#general` from the command routing. The bot no longer accepts
    commands there — only `app_mention` in `#maos` and DMs.
17. The bot posts bridge start/stop to `#general` only (workspace visibility).
18. Update `docs/slack-integration.md`, `docs/ceo-slack-computer-bridge.md`,
    and `docs/slack-umbrella.md` to reference `#maos` as the command channel.

### Phase 6 — Update docs

19. Update this document's status from **Proposal** to **Active**.
20. Update `docs/slack-integration.md` Step 7 to create `#maos` (not `#maos`).
21. Update `docs/ceo-slack-computer-bridge.md` destination channel.

**No token changes, no GitHub changes, no Slack API calls from MAOS.** All
channel creation and renaming is manual in the Slack UI by David. The code
changes in Phase 3 are config + one notifier method.

---

## Morning verification checklist

Run this at the start of each day (or have the morning workflow run it and
post results to `#maos-daily`).

```
# 1. Bridge is online and in the right channels
uv run app slack status
#    → enabled: true, channel: #maos, alerts_channel: #maos-alerts

# 2. Command round-trip works
uv run app slack test
#    → message appears in #maos

# 3. Alert routing is correct (post to alerts channel, not command channel)
uv run app slack send --channel "#maos-alerts" "morning alert routing check"
#    → message appears in #maos-alerts only

# 4. Daily brief posts to #maos-daily, not #maos
uv run app slack digest
#    → digest appears in #maos-daily

# 5. Bot is not accepting commands in #general
#    (manual: send "status" in #general, confirm no response)
#    (manual: send "status" in #maos, confirm response)

# 6. Audit channel received last night's actions
#    (manual: check #maos-audit for ACTION_EXECUTED / ACTION_DENIED entries)

# 7. No secrets leaked in any channel
#    (manual: scan #maos, #maos-alerts, #maos-audit for xoxb-/xapp-/ghp_/
#     AKIA patterns — should be zero)
```

**Pass criteria:** steps 1–4 return success from the CLI. Steps 5–7 are
manual visual checks. If any step fails, escalate via
`uv run app onboard escalate "Slack channel verification" "<details>"`.

---

## Relationship to Trajectory OS

Trajectory OS is the personal decision/context substrate. MAOS reads its
context pack read-only (`trajectory context daily --format markdown`).
Slack does not talk to Trajectory OS directly. The `trajectory` Slack
command synthesizes MAOS + Trajectory state via Nemotron and posts the
brief to `#maos` (command response) or `#maos-daily` (scheduled).

The `#trajectory-os` project channel is for MAOS-internal workspace alerts
and project briefs about the `mac-ai-os` and `dnichols-ops-` repos. It is
not a Trajectory OS control channel.

---

## What this proposal does NOT do

- Does not call any Slack API.
- Does not create, rename, or archive any channel.
- Does not send any Slack message.
- Does not change `SLACK_BOT_TOKEN` or `SLACK_APP_TOKEN`.
- Does not modify GitHub, repos, or PRs.
- Does not change the allowlist or approval policy.
- Does not bypass MAOS safety, audit, or evidence gates.

All actions are manual by David in the Slack UI, except the Phase 3 code
change which is a standard PR with evidence-producing guards.
