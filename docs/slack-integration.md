# Slack Integration

Remote management and alerts for MAOS via Slack. Bidirectional — send
commands from Slack to your Mac, and receive push alerts when MAOS detects
issues.

Uses **Slack Socket Mode** (WebSocket) — no public endpoint, no ngrok tunnel,
no exposed ports. The Slack workspace connects to your Mac, not the other way
around.

---

## Architecture

```
Slack Workspace
    ↕ (Socket Mode — WebSocket)
MAOS Slack Bridge (app/slack/)
    ├── SlackCommandRouter  →  ActionExecutor / Context API
    ├── SlackNotifier       →  EventBus → Slack channel
    ├── Personal memory     →  SQLite + RAG + verified LANBlockchain archive
    └── DigestBuilder       →  scheduled summaries
    ↓
MAOS Core (shell, telemetry, network, tasks, RAG, agents)
```

### Security layers

1. **User allowlist** — only specified Slack user IDs can send commands.
   Empty allowlist = allow all workspace members.
2. **Risk classification** — every command is classified as LOW / MEDIUM /
   HIGH / CRITICAL. HIGH and CRITICAL commands require explicit approval
   (reply with `approve <command>` in Slack).
3. **Rate limiting** — max 1 command per 2 seconds per user.
4. **Audit log** — every command is logged via ActionExecutor → `action_log`
   and `decisions` tables.
5. **Agentic loop guardrail** — `do <goal>` is limited to LOW/MEDIUM risk
   goals. HIGH/CRITICAL agentic loops are blocked (use `run` with approval
   instead).
6. **Safe memory action** — `remember`, `note`, and `index` are allowlisted
   personal-memory writes. They execute through `ActionExecutor`, record
   provenance, index locally, and archive to LANBlockchain only after a
   remote SHA-256 verification. Arbitrary shell and agentic actions remain
   approval-gated or read-only.

---

## Setup Guide

### Step 1: Create a Slack App

1. Go to https://api.slack.com/apps and click **Create New App**.
2. Choose **From scratch**.
3. Name it "MAOS" (or whatever you like).
4. Select your workspace.

### Step 2: Enable Socket Mode

1. In your app settings, go to **Socket Mode** (left sidebar).
2. Toggle it **On**.
3. Give it a token name (e.g. "maos-socket").
4. Copy the **App-Level Token** (starts with `xapp-`). This is your
   `SLACK_APP_TOKEN`.

### Step 3: Add Bot Token Scopes

1. Go to **OAuth & Permissions** (left sidebar).
2. Add these **Bot Token Scopes**:
   - `chat:write` — send messages
   - `channels:history` — read channel messages
   - `channels:read` — list channels
   - `app_mentions:read` — detect @maos mentions
   - `im:history` — read DMs
   - `im:write` — send DMs
   - `commands` — slash commands

### Step 4: Enable Event Subscriptions

1. Go to **Event Subscriptions** (left sidebar).
2. Toggle **On**.
3. Under **Subscribe to bot events**, add:
   - `message.channels` — messages in channels the bot is in
   - `app_mention` — when someone @mentions the bot

4. With Socket Mode enabled, you do **not** need a Request URL. Slack
   connects via WebSocket instead.

### Step 5: Create a Slash Command (optional)

1. Go to **Slash Commands** (left sidebar).
2. Create a command: `/maos`
3. Request URL: not needed with Socket Mode.
4. Description: "Send a command to MAOS"

### Step 6: Install the App

1. Go to **Install App** (left sidebar).
2. Click **Install to Workspace**.
3. Copy the **Bot User OAuth Token** (starts with `xoxb-`). This is your
   `SLACK_BOT_TOKEN`.

### Step 7: Create the #maos Channel

1. In Slack, create a channel named `#maos` (or use any existing channel).
2. Invite the MAOS bot: `/invite @maos`

### Step 8: Get Your Slack User ID (for allowlist)

1. In Slack, click your profile picture → **Profile**.
2. Click the three dots → **Copy member ID**.
3. This is your Slack user ID (starts with `U`).

### Step 9: Configure MAOS

Add the tokens to your shell environment (in `~/.zshrc` or equivalent):

```bash
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export SLACK_APP_TOKEN="xapp-your-app-token"
# Optional:
export SLACK_CHANNEL="#maos"
export SLACK_ALLOWLIST="U12345678,U87654321"  # comma-separated user IDs
```

Or set them in `configs/user.yaml`:

```yaml
slack:
  enabled: true
  channel: "#maos"
  allowlist:
    - "U12345678"
    - "U87654321"
  require_approval: true
  digest_time: "09:00"  # UTC, empty to disable
  alert_types:
    - NETWORK_INTERRUPTED
    - NETWORK_DEGRADED
    - WORKFLOW_CRASHED
```

**Never commit tokens to YAML.** Use env vars for tokens, YAML for
non-secret config. The guards system flags Slack tokens in diffs.

### Step 10: Install Slack Dependencies

```bash
cd ~/mac-ai-os
uv sync --extra slack
```

### Step 11: Test the Connection

```bash
maos slack status     # verify configuration
maos slack test       # send a test message to Slack
```

### Step 12: Start the Bridge

```bash
# Foreground (blocks terminal, Ctrl+C to stop):
maos slack start

# Background (daemon thread):
maos slack start -d
```

### Step 13: Send Commands from Slack

In the `#maos` channel (or DM the bot):

```
help
status
network
telemetry
tasks
run ls -la /tmp
ask what is MAOS?
remember The next release should include a scoped LANBlockchain reconcile.
index This is another alias for personal memory capture.
vault status
vault reconcile personal
do list all Python files in the current directory
```

Or use the slash command:

```
/maos status
/maos network
```

Or @mention the bot in any channel it's in:

```
@maos status
```

---

## Available Commands

| Command | Description | Risk |
| --- | --- | --- |
| `status` | System + network + tasks summary | LOW |
| `network` | Network health and bandwidth | LOW |
| `telemetry` | CPU, memory, battery, thermal | LOW |
| `tasks` | List pending tasks | LOW |
| `task done <id>` | Complete a task | LOW |
| `run <cmd>` | Run a shell command | varies (HIGH/CRITICAL need approval) |
| `ask <question>` | Search local knowledge (RAG) | LOW |
| `do <goal>` | Run an agentic loop (LOW/MEDIUM only) | LOW/MEDIUM |
| `digest` | Post a daily digest now | LOW |
| `context` | Full context packet (JSON) | LOW |
| `help` | List available commands | LOW |

Natural language questions (ending with `?` or containing what/how/why/when/
where) are auto-routed to `ask`.

---

## Push Alerts

MAOS posts alerts to Slack for these events (configurable via
`alert_types`):

| Event | When | Default |
| --- | --- | --- |
| `NETWORK_INTERRUPTED` | Network went down | ✅ |
| `NETWORK_DEGRADED` | Latency spike or packet loss | ✅ |
| `NETWORK_RECONNECTED` | Network came back | ❌ |
| `WORKFLOW_CRASHED` | A workflow failed | ✅ |
| `WORKFLOW_COMPLETED` | A workflow finished | ❌ |
| `TASK_COMPLETED` | A task was marked done | ❌ |
| `ACTION_DENIED` | An action was blocked by permissions | ✅ |

To customize, set `alert_types` in `configs/user.yaml`:

```yaml
slack:
  alert_types:
    - NETWORK_INTERRUPTED
    - NETWORK_DEGRADED
    - WORKFLOW_CRASHED
```

---

## Daily Digest

Set `digest_time` in `configs/user.yaml` (24h UTC format) to receive a
daily summary:

```yaml
slack:
  digest_time: "09:00"  # 9:00 AM UTC
```

The digest includes:
- Network health (data transferred, interruptions)
- System averages (CPU, memory)
- Task status (pending, completed)
- Event summary

To post a digest on demand:

```bash
maos slack digest
```

---

## CLI Reference

| Command | Description |
| --- | --- |
| `maos slack start` | Start the bridge (foreground) |
| `maos slack start -d` | Start in background (daemon) |
| `maos slack stop` | Stop (use Ctrl+C for foreground) |
| `maos slack status` | Show configuration status |
| `maos slack test` | Send a test message |
| `maos slack send "msg"` | Send a one-off message |
| `maos slack digest` | Post a daily digest now |
| `maos slack config` | Show current configuration |
| `maos slack config --json` | Configuration as JSON |

---

## Privacy Model

- **No data leaves Slack's WebSocket.** Socket Mode is a WebSocket from
  Slack to your Mac. No data is sent to any third party.
- **Tokens are env vars, not YAML.** The guards system flags Slack tokens
  in diffs to prevent accidental commits.
- **Commands are audited.** Every command is logged to `action_log` and
  `decisions` tables with the Slack user ID, command, and result.
- **No message content is stored.** The bridge processes messages
  in-memory and discards them. Only the action result is logged.
- **Allowlist is enforced.** Even if someone else in the workspace sends
  a command, it's rejected unless they're in the allowlist.

---

## Troubleshooting

### "Slack not configured"

Set the env vars:
```bash
export SLACK_BOT_TOKEN="xoxb-..."
export SLACK_APP_TOKEN="xapp-..."
```
Verify with `maos slack status`.

### "Failed to start bridge"

1. Check that `slack_bolt` is installed: `uv sync --extra slack`
2. Check that the app token is correct (starts with `xapp-`).
3. Check that Socket Mode is enabled in your Slack app settings.
4. Check that the bot is invited to the channel: `/invite @maos` in Slack.

### Bot doesn't respond to messages

1. Verify event subscriptions are enabled (`message.channels`, `app_mention`).
2. Verify the bot is in the channel (not just a member — it needs to be
   invited).
3. Check that you're in the allowlist (if set): `maos slack status`.

### "Rate limited"

Wait 2 seconds between commands. The rate limit is per-user, not global.

### HIGH/CRITICAL commands need approval

This is by design. Reply with `approve <command>` in Slack to proceed.
To disable approval (not recommended), set `require_approval: false` in
`configs/user.yaml`.

---

## Running as a Daemon

For always-on operation, create a launchd plist:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.maos.slack</string>
    <key>ProgramArguments</key>
    <array>
        <string>uv</string>
        <string>run</string>
        <string>--project</string>
        <string>~/mac-ai-os</string>
        <string>app</string>
        <string>slack</string>
        <string>start</string>
    </array>
    <key>WorkingDirectory</key>
    <string>~/mac-ai-os</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>SLACK_BOT_TOKEN</key>
        <string>xoxb-...</string>
        <key>SLACK_APP_TOKEN</key>
        <string>xapp-...</string>
    </dict>
</dict>
</plist>
```

Save to `~/Library/LaunchAgents/com.maos.slack.plist` and load:

```bash
launchctl load ~/Library/LaunchAgents/com.maos.slack.plist
```

---

## Files

| File | Purpose |
| --- | --- |
| `app/slack/__init__.py` | Package exports |
| `app/slack/bridge.py` | `SlackBridge` — Socket Mode client |
| `app/slack/auth.py` | `SlackAuth` — allowlist, rate limiting, risk classification |
| `app/slack/commands.py` | `SlackCommandRouter` — maps Slack messages to MAOS actions |
| `app/slack/notifier.py` | `SlackNotifier` — pushes alerts to Slack |
| `app/slack/digest.py` | `build_digest` — daily digest builder |
| `app/config/settings.py` | `SlackConfig` — configuration model |
| `app/runtime.py` | Wires SlackBridge (lazy) |
| `app/cli/commands/slack.py` | `maos slack` CLI |
| `app/cli/app.py` | Registers the CLI subapp |
| `tests/test_slack.py` | Unit tests (mocked Slack API) |
