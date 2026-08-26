# CEO Slack Computer Bridge

## Intended destination

- Workspace: `E0BMNNC3T6J` (`mac ai os`)
- Channel requested: `C0BMSKMUTKK` (`#general`)
- MAOS bot: `maos_local`

## Current state

The bot token is authenticated to the correct workspace and is a member of both `C0BMJKXL38T` (`#maos-bridge`) and `C0BMSKMUTKK` (`#general`).

The target channel is configured in both `.env` and the shell environment.
This is no longer a channel-membership blocker.

## Verification

Verify with:

```bash
cd ~/mac-ai-os
uv run app slack status
uv run app slack test
```

Then set the destination in `~/mac-ai-os/.env`:

```dotenv
SLACK_CHANNEL=C0BMSKMUTKK
```

Do not paste or commit tokens. The existing `.env` is local-only.

## Always-on wiring

After membership and destination are confirmed, install the launch agent:

```bash
mkdir -p ~/Library/LaunchAgents
launchctl unload ~/Library/LaunchAgents/com.maos.slack.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.maos.slack.plist
launchctl print gui/$(id -u)/com.maos.slack
```

The bridge uses Slack Socket Mode, so it can receive commands from the Mac without exposing a public HTTP endpoint. MAOS remains the permission boundary:

- allowlist controls who can interact;
- high/critical commands require approval;
- commands and outcomes are audited;
- agentic `do` goals remain risk-limited.

## Operating contract

Slack is the CEO console, not an unrestricted root shell. Use it for:

- `status`, `network`, `telemetry`, `tasks`, and `digest`;
- `ask` and bounded `do` goals;
- task delegation and decision briefs;
- approval of explicitly reviewed actions.

The bridge must not bypass MAOS safety, approval, audit, or evidence gates.
