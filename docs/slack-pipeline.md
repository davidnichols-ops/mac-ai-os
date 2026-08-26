# Slack Data Pipeline

MAOS uses two Slack identities with separate responsibilities:

- The bot token receives events, responds to commands, and posts alerts.
- The user token reads history visible to the installing user. It has no write
  or administration scopes.
- The app token maintains the Socket Mode connection. It cannot read or write
  workspace data.
- Organization-level tokens use `SLACK_TEAM_ID` to target the workspace. An
  organization install alone does not grant access until that workspace is
  selected during installation.

The authoritative app definition is `configs/slack-manifest.json`.

## Data Flow

1. Socket Mode delivers live commands to `app/slack/bridge.py`.
2. Plain messages become commands only in `#maos-bridge` or a direct message.
3. `app/slack/ingest.py` incrementally reads visible conversation history.
4. Messages, threads, channel metadata, users, checkpoints, and run evidence
   are persisted to `data/slack.db`.
5. Changed Slack messages replace the `source=slack` partition in the existing
   MAOS retrieval index.
6. Operational alerts route to `#maos-ops`; digests route to `#daily-brief`.

No Slack history is deleted remotely. The local store can be rebuilt from the
configured history window with `uv run app slack sync --full`.

## Operations

```bash
uv run app slack health
uv run app slack sync
uv run app slack sync --full --json
uv run app refresh --scope slack
uv run app slack send --channel C0BMSQGTCJV "pipeline check"
```

The background bridge syncs every 15 minutes by default. These environment
variables control the pipeline:

```text
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_USER_TOKEN=xoxp-...
SLACK_TEAM_ID=T0BMJEMLC2X
SLACK_CHANNEL=C0BMJKXL38T
SLACK_ALERTS_CHANNEL=C0BMSQGTCJV
SLACK_DAILY_CHANNEL=C0BNPFAB4MP
SLACK_AUDIT_CHANNEL=C0BMSQGTCJV
SLACK_INGEST_ENABLED=true
SLACK_INGEST_INTERVAL_SECONDS=900
SLACK_INGEST_DAYS=30
SLACK_INGEST_INCLUDE_DMS=true
```

## Recovery

1. Apply `configs/slack-manifest.json` in the Slack app manifest editor.
2. For an organization install, select the target workspace in the install
   dialog. Store that workspace's `T...` ID in `SLACK_TEAM_ID`.
3. Reinstall the app whenever OAuth scopes change. Slack reinstall is additive;
   to remove legacy scopes, remove the workspace installation and organization
   approval before reinstalling from the clean manifest.
4. Replace both `SLACK_BOT_TOKEN` and `SLACK_USER_TOKEN` from the new install.
5. Run `uv run app slack health --json` and resolve any reported scope drift.
6. Run `uv run app slack sync --full --json`.
7. Restart `com.maos.slack` and confirm a fresh successful sync in the logs.

Do not add `admin.*` or user write scopes to recover this pipeline. Slack user
tokens inherit the installing user's visibility, so broad read access is still
sensitive even without write permissions.
