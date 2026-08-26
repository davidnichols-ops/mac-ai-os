"""`maos slack` — Slack integration commands.

Remote management and alerts via Slack. Uses Socket Mode (WebSocket) —
no public endpoint needed.

Commands:
    maos slack start       — start the Slack bridge (blocks)
    maos slack start -d    — start in background (daemon thread)
    maos slack stop        — stop a running bridge
    maos slack status      — show bridge configuration status
    maos slack test        — send a test message to Slack
    maos slack send <msg>  — send a one-off message to Slack
    maos slack digest      — post a daily digest now
    maos slack config      — show current Slack configuration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from app.config.settings import SlackConfig, load_settings
from app.runtime import build_runtime
from app.ui.console import print_panel

slack_app = typer.Typer(
    name="slack",
    help="Slack integration — remote management and alerts.",
    no_args_is_help=True,
)

EXPECTED_BOT_SCOPES = {
    "app_mentions:read",
    "channels:history",
    "channels:join",
    "channels:read",
    "chat:write",
    "commands",
    "groups:history",
    "im:history",
    "mpim:history",
}
EXPECTED_USER_SCOPES = {
    "channels:history",
    "channels:read",
    "groups:history",
    "groups:read",
    "identify",
    "im:history",
    "im:read",
    "mpim:history",
    "mpim:read",
    "reactions:read",
    "users.profile:read",
    "users:read",
}


def _build_bridge():
    """Build a SlackBridge from the current runtime + config."""
    rt = build_runtime()
    from app.slack.bridge import SlackBridge
    bridge = SlackBridge(rt, rt.settings.slack)
    return bridge, rt


def _store_path(config: SlackConfig) -> Path:
    path = Path(config.ingest_db).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path


def _masked(value: str) -> str:
    if not value:
        return "not set"
    if len(value) <= 16:
        return "set"
    return f"{value[:8]}...{value[-4:]}"


def _api_result(call: Any) -> dict[str, Any]:
    try:
        response = call()
        headers = getattr(response, "headers", {}) or {}
        scopes = sorted(
            scope.strip()
            for scope in headers.get("x-oauth-scopes", "").split(",")
            if scope.strip()
        )
        return {
            "ok": bool(response.get("ok")),
            "error": response.get("error"),
            "scopes": scopes,
        }
    except Exception as exc:
        response = getattr(exc, "response", None)
        error = response.get("error") if response is not None else str(exc)
        return {"ok": False, "error": error, "scopes": []}


def _scope_drift(actual: list[str], expected: set[str]) -> dict[str, list[str]]:
    actual_set = set(actual)
    return {
        "missing": sorted(expected - actual_set),
        "unexpected": sorted(actual_set - expected),
    }


@slack_app.command("start")
def start(
    daemon: bool = typer.Option(
        False, "--daemon", "-d",
        help="Run in background (non-blocking). Default: blocks foreground.",
    ),
) -> None:
    """Start the Slack bridge (Socket Mode).

    Requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN env vars to be set.
    The bridge listens for messages in the configured channel and
    posts alerts for MAOS events.

    Without --daemon, this blocks the terminal. Use --daemon for
    background operation (the bridge runs in a thread).
    """
    bridge, rt = _build_bridge()

    if not bridge.is_configured():
        typer.echo("❌ Slack not configured. Set these env vars:")
        typer.echo("  export SLACK_BOT_TOKEN='xoxb-...'")
        typer.echo("  export SLACK_APP_TOKEN='xapp-...'")
        typer.echo("")
        typer.echo("Then optionally set in configs/user.yaml or env:")
        typer.echo("  SLACK_CHANNEL='#maos'")
        typer.echo("  SLACK_ALLOWLIST='U12345,U67890'")
        typer.echo("")
        typer.echo("See docs/slack-integration.md for full setup instructions.")
        raise typer.Exit(1)

    typer.echo(f"Starting Slack bridge on channel {bridge.config.channel}...")
    if daemon:
        ok = bridge.start(block=False)
        if ok:
            typer.echo("✅ Bridge started in background. Send 'help' in Slack to test.")
            # Keep the process alive.
            try:
                import time
                while bridge.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                typer.echo("\nStopping bridge...")
                bridge.stop()
        else:
            typer.echo("❌ Failed to start bridge. Check logs.")
            raise typer.Exit(1)
    else:
        typer.echo("Bridge running (Ctrl+C to stop)...")
        ok = bridge.start(block=True)
        if not ok:
            typer.echo("❌ Failed to start bridge.")
            raise typer.Exit(1)


@slack_app.command("stop")
def stop() -> None:
    """Stop a running Slack bridge.

    Note: this only works for bridges started in the same process.
    For daemon-mode bridges, kill the process.
    """
    typer.echo("Use Ctrl+C to stop a foreground bridge, or kill the process for daemon mode.")


@slack_app.command("status")
def status() -> None:
    """Show Slack bridge configuration status."""
    cfg = load_settings(detect_hardware=False).slack

    lines = [
        f"Enabled:     {cfg.enabled}",
        f"Configured:  {'✅' if (cfg.bot_token and cfg.app_token) else '❌'}",
        f"Workspace:   {cfg.team_id or 'token default'}",
        f"Command:     {cfg.channel}",
        f"Alerts:      {cfg.alerts_channel or cfg.channel}",
        f"Daily:       {cfg.daily_channel or cfg.channel}",
        f"Audit:       {cfg.audit_channel or cfg.alerts_channel or cfg.channel}",
        f"Allowlist:   {len(cfg.allowlist)} user(s)" + (f" ({', '.join(cfg.allowlist)})" if cfg.allowlist else " (allow all)"),
        f"Approval:    {'required for HIGH/CRITICAL' if cfg.require_approval else 'not required'}",
        f"Digest:      {cfg.digest_time or 'disabled'}",
        f"Ingestion:   {'enabled' if cfg.ingest_enabled else 'disabled'} ({cfg.ingest_days} days, every {cfg.ingest_interval_seconds}s)",
        f"Alert types: {', '.join(cfg.alert_types) if cfg.alert_types else 'default set'}",
    ]

    lines.append(f"Bot token:   {_masked(cfg.bot_token)}")
    lines.append(f"App token:   {_masked(cfg.app_token)}")
    lines.append(f"User token:  {_masked(cfg.user_token)}")

    print_panel("\n".join(lines), title="Slack Bridge Status")


@slack_app.command("test")
def test() -> None:
    """Send a test message to Slack to verify the connection."""
    bridge, rt = _build_bridge()

    if not bridge.is_configured():
        typer.echo("❌ Slack not configured. Run `maos slack status` for details.")
        raise typer.Exit(1)

    typer.echo("Sending test message...")
    ok = bridge.send_test_message()
    if ok:
        typer.echo(f"✅ Test message sent to {bridge.config.channel}")
    else:
        typer.echo("❌ Failed to send. Check tokens and channel name.")
        raise typer.Exit(1)


@slack_app.command("send")
def send(
    message: str = typer.Argument(..., help="Message to send"),
    channel: str = typer.Option("", "--channel", "-c", help="Override destination channel"),
) -> None:
    """Send a one-off message to the Slack channel."""
    bridge, rt = _build_bridge()

    if not bridge.is_configured():
        typer.echo("❌ Slack not configured.")
        raise typer.Exit(1)

    target = channel or bridge.config.channel
    ok = bridge.notifier.post_to_channel(target, message)
    if ok:
        typer.echo(f"✅ Sent to {target}")
    else:
        typer.echo("❌ Failed to send.")
        raise typer.Exit(1)


@slack_app.command("digest")
def digest() -> None:
    """Post a daily digest to Slack now."""
    bridge, rt = _build_bridge()

    if not bridge.is_configured():
        typer.echo("❌ Slack not configured.")
        raise typer.Exit(1)

    from app.slack.digest import build_digest
    d = build_digest(rt.db)
    ok = bridge.notifier.post_digest(d)
    if ok:
        typer.echo(f"✅ Digest posted to {bridge.notifier.daily_channel}")
    else:
        typer.echo("❌ Failed to post digest.")
        raise typer.Exit(1)


@slack_app.command("config")
def config(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current Slack configuration (tokens are masked)."""
    cfg = load_settings(detect_hardware=False).slack

    config_dict = {
        "enabled": cfg.enabled,
        "configured": bool(cfg.bot_token and cfg.app_token),
        "team_id": cfg.team_id,
        "command_channel": cfg.channel,
        "alerts_channel": cfg.alerts_channel or cfg.channel,
        "daily_channel": cfg.daily_channel or cfg.channel,
        "audit_channel": cfg.audit_channel or cfg.alerts_channel or cfg.channel,
        "allowlist": cfg.allowlist,
        "require_approval": cfg.require_approval,
        "digest_time": cfg.digest_time,
        "alert_types": cfg.alert_types,
        "bot_token_set": bool(cfg.bot_token),
        "app_token_set": bool(cfg.app_token),
        "user_token_set": bool(cfg.user_token),
        "ingest_enabled": cfg.ingest_enabled,
        "ingest_days": cfg.ingest_days,
        "ingest_interval_seconds": cfg.ingest_interval_seconds,
        "ingest_include_dms": cfg.ingest_include_dms,
        "ingest_channels": cfg.ingest_channels,
        "ingest_db": str(_store_path(cfg)),
    }

    if json_output:
        typer.echo(json.dumps(config_dict, indent=2))
        return

    lines = [f"{k}: {v}" for k, v in config_dict.items()]
    print_panel("\n".join(lines), title="Slack Configuration")


@slack_app.command("health")
def health(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Validate all Slack credentials and the durable history store."""
    from slack_sdk import WebClient

    cfg = load_settings(detect_hardware=False).slack
    checks: dict[str, Any] = {
        "bot": {"ok": False, "error": "token_not_set"},
        "app": {"ok": False, "error": "token_not_set"},
        "user": {"ok": False, "error": "token_not_set"},
    }
    if cfg.bot_token:
        checks["bot"] = _api_result(lambda: WebClient(token=cfg.bot_token).auth_test())
    if cfg.app_token:
        checks["app"] = _api_result(
            lambda: WebClient().apps_connections_open(app_token=cfg.app_token)
        )
    if cfg.user_token:
        checks["user"] = _api_result(lambda: WebClient(token=cfg.user_token).auth_test())

    workspace_args: dict[str, Any] = {"types": "public_channel", "limit": 1}
    if cfg.team_id:
        workspace_args["team_id"] = cfg.team_id
    checks["bot_workspace"] = (
        _api_result(
            lambda: WebClient(token=cfg.bot_token).conversations_list(**workspace_args)
        )
        if cfg.bot_token
        else {"ok": False, "error": "token_not_set", "scopes": []}
    )
    checks["user_workspace"] = (
        _api_result(
            lambda: WebClient(token=cfg.user_token).conversations_list(**workspace_args)
        )
        if cfg.user_token
        else {"ok": False, "error": "token_not_set", "scopes": []}
    )

    checks["bot"]["scope_drift"] = _scope_drift(
        checks["bot"].get("scopes", []), EXPECTED_BOT_SCOPES
    )
    checks["user"]["scope_drift"] = _scope_drift(
        checks["user"].get("scopes", []), EXPECTED_USER_SCOPES
    )

    from app.slack.ingest import SlackStore

    checks["store"] = SlackStore(_store_path(cfg)).stats()
    required = [
        checks["bot"]["ok"],
        checks["app"]["ok"],
        checks["bot_workspace"]["ok"],
    ]
    required.append(not any(checks["bot"]["scope_drift"].values()))
    if cfg.ingest_enabled:
        required.append(checks["user"]["ok"])
        required.append(checks["user_workspace"]["ok"])
        required.append(not any(checks["user"]["scope_drift"].values()))
    checks["healthy"] = all(required)

    if json_output:
        typer.echo(json.dumps(checks, indent=2))
    else:
        lines = []
        for name in ("bot", "app", "user"):
            item = checks[name]
            detail = "ok" if item["ok"] else item.get("error") or "failed"
            lines.append(f"{name.title():5}: {'✅' if item['ok'] else '❌'} {detail}")
        store = checks["store"]
        lines.append(
            f"Store: {store['messages']} messages, {store['channels']} channels, "
            f"{store['users']} users"
        )
        print_panel("\n".join(lines), title="Slack Pipeline Health")
    if not checks["healthy"]:
        raise typer.Exit(1)


@slack_app.command("sync")
def sync(
    full: bool = typer.Option(False, "--full", help="Re-read the configured history window"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Incrementally sync visible Slack history into SQLite and MAOS search."""
    rt = build_runtime()
    cfg = rt.settings.slack
    if not cfg.user_token:
        typer.echo("Slack history sync requires SLACK_USER_TOKEN.")
        raise typer.Exit(1)

    from app.slack.ingest import SlackIngestor, SlackStore

    ingestor = SlackIngestor(
        cfg.user_token,
        SlackStore(_store_path(cfg)),
        team_id=cfg.team_id,
        days=cfg.ingest_days,
        max_messages=cfg.ingest_max_messages,
        include_dms=cfg.ingest_include_dms,
        channels=tuple(cfg.ingest_channels),
        retriever=rt.retriever,
    )
    try:
        result = ingestor.sync(full=full)
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            typer.echo(f"Slack sync failed: {exc}")
        raise typer.Exit(1) from exc

    output = {"ok": not result.errors, **result.as_dict()}
    if json_output:
        typer.echo(json.dumps(output, indent=2))
    else:
        print_panel(
            "\n".join(
                [
                    f"Channels: {result.channels_synced}/{result.channels_seen}",
                    f"Messages: {result.messages_seen} seen, {result.messages_changed} changed",
                    f"Users: {result.users_seen}",
                    f"Search refreshed: {result.rag_refreshed}",
                    f"Errors: {len(result.errors)}",
                ]
            ),
            title="Slack History Sync",
        )
    if result.errors:
        raise typer.Exit(1)
