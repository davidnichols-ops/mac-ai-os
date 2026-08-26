"""Slack notifier — pushes MAOS alerts to a Slack channel.

Subscribes to EventBus events and posts relevant ones to Slack.
Uses the Slack Web API (chat.postMessage) via slack_sdk.

Alertable events (configurable via SlackConfig.alert_types):
  - NETWORK_INTERRUPTED  — network went down
  - NETWORK_DEGRADED     — latency spike or packet loss
  - NETWORK_RECONNECTED  — network came back
  - WORKFLOW_CRASHED     — a workflow failed
  - WORKFLOW_COMPLETED   — a workflow finished (info, not alert)
  - TASK_COMPLETED       — a task was marked done
  - ACTION_DENIED        — an action was blocked by permissions

The notifier is intentionally simple — it formats the event as a Slack
message and posts it. No business logic. The decision of what's alertable
lives in the config, not in the notifier.
"""

from __future__ import annotations

from typing import Any

from app.events.bus import EventBus
from app.events.models import Event
from app.log import get_logger

log = get_logger(__name__)

# Default alertable events (when alert_types config is empty).
DEFAULT_ALERT_TYPES = [
    "NETWORK_INTERRUPTED",
    "NETWORK_DEGRADED",
    "WORKFLOW_CRASHED",
    "ACTION_DENIED",
]

# Event type → emoji + title for Slack messages.
EVENT_FORMAT = {
    "NETWORK_INTERRUPTED": ("⚠️", "Network Interrupted"),
    "NETWORK_DEGRADED": ("📉", "Network Degraded"),
    "NETWORK_RECONNECTED": ("✅", "Network Reconnected"),
    "WORKFLOW_CRASHED": ("💥", "Workflow Crashed"),
    "WORKFLOW_COMPLETED": ("✅", "Workflow Completed"),
    "TASK_COMPLETED": ("✅", "Task Completed"),
    "ACTION_DENIED": ("🚫", "Action Denied"),
    "ACTION_EXECUTED": ("⚡", "Action Executed"),
    "AGENT_SESSION_COMPLETED": ("🤖", "Agent Session Completed"),
}

# Workspace → channel routing. Alerts for tasks/events from these
# workspaces go to the matching channel (by Slack channel ID).
# Falls back to self.channel (default) if the workspace isn't mapped.
WORKSPACE_CHANNELS: dict[str, str] = {
    # Product Engineering → #x-mac (C0BNFA7KWBD)
    "X-MaC": "C0BNFA7KWBD",
    "X-MaC-digital-twin": "C0BNFA7KWBD",
    # R&D / Networking → #aafp (C0BN669C4S3)
    "aafp": "C0BN669C4S3",
    "aafp-go": "C0BN669C4S3",
    "AAFP-research": "C0BN669C4S3",
    # MAOS agent operations → #maos-ops (C0BMSQGTCJV)
    "mac-ai-os": "C0BMSQGTCJV",
    # Platform / Trajectory OS → #trajectory-os (C0BNKG96KLN)
    "dnichols-ops-": "C0BNKG96KLN",
    # Engineering → #engineering (C0BNPDJEJUR)
    "repo-archaeologist": "C0BNPDJEJUR",
    "trustcard": "C0BNPDJEJUR",
    "jarword": "C0BNPDJEJUR",
    "jarvis-word": "C0BNPDJEJUR",
    "word_proc": "C0BNPDJEJUR",
    # Ops / Monitoring → #ops (C0BNPFBH64R)
    "linkedin-dev-pipeline": "C0BNPFBH64R",
    "elonwatch": "C0BNPFBH64R",
    "elonwatch-mac": "C0BNPFBH64R",
    # R&D / CV → #research (C0BPFRSEDL0)
    "cvconform": "C0BPFRSEDL0",
    "apple-quality-recognition-engine": "C0BPFRSEDL0",
    "apple_quality_recognition_engine": "C0BPFRSEDL0",
    "Gargantua": "C0BPFRSEDL0",
    # Security research → #ops (no dedicated security channel, route to ops)
    "h1-engagement": "C0BNPFBH64R",
    "slack_recon": "C0BNPFBH64R",
    "wf_recon": "C0BNPFBH64R",
    "webfuzz": "C0BNPFBH64R",
    # AI session routing → #maos-ops (C0BMSQGTCJV)
    "grokhack": "C0BMSQGTCJV",
    "claude-code": "C0BMSQGTCJV",
    # HAL9000 → #x-mac
    "HAL9000-Core": "C0BNFA7KWBD",
}

# Channel IDs for direct posting.
CHANNEL_IDS: dict[str, str] = {
    "aafp": "C0BN669C4S3",
    "daily-brief": "C0BNPFAB4MP",
    "engineering": "C0BNPDJEJUR",
    "general": "C0BMSKMUTKK",
    "github-prs": "C0BNPDPFYAV",
    "maos-bridge": "C0BMJKXL38T",
    "maos-ops": "C0BMSQGTCJV",
    "ops": "C0BNPFBH64R",
    "random": "C0BMMB3S11B",
    "research": "C0BPFRSEDL0",
    "trajectory-os": "C0BNKG96KLN",
    "x-mac": "C0BNFA7KWBD",
}


class SlackNotifier:
    """Pushes MAOS events to a Slack channel."""

    def __init__(
        self,
        bot_token: str,
        channel: str,
        bus: EventBus | None = None,
        *,
        team_id: str = "",
        alert_types: list[str] | None = None,
        alerts_channel: str = "",
        daily_channel: str = "",
        audit_channel: str = "",
    ):
        self.bot_token = bot_token
        self.team_id = team_id
        self.channel = channel
        self.bus = bus
        self.alert_types = alert_types or DEFAULT_ALERT_TYPES
        self.alerts_channel = alerts_channel or channel
        self.daily_channel = daily_channel or channel
        self.audit_channel = audit_channel or self.alerts_channel
        self._client: Any = None  # lazy-init slack_sdk WebClient

    def _get_client(self) -> Any:
        if self._client is None:
            from slack_sdk import WebClient
            from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

            self._client = WebClient(token=self.bot_token)
            self._client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=3))
        return self._client

    def post_message(self, text: str, *, blocks: list[dict] | None = None) -> bool:
        """Post a message to the default Slack channel. Returns True on success."""
        return self._post_to(self.channel, text, blocks=blocks)

    def post_to_channel(self, channel: str, text: str, *, blocks: list[dict] | None = None) -> bool:
        """Post a message to a specific channel. Falls back to default if it fails."""
        ok = self._post_to(channel, text, blocks=blocks)
        if not ok and channel != self.channel:
            log.info("slack.notify.fallback", target=channel, fallback=self.channel)
            return self._post_to(self.channel, text, blocks=blocks)
        return ok

    def _post_to(self, channel: str, text: str, *, blocks: list[dict] | None = None) -> bool:
        """Internal: post to a specific channel. Returns True on success."""
        try:
            client = self._get_client()
            kwargs: dict[str, Any] = {"channel": channel, "text": text}
            if self.team_id:
                kwargs["team_id"] = self.team_id
            if blocks:
                kwargs["blocks"] = blocks
            response = client.chat_postMessage(**kwargs)
            return response.get("ok", False)
        except Exception as e:
            log.warning("slack.notify.failed", channel=channel, error=str(e))
            return False

    def post_alert(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Format and post an event as a Slack alert, routed by workspace."""
        emoji, title = EVENT_FORMAT.get(event_type, ("📢", event_type))

        # Build a readable message from the payload.
        details = []
        for key, value in payload.items():
            if key in ("interface", "network_type", "reason", "error", "workflow", "action", "workspace", "agent_id", "provider", "role", "summary", "result"):
                details.append(f"  • {key}: {value}")

        text = f"{emoji} *{title}*\n" + "\n".join(details) if details else f"{emoji} *{title}*"

        # Route to workspace channel if mapped.
        workspace = payload.get("workspace", "")
        target_channel = WORKSPACE_CHANNELS.get(workspace, self.alerts_channel)
        if target_channel != self.channel:
            return self.post_to_channel(target_channel, text)
        return self.post_message(text)

    def post_status(self, status: dict[str, Any]) -> bool:
        """Post a system status summary to Slack."""
        lines = ["📊 *MAOS Status*\n"]
        if "network" in status:
            n = status["network"]
            healthy = "✅" if n.get("healthy") else "⚠️"
            lines.append(f"Network: {healthy} {n.get('network_type', '?')} / {n.get('connection_state', '?')}")
            if n.get("latency_ms") is not None:
                lines.append(f"  Latency: {n['latency_ms']:.0f}ms, Quality: {n.get('bandwidth_quality', 0)}/100")
        if "system" in status:
            s = status["system"]
            if s.get("available"):
                lines.append(f"CPU: {s.get('cpu_percent', 0):.1f}%, Mem: {s.get('memory_used_gb', 0):.1f}GB")
        if "tasks" in status:
            tasks = status["tasks"]
            lines.append(f"Tasks: {len(tasks)} pending")

        return self.post_message("\n".join(lines))

    def post_audit(self, text: str) -> bool:
        """Post integration lifecycle and data-pipeline audit messages."""
        return self.post_to_channel(self.audit_channel, text)

    def post_digest(self, digest: dict[str, Any]) -> bool:
        """Post a daily digest to Slack."""
        lines = ["📋 *Daily Digest*\n"]

        if "network" in digest:
            n = digest["network"]
            lines.append(f"Network health: {n.get('healthy_days', 0)}/{n.get('total_days', 0)} healthy days")
            if n.get("total_rx_gb"):
                lines.append(f"  Data: {n['total_rx_gb']:.1f}GB down, {n['total_tx_gb']:.1f}GB up")
            if n.get("interruptions"):
                lines.append(f"  Interruptions: {n['interruptions']}")

        if "system" in digest:
            s = digest["system"]
            lines.append(f"CPU avg: {s.get('cpu_avg', 0):.1f}%, Mem avg: {s.get('mem_avg', 0):.1f}GB")

        if "tasks" in digest:
            t = digest["tasks"]
            lines.append(f"Tasks: {t.get('completed', 0)} completed, {t.get('pending', 0)} pending")

        if "events" in digest:
            e = digest["events"]
            lines.append(f"Events: {e.get('total', 0)} total, {e.get('alerts', 0)} alerts")

        return self.post_to_channel(self.daily_channel, "\n".join(lines))

    def start_listening(self) -> None:
        """Subscribe to EventBus and push alertable events to Slack."""
        if not self.bus:
            return

        def on_event(event: Event) -> None:
            if event.type in self.alert_types:
                self.post_alert(event.type, event.payload)

        # Subscribe to all events, filter inside the handler.
        self.bus.subscribe("*", on_event)
        log.info("slack.notifier.listening", alert_types=self.alert_types)

    def stop_listening(self) -> None:
        """Unsubscribe from EventBus (if subscribed)."""
        # EventBus doesn't have unsubscribe-by-handler in all versions.
        # The bridge handles lifecycle — when the bridge stops, the notifier
        # is garbage collected. This is a no-op for now.
        pass
