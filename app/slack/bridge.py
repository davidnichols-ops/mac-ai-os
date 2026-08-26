"""Slack bridge — bidirectional MAOS ↔ Slack via Socket Mode.

Uses Slack's Socket Mode (WebSocket) so no public endpoint is needed.
The Slack workspace connects to MAOS, not the other way around.

Lifecycle:
    bridge = SlackBridge(runtime)
    bridge.start()  # blocks in a background thread, listens for Slack events
    bridge.stop()   # clean shutdown

The bridge handles:
  - Incoming messages in the #maos channel → command router → MAOS actions
  - App mentions → command router (same path)
  - Direct messages → command router (same path)
  - EventBus events → notifier → Slack channel (push alerts)
  - Daily digest scheduling (if configured)

Security:
  - User allowlist (SlackAuth)
  - Per-command approval for HIGH/CRITICAL risk (SlackAuth + command router)
  - Rate limiting (SlackAuth)
  - All actions logged via ActionExecutor → action_log + decisions tables
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.config.settings import SlackConfig
from app.events.bus import EventBus
from app.log import get_logger
from app.runtime import Runtime
from app.slack.auth import SlackAuth
from app.slack.commands import SlackCommandRouter
from app.slack.notifier import SlackNotifier

log = get_logger(__name__)


class SlackBridge:
    """Bidirectional MAOS ↔ Slack bridge via Socket Mode."""

    def __init__(self, runtime: Runtime, config: SlackConfig | None = None):
        self.runtime = runtime
        self.config = config or runtime.settings.slack
        self.bus: EventBus | None = runtime.bus
        self.auth = SlackAuth(
            allowlist=self.config.allowlist,
        )
        self.router = SlackCommandRouter(runtime, self.auth)
        self.notifier = SlackNotifier(
            bot_token=self.config.bot_token,
            channel=self.config.channel,
            bus=self.bus,
            team_id=self.config.team_id,
            alert_types=self.config.alert_types or None,
            alerts_channel=self.config.alerts_channel,
            daily_channel=self.config.daily_channel,
            audit_channel=self.config.audit_channel,
        )
        self._app: Any = None  # slack_bolt App
        self._handler: Any = None  # SocketModeHandler
        self._thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_stop = threading.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def is_configured(self) -> bool:
        """Check if the bridge has the required tokens."""
        return bool(self.config.bot_token and self.config.app_token)

    def start(self, *, block: bool = False) -> bool:
        """Start the Slack bridge.

        Args:
            block: If True, blocks the calling thread. If False, runs in
                a background thread.

        Returns:
            True if started successfully, False if not configured or failed.
        """
        if not self.is_configured():
            log.warning("slack.bridge.not_configured")
            return False

        if self._running:
            log.warning("slack.bridge.already_running")
            return True

        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
        except ImportError:
            log.error("slack.bridge.import_failed", msg="slack_bolt not installed. Run: uv sync --extra slack")
            return False

        # Build the Bolt app.
        self._app = App(token=self.config.bot_token)

        # Register event handlers.
        self._register_handlers(self._app)

        # Start the notifier (push alerts).
        self.notifier.start_listening()

        self._running = True
        self._start_sync_worker()

        # Keep lifecycle noise out of the command channel.
        self.notifier.post_audit("MAOS Slack bridge online. History sync is active.")

        if block:
            log.info("slack.bridge.starting", mode="blocking")
            self._handler = SocketModeHandler(self._app, self.config.app_token)
            self._handler.start()
        else:
            log.info("slack.bridge.starting", mode="background")
            self._thread = threading.Thread(target=self._run_socket_mode, daemon=True)
            self._thread.start()

        return True

    def _run_socket_mode(self) -> None:
        """Run the Socket Mode handler in a background thread."""
        try:
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            self._handler = SocketModeHandler(self._app, self.config.app_token)
            self._handler.start()
        except Exception as e:
            log.error("slack.bridge.socket_mode_failed", error=str(e))
            self._running = False

    def stop(self) -> None:
        """Stop the bridge cleanly."""
        if not self._running:
            return

        self._running = False
        self._sync_stop.set()

        if self._handler:
            try:
                self._handler.close()
            except Exception:
                pass

        self.notifier.stop_listening()

        if self._thread:
            self._thread.join(timeout=5)

        if self._sync_thread:
            self._sync_thread.join(timeout=5)

        # Post a shutdown message.
        try:
            self.notifier.post_audit("MAOS Slack bridge offline.")
        except Exception:
            pass

        log.info("slack.bridge.stopped")

    def _register_handlers(self, app: Any) -> None:
        """Register Slack event handlers on the Bolt app."""

        @app.event("message")
        def handle_message(event: dict, say: Any) -> None:
            """Handle messages in the #maos channel."""
            # Ignore bot messages (including our own).
            if event.get("bot_id") or event.get("subtype"):
                return

            text = event.get("text", "").strip()
            user = event.get("user", "")
            channel = event.get("channel", "")
            direct = event.get("channel_type") in {"im", "mpim"}

            if not text or not user:
                return

            # Plain messages are commands only in the dedicated command channel
            # or a direct conversation. Mentions remain available everywhere.
            if not direct and channel != self.config.channel:
                return

            # Route through the command router with provenance for memory capture.
            if "ts" in event:
                response = self.router.handle(
                    user,
                    text,
                    source_context={
                        "channel": channel,
                        "ts": event.get("ts", ""),
                        "thread_ts": event.get("thread_ts", ""),
                        "title": event.get("username", ""),
                    },
                )
            else:
                response = self.router.handle(user, text)
            say(response.get("text", "Error: no response"))

        @app.event("app_mention")
        def handle_mention(event: dict, say: Any) -> None:
            """Handle @maos mentions in any channel."""
            text = event.get("text", "").strip()
            user = event.get("user", "")

            # Strip the mention prefix (@maos).
            if "<@" in text:
                text = text.split(">", 1)[-1].strip()

            if not text:
                say("Yes? Send `help` for available commands.")
                return

            response = self.router.handle(
                user,
                text,
                source_context={
                    "channel": event.get("channel", ""),
                    "ts": event.get("ts", ""),
                    "thread_ts": event.get("thread_ts", ""),
                },
            )
            say(response.get("text", "Error: no response"))

        @app.command("/maos")
        def handle_slash_command(ack: Any, command: dict, say: Any) -> None:
            """Handle /maos slash commands."""
            ack()
            text = command.get("text", "").strip()
            user_id = command.get("user_id", "")

            if not text:
                say("Send `help` for available commands.")
                return

            response = self.router.handle(
                user_id,
                text,
                source_context={
                    "channel": command.get("channel_id", ""),
                    "ts": command.get("trigger_id", ""),
                },
            )
            say(response.get("text", "Error: no response"))

    def _start_sync_worker(self) -> None:
        if not self.config.ingest_enabled:
            log.info("slack.ingest.disabled")
            return
        if not self.config.user_token:
            log.warning("slack.ingest.user_token_missing")
            return
        self._sync_stop.clear()
        self._sync_thread = threading.Thread(
            target=self._run_sync_worker,
            name="maos-slack-sync",
            daemon=True,
        )
        self._sync_thread.start()

    def _run_sync_worker(self) -> None:
        from app.slack.ingest import SlackIngestor, SlackStore

        store_path = Path(self.config.ingest_db).expanduser()
        if not store_path.is_absolute():
            store_path = Path(__file__).resolve().parents[2] / store_path
        ingestor = SlackIngestor(
            self.config.user_token,
            SlackStore(store_path),
            team_id=self.config.team_id,
            days=self.config.ingest_days,
            max_messages=self.config.ingest_max_messages,
            include_dms=self.config.ingest_include_dms,
            channels=tuple(self.config.ingest_channels),
            retriever=self.runtime.retriever,
        )
        while not self._sync_stop.is_set():
            try:
                result = ingestor.sync()
                log.info("slack.ingest.completed", **result.as_dict())
            except Exception as exc:
                log.error("slack.ingest.failed", error=str(exc))
                self.notifier.post_audit(f"Slack history sync failed: {exc}")
            self._sync_stop.wait(self.config.ingest_interval_seconds)

    def send_test_message(self) -> bool:
        """Send a test message to verify the Slack connection works."""
        return self.notifier.post_message("🧪 MAOS Slack bridge test. If you see this, the connection works!")

    def send_status(self) -> bool:
        """Send a status update to Slack."""
        from app.server.context import build_context_packet
        packet = build_context_packet(
            self.runtime.db, self.runtime.blackboard,
            retriever=self.runtime.retriever,
        )
        return self.notifier.post_status(packet)
