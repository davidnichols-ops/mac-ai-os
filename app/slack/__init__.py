"""Slack integration — remote management and alerts via Slack.

Bidirectional bridge between MAOS and a Slack workspace:
  - Push alerts: MAOS events → Slack channel notifications
  - Remote commands: Slack messages → MAOS ActionExecutor
  - Daily digest: scheduled summary of device state
  - Two-way chat: ask questions, run agentic loops from Slack

Uses Slack Socket Mode (WebSocket) — no public endpoint, no ngrok tunnel.
The Slack workspace connects to MAOS, not the other way around.

Security:
  - User allowlist: only specified Slack user IDs can send commands
  - Per-command approval: HIGH/CRITICAL risk actions require explicit
    Approve/Deny button clicks in Slack
  - Audit log: every command is recorded via ActionExecutor → action_log
  - Rate limiting: max 1 command per 2 seconds per user

Note: SlackBridge is imported lazily (not in __init__) to avoid a circular
import with app.runtime. Import it directly:
    from app.slack.bridge import SlackBridge
"""

from __future__ import annotations

from app.slack.auth import SlackAuth
from app.slack.commands import SlackCommandRouter
from app.slack.notifier import SlackNotifier

__all__ = [
    "SlackAuth",
    "SlackCommandRouter",
    "SlackNotifier",
]
