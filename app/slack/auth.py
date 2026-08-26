"""Slack authentication and authorization.

Enforces:
  - User allowlist: only specified Slack user IDs can send commands.
    Empty allowlist = allow all workspace members.
  - Per-command approval: HIGH/CRITICAL risk actions require explicit
    Approve/Deny via Slack interactive messages.
  - Rate limiting: max 1 command per 2 seconds per user.
  - Audit logging: every authorization decision is logged.
"""

from __future__ import annotations

import time

from app.log import get_logger

log = get_logger(__name__)

# Rate limit: minimum seconds between commands from the same user.
RATE_LIMIT_SECONDS = 2.0


class SlackAuth:
    """Authorization gate for Slack commands."""

    def __init__(
        self,
        allowlist: list[str] | None = None,
        *,
        rate_limit_seconds: float = RATE_LIMIT_SECONDS,
    ):
        self.allowlist = allowlist or []
        self.rate_limit = rate_limit_seconds
        self._last_command: dict[str, float] = {}  # user_id -> timestamp

    def is_authorized(self, user_id: str) -> bool:
        """Check if a Slack user is allowed to send commands."""
        if not self.allowlist:
            # Empty allowlist = allow all workspace members.
            return True
        return user_id in self.allowlist

    def check_rate_limit(self, user_id: str) -> tuple[bool, str]:
        """Check rate limit. Returns (allowed, reason)."""
        now = time.time()
        last = self._last_command.get(user_id, 0.0)
        if now - last < self.rate_limit:
            wait = self.rate_limit - (now - last)
            return False, f"Rate limited. Wait {wait:.1f}s."
        self._last_command[user_id] = now
        return True, ""

    def authorize(self, user_id: str, command: str) -> tuple[bool, str]:
        """Full authorization check: allowlist + rate limit.

        Returns (authorized, reason_if_denied).
        """
        if not self.is_authorized(user_id):
            log.warning("slack.auth.denied", user=user_id, command=command, reason="not_in_allowlist")
            return False, "You are not authorized to send commands to this device."

        ok, reason = self.check_rate_limit(user_id)
        if not ok:
            log.info("slack.auth.rate_limited", user=user_id, command=command)
            return False, reason

        log.info("slack.auth.authorized", user=user_id, command=command)
        return True, ""

    @staticmethod
    def classify_risk(command: str) -> str:
        """Classify a command's risk level: LOW / MEDIUM / HIGH / CRITICAL.

        HIGH/CRITICAL commands require explicit approval in Slack.
        """
        cmd_lower = command.lower().strip()

        # CRITICAL: destructive system commands.
        critical_patterns = [
            "rm -rf", "mkfs", "dd if=", "shutdown", "reboot", "halt",
            "sudo rm", "diskutil eraseDisk", "diskutil eraseVolume",
        ]
        for p in critical_patterns:
            if p in cmd_lower:
                return "CRITICAL"

        # HIGH: file deletion, system config changes, network changes.
        high_patterns = [
            "rm ", "rmdir", "mv /", "chmod 777", "chown",
            "defaults write", "launchctl unload", "kill -9",
            "pkill", "ifconfig ", "networksetup -set",
        ]
        for p in high_patterns:
            if p in cmd_lower:
                return "HIGH"

        # MEDIUM: file writes, package installs, git push.
        medium_patterns = [
            ">", "pip install", "npm install", "brew install",
            "git push", "git commit", "git reset --hard",
            "curl -o", "wget ", "scp ", "rsync ",
        ]
        for p in medium_patterns:
            if p in cmd_lower:
                return "MEDIUM"

        # LOW: read-only commands (default).
        return "LOW"
