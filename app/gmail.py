"""Email — send mail via Gmail OAuth.

This is the honest signal channel. When an agent encounters something
that needs human attention — a verification failed, a lesson was learned
the hard way, context is getting diluted with "everything works" vibes —
it sends an email instead of silently moving on.

Architecture:
    Gmail OAuth credentials live in ~/.gmail-mcp/ (shared with the
    gmail-github-to-maos.js sync script). The default recipient is read
    from the ``MAOS_EMAIL_RECIPIENT`` environment variable so the module
    remains account-agnostic.

    Tokens expire. This module refreshes them on demand using the
    stored refresh_token + client credentials. No interactive auth flow.

    Sending uses the Gmail API's messages.send endpoint with
    base64url-encoded RFC 2822 messages. No external dependencies —
    just stdlib (urllib, base64, email).

Security:
    - Credentials are read from ~/.gmail-mcp/, never written to logs.
    - Refreshed tokens are saved back to the credentials file so the
      sync script benefits too.
    - The recipient defaults to the account owner; override with the
      ``to`` argument or the ``MAOS_EMAIL_RECIPIENT`` environment variable.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from app.log import get_logger

log = get_logger(__name__)

GMAIL_DIR = Path.home() / ".gmail-mcp"
SENDER_KEYS = GMAIL_DIR / "account-b-keys.json"
SENDER_CREDS = GMAIL_DIR / "account-b-credentials.json"
RECIPIENT = os.environ.get("MAOS_EMAIL_RECIPIENT", "")

GMAIL_API_SEND = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"


class EmailError(Exception):
    """Failed to send or refresh."""


def _refresh_access_token() -> str:
    """Refresh the Gmail OAuth access token using the stored refresh token.

    Returns a fresh access token. Also saves it back to the credentials
    file so the sync script and other consumers benefit.
    """
    if not SENDER_KEYS.exists():
        raise EmailError(f"Gmail keys file not found: {SENDER_KEYS}")
    if not SENDER_CREDS.exists():
        raise EmailError(f"Gmail credentials file not found: {SENDER_CREDS}")

    keys = json.loads(SENDER_KEYS.read_text())
    installed = keys.get("installed", keys)
    creds = json.loads(SENDER_CREDS.read_text())

    refresh_token = creds.get("refresh_token")
    if not refresh_token:
        raise EmailError("No refresh_token in credentials file")

    data = urllib.parse.urlencode({
        "client_id": installed["client_id"],
        "client_secret": installed["client_secret"],
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()

    req = urllib.request.Request(OAUTH_TOKEN_URL, data=data)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        token_data = json.loads(resp.read())
    except Exception as e:
        raise EmailError(f"Token refresh failed: {e}") from e

    access_token = token_data.get("access_token")
    if not access_token:
        raise EmailError("No access_token in refresh response")

    # Save refreshed token back.
    creds["access_token"] = access_token
    creds["expiry_date"] = token_data.get("expires_in", 3600)
    try:
        SENDER_CREDS.write_text(json.dumps(creds, indent=2))
    except Exception as e:
        log.warning("email.token_save_failed", error=str(e))

    log.info("email.token.refreshed")
    return access_token


def _build_rfc2822(
    subject: str,
    body: str,
    to: str = "",
    sender: str = "",
) -> bytes:
    """Build an RFC 2822 message as bytes."""
    msg = EmailMessage()
    if to:
        msg["To"] = to
    if sender:
        msg["From"] = sender
    msg["Subject"] = subject
    msg.set_content(body)
    return msg.as_bytes()


def send_email(
    subject: str,
    body: str,
    *,
    to: str = "",
) -> dict[str, Any]:
    """Send an email via Gmail.

    Args:
        subject: Email subject line.
        body: Plain text body.
        to: Recipient. Defaults to ``MAOS_EMAIL_RECIPIENT`` if set,
            otherwise raises an error.

    Returns:
        Dict with success, message_id, and any error.

    Raises:
        EmailError: If no recipient is configured or sending fails.
    """
    recipient = to or RECIPIENT
    if not recipient:
        raise EmailError(
            "No email recipient configured. Set MAOS_EMAIL_RECIPIENT "
            "or pass the 'to' argument."
        )

    access_token = _refresh_access_token()

    raw_message = _build_rfc2822(subject, body, to=recipient)
    encoded = base64.urlsafe_b64encode(raw_message).decode()

    payload = json.dumps({"raw": encoded}).encode()

    req = urllib.request.Request(
        GMAIL_API_SEND,
        data=payload,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        message_id = result.get("id", "")
        log.info("email.sent", subject=subject, message_id=message_id)
        return {"success": True, "message_id": message_id, "subject": subject}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()[:500]
        log.error("email.send_failed", status=e.code, error=error_body)
        raise EmailError(f"Gmail API error {e.code}: {error_body}") from e
    except Exception as e:
        log.error("email.send_failed", error=str(e))
        raise EmailError(f"Send failed: {e}") from e


def escalate(
    subject: str,
    body: str,
    *,
    context: dict[str, Any] | None = None,
    lesson_slug: str | None = None,
    to: str = "",
) -> dict[str, Any]:
    """Send an escalation email — 'this needs human attention'.

    This is the anti-vibe-overload channel. When something isn't working,
    when a verification fails, when context is getting diluted with false
    confidence, the agent escalates instead of glossing over it.

    The email includes:
    - The subject (what happened)
    - The body (the details)
    - Optional structured context (appended as a section)
    - Optional lesson slug (if this escalation resulted in a lesson)

    Args:
        subject: Short summary of what needs attention.
        body: The detailed explanation.
        context: Optional structured context to append.
        lesson_slug: Optional lesson slug if this is related to a recorded lesson.
        to: Optional recipient. Defaults to ``MAOS_EMAIL_RECIPIENT``.

    Returns:
        Dict with success, message_id, and subject.
    """
    sections = [body]

    if context:
        sections.append("")
        sections.append("--- Context ---")
        for key, value in context.items():
            val_str = str(value)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            sections.append(f"{key}: {val_str}")

    if lesson_slug:
        sections.append("")
        sections.append(f"Lesson recorded: {lesson_slug}")

    sections.append("")
    sections.append("-- MAOS")

    full_body = "\n".join(sections)
    prefix = "[MAOS]"
    if not subject.startswith(prefix):
        subject = f"{prefix} {subject}"

    return send_email(subject, full_body, to=to)
