"""Tests for durable Slack history ingestion and command-channel isolation."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.cli.commands.slack import _scope_drift
from app.config.settings import SlackConfig
from app.events.bus import EventBus
from app.slack.bridge import SlackBridge
from app.slack.ingest import SlackIngestor, SlackStore


class FakeSlackClient:
    def auth_test(self):
        return {"ok": True, "team": "Test", "team_id": "T1", "user": "David"}

    def users_list(self, **kwargs):
        return {
            "ok": True,
            "members": [
                {
                    "id": "U1",
                    "name": "david",
                    "profile": {"display_name": "David", "real_name": "David"},
                }
            ],
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_list(self, **kwargs):
        return {
            "ok": True,
            "channels": [
                {
                    "id": "C1",
                    "name": "maos-bridge",
                    "is_member": True,
                    "is_archived": False,
                },
                {
                    "id": "D1",
                    "user": "U1",
                    "is_im": True,
                    "is_member": True,
                    "is_archived": False,
                },
            ],
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_history(self, channel, **kwargs):
        messages = {
            "C1": [
                {
                    "ts": "1786000000.000001",
                    "user": "U1",
                    "text": "Ship the Slack pipeline",
                    "reply_count": 1,
                }
            ],
            "D1": [
                {
                    "ts": "1786000100.000001",
                    "user": "U1",
                    "text": "Private planning note",
                }
            ],
        }
        return {
            "ok": True,
            "messages": messages[channel],
            "response_metadata": {"next_cursor": ""},
        }

    def conversations_replies(self, **kwargs):
        return {
            "ok": True,
            "messages": [
                {
                    "ts": kwargs["ts"],
                    "user": "U1",
                    "text": "Ship the Slack pipeline",
                },
                {
                    "ts": "1786000001.000001",
                    "thread_ts": kwargs["ts"],
                    "user": "U1",
                    "text": "Verified",
                },
            ],
            "response_metadata": {"next_cursor": ""},
        }


def test_scope_drift_reports_missing_and_unexpected_permissions():
    drift = _scope_drift(
        ["channels:read", "admin.users:write"],
        {"channels:read", "channels:history"},
    )

    assert drift == {
        "missing": ["channels:history"],
        "unexpected": ["admin.users:write"],
    }


def test_sync_persists_messages_and_refreshes_search(tmp_path):
    store = SlackStore(tmp_path / "slack.db")
    retriever = MagicMock()
    ingestor = SlackIngestor(
        "xoxp-test",
        store,
        days=3650,
        retriever=retriever,
        client=FakeSlackClient(),
    )

    result = ingestor.sync(full=True)

    assert result.channels_seen == 2
    assert result.channels_synced == 2
    assert result.messages_seen == 3
    assert result.messages_changed == 3
    assert result.users_seen == 1
    assert result.errors == ()
    assert result.rag_refreshed is True
    assert store.stats()["messages"] == 3
    source, chunks = retriever.update.call_args.args
    assert source == "slack"
    assert {chunk["metadata"]["channel_type"] for chunk in chunks} == {
        "public_channel",
        "im",
    }


def test_sync_can_exclude_direct_messages(tmp_path):
    store = SlackStore(tmp_path / "slack.db")
    ingestor = SlackIngestor(
        "xoxp-test",
        store,
        days=3650,
        include_dms=False,
        client=FakeSlackClient(),
    )

    result = ingestor.sync(full=True)

    assert result.channels_seen == 2
    assert result.channels_synced == 1
    assert store.stats()["messages"] == 2


def test_sync_skips_inaccessible_direct_message_shells(tmp_path):
    ingestor = SlackIngestor(
        "xoxp-test",
        SlackStore(tmp_path / "slack.db"),
        client=FakeSlackClient(),
    )

    assert not ingestor._selected(
        {"id": "D-STALE", "is_im": True, "is_member": False}
    )


class FakeBoltApp:
    def __init__(self):
        self.handlers = {}

    def event(self, name):
        def register(handler):
            self.handlers[f"event:{name}"] = handler
            return handler

        return register

    def command(self, name):
        def register(handler):
            self.handlers[f"command:{name}"] = handler
            return handler

        return register


def test_plain_messages_are_limited_to_command_channel(tmp_path):
    runtime = MagicMock()
    runtime.db = MagicMock()
    runtime.bus = MagicMock(spec=EventBus)
    runtime.settings.slack = SlackConfig(
        bot_token="xoxb-test",
        app_token="xapp-test",
        channel="C-COMMAND",
        ingest_enabled=False,
    )
    bridge = SlackBridge(runtime, runtime.settings.slack)
    bridge.router = MagicMock()
    bridge.router.handle.return_value = {"text": "ok"}
    app = FakeBoltApp()
    bridge._register_handlers(app)
    handler = app.handlers["event:message"]
    say = MagicMock()

    handler({"channel": "C-GENERAL", "channel_type": "channel", "user": "U1", "text": "status"}, say)
    bridge.router.handle.assert_not_called()

    handler({"channel": "C-COMMAND", "channel_type": "channel", "user": "U1", "text": "status"}, say)
    bridge.router.handle.assert_called_once_with("U1", "status")
