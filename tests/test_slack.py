"""Tests for the Slack integration layer.

Tests use mocked Slack clients — no real Slack API calls are made.
Covers:
  - SlackAuth: allowlist, rate limiting, risk classification
  - SlackNotifier: message formatting, event filtering
  - SlackCommandRouter: all commands (status, network, telemetry, tasks,
    run, ask, help, digest, context), error handling, approval flow
  - SlackBridge: configuration check, lifecycle
  - Digest builder
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.db import Database
from app.events.bus import EventBus
from app.events.models import NETWORK_INTERRUPTED, Event
from app.slack.auth import SlackAuth
from app.slack.commands import SlackCommandRouter
from app.slack.digest import build_digest
from app.slack.notifier import SlackNotifier

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "test.db"))
    yield d
    d.close()


@pytest.fixture
def bus(db):
    return EventBus(db=db)


@pytest.fixture
def auth():
    return SlackAuth(allowlist=["U12345", "U67890"])


# --- SlackAuth --------------------------------------------------------------


class TestSlackAuth:
    def test_allowlist_allows_listed_user(self, auth):
        assert auth.is_authorized("U12345") is True

    def test_allowlist_denies_unlisted_user(self, auth):
        assert auth.is_authorized("U99999") is False

    def test_empty_allowlist_allows_all(self):
        a = SlackAuth(allowlist=[])
        assert a.is_authorized("U99999") is True

    def test_authorize_denied_user(self, auth):
        ok, reason = auth.authorize("U99999", "status")
        assert ok is False
        assert "not authorized" in reason.lower()

    def test_authorize_allowed_user(self, auth):
        ok, reason = auth.authorize("U12345", "status")
        assert ok is True
        assert reason == ""

    def test_rate_limiting(self, auth):
        # First call succeeds.
        ok1, _ = auth.authorize("U12345", "status")
        assert ok1 is True
        # Second call immediately is rate-limited.
        ok2, reason2 = auth.authorize("U12345", "status")
        assert ok2 is False
        assert "rate" in reason2.lower()

    def test_rate_limit_different_users(self, auth):
        ok1, _ = auth.authorize("U12345", "status")
        assert ok1 is True
        # Different user is not rate-limited.
        ok2, _ = auth.authorize("U67890", "status")
        assert ok2 is True

    def test_risk_classification_low(self, auth):
        assert SlackAuth.classify_risk("ls -la") == "LOW"
        assert SlackAuth.classify_risk("status") == "LOW"
        assert SlackAuth.classify_risk("cat file.txt") == "LOW"

    def test_risk_classification_medium(self, auth):
        assert SlackAuth.classify_risk("echo hello > file.txt") == "MEDIUM"
        assert SlackAuth.classify_risk("git push origin main") == "MEDIUM"

    def test_risk_classification_high(self, auth):
        assert SlackAuth.classify_risk("rm file.txt") == "HIGH"
        assert SlackAuth.classify_risk("kill -9 1234") == "HIGH"

    def test_risk_classification_critical(self, auth):
        assert SlackAuth.classify_risk("rm -rf /") == "CRITICAL"
        assert SlackAuth.classify_risk("shutdown -h now") == "CRITICAL"
        assert SlackAuth.classify_risk("sudo rm -rf /tmp") == "CRITICAL"


# --- SlackNotifier ----------------------------------------------------------


class TestSlackNotifier:
    def test_post_message_success(self):
        notifier = SlackNotifier("xoxb-fake", "#maos")
        with patch.object(notifier, "_get_client") as mock_client:
            client = MagicMock()
            client.chat_postMessage.return_value = {"ok": True}
            mock_client.return_value = client
            ok = notifier.post_message("test message")
            assert ok is True
            client.chat_postMessage.assert_called_once()

    def test_post_message_failure(self):
        notifier = SlackNotifier("xoxb-fake", "#maos")
        with patch.object(notifier, "_get_client") as mock_client:
            client = MagicMock()
            client.chat_postMessage.side_effect = Exception("API error")
            mock_client.return_value = client
            ok = notifier.post_message("test message")
            assert ok is False

    def test_post_alert_formats_event(self):
        notifier = SlackNotifier("xoxb-fake", "#maos")
        with patch.object(notifier, "post_message") as mock_post:
            mock_post.return_value = True
            notifier.post_alert(NETWORK_INTERRUPTED, {
                "interface": "en0",
                "network_type": "wifi",
            })
            mock_post.assert_called_once()
            msg = mock_post.call_args[0][0]
            assert "Network Interrupted" in msg
            assert "en0" in msg

    def test_maos_agent_handoff_routes_to_maos_ops(self):
        notifier = SlackNotifier("xoxb-fake", "#maos")
        with patch.object(notifier, "post_to_channel", return_value=True) as mock_post:
            notifier.post_alert(
                "AGENT_SESSION_COMPLETED",
                {"workspace": "mac-ai-os", "summary": "verified"},
            )

        assert mock_post.call_args.args[0] == "C0BMSQGTCJV"

    def test_post_status(self):
        notifier = SlackNotifier("xoxb-fake", "#maos")
        with patch.object(notifier, "post_message") as mock_post:
            mock_post.return_value = True
            notifier.post_status({
                "network": {"healthy": True, "network_type": "wifi",
                            "latency_ms": 20, "bandwidth_quality": 95},
                "system": {"available": True, "cpu_percent": 15.0,
                           "memory_used_gb": 8.0},
                "tasks": [{"id": 1}, {"id": 2}],
            })
            mock_post.assert_called_once()
            msg = mock_post.call_args[0][0]
            assert "MAOS Status" in msg
            assert "Network" in msg

    def test_event_bus_subscription(self, bus):
        """The notifier subscribes to EventBus and pushes alertable events."""
        notifier = SlackNotifier("xoxb-fake", "#maos", bus=bus,
                                  alert_types=[NETWORK_INTERRUPTED])
        with patch.object(notifier, "post_message") as mock_post:
            mock_post.return_value = True
            notifier.start_listening()
            # Publish an alertable event.
            bus.publish(Event(NETWORK_INTERRUPTED, {"interface": "en0"}))
            assert mock_post.called
            # Publish a non-alertable event.
            mock_post.reset_mock()
            bus.publish(Event("SOME_OTHER_EVENT", {"data": "x"}))
            assert not mock_post.called


# --- SlackCommandRouter -----------------------------------------------------


class TestSlackCommandRouter:
    def _make_router(self, db, auth=None):
        """Build a router with a mock runtime."""
        from app.config.settings import Settings, SlackConfig
        from app.network import NetworkProvider
        from app.runtime import Runtime

        rt = MagicMock(spec=Runtime)
        rt.db = db
        rt.bus = EventBus(db=db)
        rt.network = NetworkProvider(db)
        rt.settings = Settings()
        rt.settings.slack = SlackConfig(require_approval=True)
        rt.blackboard = MagicMock()
        rt.blackboard.snapshot.return_value = {
            "current_goal": "", "current_workspace": "",
            "open_problems": [], "current_plan": "",
            "suggested_actions": [], "confidence": 0.5,
            "last_updated": "",
        }
        rt.retriever = MagicMock()
        rt.retriever.search.return_value = []
        rt.action_executor = MagicMock()
        rt.orchestrator = MagicMock()

        auth = auth or SlackAuth(allowlist=[])  # allow all for tests
        return SlackCommandRouter(rt, auth), rt

    def test_help_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "help")
        assert resp["ok"] is True
        assert "Available Commands" in resp["text"]
        assert "status" in resp["text"]

    def test_network_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "network")
        assert resp["ok"] is True
        assert "Network Health" in resp["text"]

    def test_telemetry_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "telemetry")
        assert resp["ok"] is True
        assert "Telemetry" in resp["text"]

    def test_tasks_command_empty(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "tasks")
        assert resp["ok"] is True
        assert "No pending tasks" in resp["text"]

    def test_tasks_command_with_data(self, db):
        router, _ = self._make_router(db)
        db.execute(
            "INSERT INTO tasks (title, status, created_at) VALUES (?, 'pending', ?)",
            ("Test task", "2026-08-03T12:00:00+00:00"),
        )
        resp = router.handle("U1", "tasks")
        assert resp["ok"] is True
        assert "Test task" in resp["text"]

    def test_run_command_low_risk(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "run echo hello")
        assert resp["ok"] is True
        assert "hello" in resp["text"]

    def test_run_command_high_risk_needs_approval(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "run rm somefile.txt")
        assert resp["ok"] is False
        assert resp.get("needs_approval") is True
        assert resp["risk"] == "HIGH"
        assert "approve" in resp["text"].lower()

    def test_run_command_critical_needs_approval(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "run rm -rf /tmp/test")
        assert resp["ok"] is False
        assert resp.get("needs_approval") is True
        assert resp["risk"] == "CRITICAL"

    def test_run_command_empty(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "run")
        assert resp["ok"] is False
        assert "Usage" in resp["text"]

    def test_unknown_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "foobar")
        assert resp["ok"] is False
        assert "Unknown command" in resp["text"]

    def test_empty_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "")
        assert resp["ok"] is False

    def test_natural_language_question_routed_to_ask(self, db):
        router, rt = self._make_router(db)
        # Mock retriever to return a result.
        mock_result = MagicMock()
        mock_result.source = "test.md"
        mock_result.score = 0.9
        mock_result.text = "This is the answer."
        rt.retriever.search.return_value = [mock_result]
        resp = router.handle("U1", "what is MAOS?")
        assert resp["ok"] is True
        assert "Results for" in resp["text"]

    def test_authorized_user_denied(self, db):
        auth = SlackAuth(allowlist=["U_ALLOWED"])
        router, _ = self._make_router(db, auth=auth)
        resp = router.handle("U_DENIED", "status")
        assert resp["ok"] is False
        assert "not authorized" in resp["text"].lower()

    def test_slash_command_prefix(self, db):
        """Commands with a leading / (slash) are handled."""
        router, _ = self._make_router(db)
        resp = router.handle("U1", "/status")
        assert resp["ok"] is True
        assert "MAOS Status" in resp["text"]

    def test_context_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "context")
        assert resp["ok"] is True
        assert "json" in resp["text"].lower()

    def test_digest_command(self, db):
        router, _ = self._make_router(db)
        resp = router.handle("U1", "digest")
        assert resp["ok"] is True

    def test_umbrella_command(self, db, tmp_path):
        auth = SlackAuth(allowlist=[], rate_limit_seconds=0.0)
        router, rt = self._make_router(db, auth=auth)
        rt.home = tmp_path
        (tmp_path / "Projects" / "demo" / ".git").mkdir(parents=True)
        (tmp_path / "bin").mkdir()
        (tmp_path / "mac-ai-os" / "docs").mkdir(parents=True)
        (tmp_path / "bin" / "mycli").write_text("#!/bin/sh\n")
        (tmp_path / "mac-ai-os" / "docs" / "vision.md").write_text("# Vision\n")
        db.create_task("Umbrella test")

        resp = router.handle("U1", "umbrella")
        assert resp["ok"] is True
        assert "MAOS Umbrella" in resp["text"]
        assert "Pending tasks: 1" in resp["text"]

        # portfolio is an alias for umbrella.
        resp2 = router.handle("U1", "portfolio")
        assert resp2["ok"] is True
        assert "MAOS Umbrella" in resp2["text"]


# --- Digest -----------------------------------------------------------------


class TestDigest:
    def test_digest_empty_db(self, db):
        d = build_digest(db)
        assert "network" in d
        assert "system" in d
        assert "tasks" in d
        assert "events" in d

    def test_digest_with_network_data(self, db):
        hour_bucket = datetime.now(UTC).strftime("%Y-%m-%dT%H")
        db.upsert_network_hourly({
            "bucket": hour_bucket, "interface": "en0", "network_type": "wifi",
            "rx_bytes": 1_000_000_000, "tx_bytes": 500_000_000,
            "rx_rate_avg": 1.0, "tx_rate_avg": 0.5,
            "latency_avg": 12.0, "packet_loss_avg": 0.0, "minute_count": 60,
        })
        d = build_digest(db, days=7)
        assert d["network"]["total_rx_gb"] == pytest.approx(1.0, rel=0.1)

    def test_digest_with_telemetry(self, db):
        from app.telemetry.collector import TelemetryCollector
        with patch("app.telemetry.collector._collect_snapshot") as mock:
            mock.return_value = {
                "timestamp": datetime.now(UTC).isoformat(),
                "cpu_percent": 25.0, "cpu_cores": 10,
                "memory_total_gb": 16.0, "memory_used_gb": 8.0,
                "memory_percent": 50.0,
            }
            TelemetryCollector(db, interval_seconds=1.0).collect_once()
        d = build_digest(db, days=7)
        assert d["system"]["samples"] >= 1


# --- SlackBridge (configuration + lifecycle, no real Slack) -----------------


class TestSlackBridge:
    def test_not_configured_without_tokens(self, db):
        from app.config.settings import SlackConfig
        from app.slack.bridge import SlackBridge

        cfg = SlackConfig(bot_token="", app_token="")
        rt = MagicMock()
        rt.db = db
        rt.bus = EventBus(db=db)
        rt.settings = MagicMock()
        rt.settings.slack = cfg

        bridge = SlackBridge(rt, cfg)
        assert bridge.is_configured() is False

    def test_configured_with_tokens(self, db):
        from app.config.settings import SlackConfig
        from app.slack.bridge import SlackBridge

        cfg = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
        rt = MagicMock()
        rt.db = db
        rt.bus = EventBus(db=db)
        rt.settings = MagicMock()
        rt.settings.slack = cfg

        bridge = SlackBridge(rt, cfg)
        assert bridge.is_configured() is True

    def test_start_without_config_returns_false(self, db):
        from app.config.settings import SlackConfig
        from app.slack.bridge import SlackBridge

        cfg = SlackConfig(bot_token="", app_token="")
        rt = MagicMock()
        rt.db = db
        rt.bus = EventBus(db=db)
        rt.settings = MagicMock()
        rt.settings.slack = cfg

        bridge = SlackBridge(rt, cfg)
        assert bridge.start() is False


class TestQaModelApiKeyResolution:
    """Tests for the OpenRouter API key resolution fix.

    The Slack bridge process may not have OPENROUTER_API_KEY in its
    environment (e.g. started via launchd). The settings loader and
    APIModel must both resolve the key from the environment.
    """

    def test_settings_loads_openrouter_key_from_env(self, monkeypatch):
        """load_settings() should populate qa_api_key from OPENROUTER_API_KEY."""
        from app.config.settings import load_settings

        # Mock load_dotenv so the real .env doesn't interfere.
        monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **kw: None)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = load_settings()
        assert settings.slack.qa_api_key == "sk-or-test-key-12345"

    def test_settings_falls_back_to_openai_key(self, monkeypatch):
        """If OPENROUTER_API_KEY is unset, fall back to OPENAI_API_KEY."""
        from app.config.settings import load_settings

        # Mock load_dotenv to prevent .env from re-setting OPENROUTER_API_KEY.
        monkeypatch.setattr("app.config.settings.load_dotenv", lambda *a, **kw: None)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
        settings = load_settings()
        assert settings.slack.qa_api_key == "sk-openai-fallback"

    def test_api_model_resolves_key_at_client_creation(self, monkeypatch):
        """APIModel._get_client should resolve env vars if api_key is empty or ${VAR}."""
        from app.models.api_provider import APIModel

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-resolved-key")
        model = APIModel(
            "test-model",
            base_url="https://openrouter.ai/api/v1",
            api_key="",  # empty — should fall back to env
        )
        client = model._get_client()
        assert client.api_key == "sk-or-resolved-key"

    def test_api_model_resolves_unexpanded_var(self, monkeypatch):
        """APIModel should resolve ${OPENROUTER_API_KEY} if it wasn't expanded."""
        from app.models.api_provider import APIModel

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key-999")
        model = APIModel(
            "test-model",
            base_url="https://openrouter.ai/api/v1",
            api_key="${OPENROUTER_API_KEY}",  # unexpanded
        )
        client = model._get_client()
        assert client.api_key == "sk-or-env-key-999"
