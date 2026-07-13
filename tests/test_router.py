"""Tests for the intent router."""

from app.db import Database
from app.router import IntentRouter, _keyword_route


def test_keyword_routing():
    agents = ["session", "git", "repository", "filesystem", "research", "system"]

    agent, conf = _keyword_route("What was I working on?", agents)
    assert agent == "session"
    assert conf > 0.5

    agent, conf = _keyword_route("Show me recent git commits", agents)
    assert agent == "git"
    assert conf > 0.5

    agent, conf = _keyword_route("How does the retrieval pipeline work?", agents)
    assert agent == "repository"

    agent, conf = _keyword_route("Why is CPU usage so high?", agents)
    assert agent == "system"

    agent, conf = _keyword_route("What is the meaning of life?", agents)
    assert agent == "research"  # fallback
    assert conf < 0.5


def test_router_logs_decision(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    router = IntentRouter(utility=None, db=db)

    agents = ["session", "git", "repository", "filesystem", "research", "system"]
    agent, conf, reasoning = router.route("What was I working on?", agents)

    assert agent == "session"
    assert conf > 0.5
    assert "keyword" in reasoning.lower()

    # Check decision was logged.
    decisions = db.recent_decisions()
    assert len(decisions) == 1
    assert decisions[0]["agent"] == "IntentRouter"
    assert "session" in decisions[0]["decision"]
    db.close()


def test_router_with_llm(tmp_path):
    """Test LLM routing with a fake model."""

    db = Database(str(tmp_path / "test.db"))

    # Create a fake utility model that returns a routing response.
    class FakeUtilityModel:
        def chat(self, messages, **kwargs):
            class Resp:
                text = "agent: session\nconfidence: 0.92\nreasoning: query about recent work activity"
            return Resp()

    router = IntentRouter(utility=FakeUtilityModel(), db=db)
    agents = ["session", "git", "repository", "filesystem", "research", "system"]

    agent, conf, reasoning = router.route("What was I working on before I stopped?", agents)
    assert agent == "session"
    assert conf == 0.92
    assert "recent work" in reasoning

    decisions = db.recent_decisions()
    assert len(decisions) == 1
    assert decisions[0]["confidence"] == 0.92
    db.close()
