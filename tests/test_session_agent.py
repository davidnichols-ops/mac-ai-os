from datetime import UTC, datetime, timedelta

from app.agents.session import SessionAgent
from app.db import Database
from app.events.bus import EventBus
from app.events.models import FILE_INDEXED, Event
from tests.conftest import FakeModel


def _populate_activity(db, bus):
    """Create a realistic activity history."""
    base = datetime.now(UTC) - timedelta(hours=2)

    # Simulate a work session: file indexing events.
    for i in range(5):
        ts = (base + timedelta(minutes=i * 2)).isoformat()
        bus.publish(Event(FILE_INDEXED, {
            "path": f"/Users/david/mac-ai-os/app/events/file{i}.py",
            "source": "/Users/david/mac-ai-os",
        }, timestamp=ts))

    # Add a git commit.
    db.upsert_commit(
        repo="/Users/david/mac-ai-os", hash="abc123", author="David",
        author_date=base.isoformat(), message="Add event bus",
        files_changed="app/events/bus.py\napp/events/models.py",
        insertions=50, deletions=2,
    )


def test_session_agent_structured(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    _populate_activity(db, bus)

    agent = SessionAgent(planner=FakeModel(), db=db)
    result = agent.execute("What was I working on?")

    assert result.ok
    # The FakeModel returns a canned response, but the agent should have
    # gathered the data and produced steps.
    assert len(result.steps) >= 1
    assert result.data["event_count"] >= 5
    db.close()


def test_session_agent_matches():
    assert SessionAgent.matches("What was I working on before I stopped?")
    assert SessionAgent.matches("What did I do yesterday?")
    assert SessionAgent.matches("show me my last session")
    assert not SessionAgent.matches("what is the weather?")
    assert not SessionAgent.matches("read this file")


def test_session_agent_no_data(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    agent = SessionAgent(planner=FakeModel(), db=db)
    result = agent.execute("What was I working on?")

    assert result.ok
    assert "don't have enough" in result.answer.lower()
    db.close()


def test_context_builder(tmp_path):
    from app.context import build_context

    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    _populate_activity(db, bus)

    ctx = build_context(db, "what was I working on?", lookback_hours=24)
    assert ctx.user_query == "what was I working on?"
    assert len(ctx.recent_events) > 0
    assert len(ctx.recent_commits) > 0
    assert "event bus" in ctx.recent_commits[0]["message"]

    prompt = ctx.to_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    db.close()
