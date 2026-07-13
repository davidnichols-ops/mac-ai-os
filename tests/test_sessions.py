from datetime import UTC, datetime, timedelta

from app.db import Database
from app.events.bus import EventBus
from app.events.models import FILE_INDEXED, Event


def test_session_storage(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    sid = db.create_session("2025-01-15T10:00:00", "2025-01-15T12:00:00",
                            project="mac-ai-os", event_count=42)
    assert sid > 0
    session = db.get_session(sid)
    assert session["project"] == "mac-ai-os"
    assert session["event_count"] == 42

    db.update_session_summary(sid, "Added event bus", project="mac-ai-os")
    session = db.get_session(sid)
    assert session["summary"] == "Added event bus"

    sessions = db.recent_sessions()
    assert len(sessions) == 1
    assert db.session_count() == 1
    db.close()


def test_session_detection(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)

    # Create a burst of events, then a gap, then another burst.
    base = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)

    # First session: 10:00 - 10:15.
    for i in range(5):
        ts = (base + timedelta(minutes=i * 3)).isoformat()
        bus.publish(Event(FILE_INDEXED, {"path": f"/a/file{i}.py", "source": "/a"}, timestamp=ts))

    # Gap of 2 hours.

    # Second session: 12:00 - 12:10.
    base2 = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
    for i in range(3):
        ts = (base2 + timedelta(minutes=i * 3)).isoformat()
        bus.publish(Event(FILE_INDEXED, {"path": f"/b/file{i}.py", "source": "/b"}, timestamp=ts))

    from app.sessions import detect_sessions

    sessions = detect_sessions(db, gap_threshold=timedelta(minutes=30))
    assert len(sessions) == 2
    assert sessions[0]["event_count"] == 5
    assert sessions[1]["event_count"] == 3
    db.close()


def test_summarize_session(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)

    base = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    for i in range(3):
        ts = (base + timedelta(minutes=i)).isoformat()
        bus.publish(Event(FILE_INDEXED, {"path": f"/project/file{i}.py", "source": "/project"},
                         timestamp=ts))

    from app.sessions import detect_sessions, summarize_session

    sessions = detect_sessions(db)
    assert len(sessions) == 1

    data = summarize_session(db, sessions[0]["id"])
    assert data["event_count"] == 3
    assert len(data["files_touched"]) == 3
    assert data["commit_count"] == 0
    db.close()
