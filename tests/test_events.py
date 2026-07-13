from app.events.bus import EventBus
from app.events.models import FILE_INDEXED, Event, file_indexed


def test_event_creation():
    e = file_indexed("/path/to/file.py", 5, "source1")
    assert e.type == FILE_INDEXED
    assert e.payload["path"] == "/path/to/file.py"
    assert e.payload["chunks"] == 5
    assert e.timestamp  # auto-generated


def test_bus_subscribe_publish():
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe(FILE_INDEXED, lambda e: received.append(e))

    e = file_indexed("/test.py", 3, "src")
    bus.publish(e)

    assert len(received) == 1
    assert received[0].payload["path"] == "/test.py"


def test_bus_wildcard():
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("*", lambda e: received.append(e))

    bus.publish(file_indexed("/a.py", 1, "src"))
    bus.publish(Event("OTHER_EVENT", {"x": 1}))

    assert len(received) == 2


def test_bus_persists_to_db(tmp_path):
    from app.db import Database

    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)

    bus.publish(file_indexed("/test.py", 3, "src"))
    bus.publish(Event("OTHER", {"y": 2}))

    assert db.event_count() == 2
    events = db.query_events(event_type=FILE_INDEXED)
    assert len(events) == 1
    db.close()


def test_bus_handler_isolation():
    """A failing handler should not prevent other handlers from running."""
    bus = EventBus()
    received: list[Event] = []

    def crashing_handler(e):
        raise RuntimeError("boom")

    bus.subscribe(FILE_INDEXED, crashing_handler)
    bus.subscribe(FILE_INDEXED, lambda e: received.append(e))

    bus.publish(file_indexed("/x.py", 1, "src"))
    assert len(received) == 1  # second handler still ran


def test_bus_unsubscribe():
    bus = EventBus()
    received: list[Event] = []
    def handler(e):
        received.append(e)

    bus.subscribe(FILE_INDEXED, handler)
    bus.publish(file_indexed("/a.py", 1, "src"))
    assert len(received) == 1

    bus.unsubscribe(FILE_INDEXED, handler)
    bus.publish(file_indexed("/b.py", 1, "src"))
    assert len(received) == 1  # no new events
