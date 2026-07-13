"""In-process event bus.

Simple pub/sub: publishers call `bus.publish(event)`, subscribers register
with `bus.subscribe(event_type, handler)`. Handlers are called synchronously
in the publisher's thread — this keeps the bus dependency-free and
predictable. For CPU-heavy subscribers, run their work in a thread inside
the handler.

The bus optionally persists every event to the SQLite `events` table if a
`Database` is attached. This gives us the activity timeline for free.

Usage:
    bus = EventBus()
    bus.subscribe(FILE_INDEXED, my_handler)
    bus.publish(file_indexed("/path", 5, "source"))
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from threading import RLock

from app.events.models import Event
from app.log import get_logger

log = get_logger(__name__)

EventHandler = Callable[[Event], None]


class EventBus:
    """Synchronous in-process event bus with optional SQLite persistence."""

    def __init__(self, db=None):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: list[EventHandler] = []
        self._db = db
        self._lock = RLock()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type. Use "*" for all events."""
        with self._lock:
            if event_type == "*":
                self._wildcard_handlers.append(handler)
            else:
                self._handlers[event_type].append(handler)
            log.debug("bus.subscribed", event_type=event_type, handler=handler.__name__)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        with self._lock:
            if event_type == "*":
                self._wildcard_handlers = [h for h in self._wildcard_handlers if h != handler]
            elif event_type in self._handlers:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]

    def publish(self, event: Event) -> None:
        """Dispatch an event to all matching handlers and persist to DB."""
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
            wildcard = list(self._wildcard_handlers)

        # Persist first so the event is recorded even if a handler crashes.
        if self._db is not None:
            try:
                self._persist(event)
            except Exception as e:
                log.warning("bus.persist_failed", event_type=event.type, error=str(e))

        # Dispatch to specific handlers.
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                log.warning("bus.handler_failed", event_type=event.type,
                            handler=handler.__name__, error=str(e))

        # Dispatch to wildcard handlers.
        for handler in wildcard:
            try:
                handler(event)
            except Exception as e:
                log.warning("bus.handler_failed", event_type=event.type,
                            handler=handler.__name__, error=str(e))

    def _persist(self, event: Event) -> None:
        import json

        self._db.execute(
            """INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)""",
            (event.type, json.dumps(event.payload), event.timestamp),
        )

    def handler_count(self) -> int:
        with self._lock:
            return sum(len(h) for h in self._handlers.values()) + len(self._wildcard_handlers)


# Module-level default bus (lazy). Components can use `get_bus()` or receive
# a bus via dependency injection. The runtime wires a shared bus.
_default_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_bus(bus: EventBus) -> None:
    global _default_bus
    _default_bus = bus


def reset_bus() -> None:
    global _default_bus
    _default_bus = None
