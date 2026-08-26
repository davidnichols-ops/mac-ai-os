"""Tests for app.session_search — FTS5 full-text search over sessions."""

from __future__ import annotations

from app.db import Database
from app.session_search import SessionSearch, _sanitize_query


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    # Two sessions with distinct summaries and events.
    db.create_session(
        "2025-01-15T10:00:00", "2025-01-15T12:00:00",
        project="mac-ai-os", summary="Added FTS5 session search to MAOS",
        event_count=3,
    )
    db.create_session(
        "2025-01-16T09:00:00", "2025-01-16T11:00:00",
        project="trustcard", summary="Wrapped MCP proxy with Ed25519 signing",
        event_count=2,
    )
    # Insert events within session time ranges.
    db.execute(
        "INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)",
        ("FILE_INDEXED", '{"path": "/a/search.py", "source": "/a"}', "2025-01-15T10:30:00"),
    )
    db.execute(
        "INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)",
        ("GIT_COMMIT", '{"message": "feat: fts5 search", "repo": "mac-ai-os"}', "2025-01-15T11:00:00"),
    )
    db.execute(
        "INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)",
        ("GIT_COMMIT", '{"message": "fix: ed25519 verify", "repo": "trustcard"}', "2025-01-16T10:00:00"),
    )
    return db


def test_rebuild_indexes_sessions(tmp_path):
    db = _make_db(tmp_path)
    searcher = SessionSearch(db)
    count = searcher.rebuild()
    assert count == 2


def test_search_matches_summary(tmp_path):
    db = _make_db(tmp_path)
    searcher = SessionSearch(db)
    searcher.rebuild()
    results = searcher.search("session search")
    assert len(results) >= 1
    assert results[0]["project"] == "mac-ai-os"


def test_search_matches_event_payload(tmp_path):
    db = _make_db(tmp_path)
    searcher = SessionSearch(db)
    searcher.rebuild()
    results = searcher.search("ed25519")
    assert len(results) >= 1
    assert results[0]["project"] == "trustcard"


def test_search_empty_query_returns_nothing(tmp_path):
    db = _make_db(tmp_path)
    searcher = SessionSearch(db)
    searcher.rebuild()
    assert searcher.search("") == []
    assert searcher.search("   ") == []


def test_search_no_match(tmp_path):
    db = _make_db(tmp_path)
    searcher = SessionSearch(db)
    searcher.rebuild()
    assert searcher.search("zzzznomatch") == []


def test_sanitize_query_neutralizes_operators():
    # FTS5 special chars should be stripped, hyphenated terms quoted.
    out = _sanitize_query('foo "bar" (baz) * ^')
    assert "foo" in out
    assert "bar" in out


def test_continuity_signal_returns_relevant(tmp_path):
    db = _make_db(tmp_path)
    searcher = SessionSearch(db)
    searcher.rebuild()
    signal = searcher.continuity_signal("trustcard proxy")
    assert len(signal) >= 1
    assert signal[0]["project"] == "trustcard"
