"""Tests for the time awareness module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.db import Database
from app.time import (
    build_time_packet,
    detect_session_start,
    format_duration,
    format_duration_compact,
    format_time_packet,
    relative_time,
    session_stats,
)


def _make_db(tmp_path, calls=None):
    """Create a test DB and optionally insert MCP calls.

    calls: list of (minutes_ago, tool_name) tuples.
    """
    db = Database(str(tmp_path / "test.db"))
    if calls:
        now = datetime.now(UTC)
        for minutes_ago, name in calls:
            ts = now - timedelta(minutes=minutes_ago)
            db.log_mcp_request(client_name="test", kind="tool", name=name, timestamp_override=ts)
    return db


# --- format_duration ---

def test_format_duration_seconds():
    assert format_duration(timedelta(seconds=3)) == "3 seconds"
    assert format_duration(timedelta(seconds=1)) == "1 second"


def test_format_duration_minutes():
    assert format_duration(timedelta(minutes=2, seconds=30)) == "2 minutes 30 seconds"
    assert format_duration(timedelta(minutes=1)) == "1 minute"


def test_format_duration_hours():
    assert format_duration(timedelta(hours=1, minutes=5)) == "1 hour 5 minutes"
    assert format_duration(timedelta(hours=2, minutes=30, seconds=15)) == "2 hours 30 minutes 15 seconds"


def test_format_duration_zero():
    assert format_duration(timedelta(0)) == "0 seconds"
    assert format_duration(timedelta(seconds=-5)) == "0 seconds"


# --- format_duration_compact ---

def test_format_duration_compact_seconds():
    assert format_duration_compact(timedelta(seconds=3)) == "3s"
    assert format_duration_compact(timedelta(seconds=45)) == "45s"


def test_format_duration_compact_minutes():
    assert format_duration_compact(timedelta(minutes=2, seconds=30)) == "2m30s"


def test_format_duration_compact_hours():
    assert format_duration_compact(timedelta(hours=1, minutes=5)) == "1h05m"
    assert format_duration_compact(timedelta(hours=2, minutes=0, seconds=0)) == "2h00m"


# --- relative_time ---

def test_relative_time_just_now():
    now = datetime.now(UTC)
    assert relative_time(now, now) == "just now"
    assert relative_time(now + timedelta(seconds=2), now) == "just now"


def test_relative_time_ago():
    now = datetime.now(UTC)
    past = now - timedelta(minutes=5)
    assert relative_time(past, now) == "5 minutes ago"


def test_relative_time_in_future():
    now = datetime.now(UTC)
    future = now + timedelta(hours=2)
    assert relative_time(future, now) == "in 2 hours"


def test_relative_time_singular():
    now = datetime.now(UTC)
    assert relative_time(now - timedelta(minutes=1), now) == "1 minute ago"
    assert relative_time(now - timedelta(hours=1), now) == "1 hour ago"


# --- detect_session_start ---

def test_detect_session_empty_db(tmp_path):
    db = _make_db(tmp_path)
    result = detect_session_start(db)
    assert result["session_start"] is None
    assert result["tool_call_count"] == 0
    assert result["is_active"] is False


def test_detect_session_single_call(tmp_path):
    db = _make_db(tmp_path, calls=[(1, "maos_get_context")])
    result = detect_session_start(db)
    assert result["session_start"] is not None
    assert result["tool_call_count"] == 1
    assert result["first_tool"] == "maos_get_context"
    assert result["is_active"] is True  # 1 minute ago < 10 min gap


def test_detect_session_multiple_calls(tmp_path):
    db = _make_db(tmp_path, calls=[
        (5, "maos_get_context"),
        (4, "maos_list_tasks"),
        (3, "maos_record_lesson"),
    ])
    result = detect_session_start(db)
    assert result["tool_call_count"] == 3
    assert result["first_tool"] == "maos_get_context"
    assert result["last_tool"] == "maos_record_lesson"
    assert result["is_active"] is True


def test_detect_session_gap_starts_new(tmp_path):
    """If there's a gap > 10 min, a new session starts."""
    db = _make_db(tmp_path, calls=[
        (30, "maos_get_context"),   # 30 min ago — old session
        (25, "maos_list_tasks"),    # 25 min ago — still old session (5 min gap)
        (2, "maos_record_lesson"),  # 2 min ago — new session (23 min gap)
    ])
    result = detect_session_start(db)
    assert result["tool_call_count"] == 1  # only the last call is in the new session
    assert result["first_tool"] == "maos_record_lesson"
    assert result["is_active"] is True


def test_detect_session_idle(tmp_path):
    """If the last call was > 10 min ago, session is not active."""
    db = _make_db(tmp_path, calls=[(15, "maos_get_context")])
    result = detect_session_start(db)
    assert result["is_active"] is False
    assert "15 minutes" in result["idle_human"]


# --- session_stats ---

def test_session_stats_empty(tmp_path):
    db = _make_db(tmp_path)
    stats = session_stats(db)
    assert stats["total_tool_calls"] == 0
    assert stats["calls_per_minute"] == 0


def test_session_stats_with_calls(tmp_path):
    db = _make_db(tmp_path, calls=[
        (5, "maos_get_context"),
        (4, "maos_list_tasks"),
        (3, "maos_get_context"),
    ])
    stats = session_stats(db)
    assert stats["total_tool_calls"] == 3
    assert "maos_get_context" in stats["tools_used"]
    assert "maos_list_tasks" in stats["tools_used"]
    assert stats["tool_frequency"]["maos_get_context"] == 2
    assert stats["tool_frequency"]["maos_list_tasks"] == 1


# --- build_time_packet ---

def test_build_time_packet(tmp_path):
    db = _make_db(tmp_path, calls=[(2, "maos_get_context")])
    packet = build_time_packet(db)
    assert "current_time" in packet
    assert "current_time_human" in packet
    assert "timezone" in packet
    assert "day_of_week" in packet
    assert "unix_timestamp" in packet
    assert "session" in packet
    assert "session_stats" in packet
    assert "relative_hints" in packet
    assert packet["session"]["tool_call_count"] == 1
    assert packet["session"]["is_active"] is True


def test_build_time_packet_empty(tmp_path):
    db = _make_db(tmp_path)
    packet = build_time_packet(db)
    assert packet["session"]["tool_call_count"] == 0
    assert packet["session"]["is_active"] is False
    assert len(packet["relative_hints"]) > 0


# --- format_time_packet ---

def test_format_time_packet(tmp_path):
    db = _make_db(tmp_path, calls=[(2, "maos_get_context")])
    packet = build_time_packet(db)
    text = format_time_packet(packet)
    assert "Current time:" in text
    assert "Session:" in text
    assert "Stats:" in text
    assert "Hints:" in text


def test_format_time_packet_empty(tmp_path):
    db = _make_db(tmp_path)
    packet = build_time_packet(db)
    text = format_time_packet(packet)
    assert "No MCP calls recorded" in text
