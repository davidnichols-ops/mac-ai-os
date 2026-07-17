"""Time awareness for AI agents.

AI agents have no native concept of time. They don't know what time it is,
how long they've been working, or how long a task took. This module gives
agents that awareness by querying the MAOS event log and MCP request log.

The key insight: a "Devin session" is a contiguous sequence of MCP tool
calls. The session starts at the first tool call and continues as long as
calls keep coming in (gap threshold: 10 minutes). This mirrors how humans
think about work sessions — you're "in a session" until you step away.

This module provides:
  - Current time (wall clock, with timezone and human-readable format)
  - Session elapsed time (how long the current Devin session has been running)
  - Session start time (when the first tool call in this session happened)
  - Task duration estimation (time between related tool calls)
  - Relative time phrasing ("3 minutes ago", "in 2 hours")
  - Time until/since a target timestamp
  - Session statistics (tool calls per minute, idle time, active time)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import Database

# Gap threshold: if no MCP tool call happens for this duration, a new
# session is considered to have started. 10 minutes matches typical
# "stepping away from the keyboard" behavior.
SESSION_GAP = timedelta(minutes=10)


def now() -> datetime:
    """Current UTC time."""
    return datetime.now(UTC)


def format_duration(td: timedelta) -> str:
    """Format a timedelta as a human-readable duration.

    Examples:
        3s          → "3 seconds"
        1m 30s      → "1 minute 30 seconds"
        1h 5m       → "1 hour 5 minutes"
        2h 30m 15s  → "2 hours 30 minutes 15 seconds"
    """
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "0 seconds"
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return " ".join(parts)


def format_duration_compact(td: timedelta) -> str:
    """Format a timedelta as a compact string.

    Examples:
        3s          → "3s"
        1m 30s      → "1m30s"
        1h 5m       → "1h05m"
        2h 0m 0s    → "2h00m"
    """
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        return "0s"
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def relative_time(target: datetime, reference: datetime | None = None) -> str:
    """Phrase a timestamp relative to now or a reference time.

    Examples:
        3 seconds ago
        5 minutes ago
        2 hours ago
        in 10 minutes
        in 3 hours
    """
    if reference is None:
        reference = now()
    delta = target - reference
    abs_seconds = abs(int(delta.total_seconds()))

    if abs_seconds < 5:
        return "just now"
    if abs_seconds < 60:
        value = abs_seconds
        unit = "second"
    elif abs_seconds < 3600:
        value = abs_seconds // 60
        unit = "minute"
    elif abs_seconds < 86400:
        value = abs_seconds // 3600
        unit = "hour"
    elif abs_seconds < 604800:
        value = abs_seconds // 86400
        unit = "day"
    else:
        value = abs_seconds // 604800
        unit = "week"

    plural = "s" if value != 1 else ""
    direction = "ago" if delta.total_seconds() < 0 else "in"
    if abs_seconds < 5:
        return "just now"
    return f"{direction} {value} {unit}{plural}" if direction == "in" else f"{value} {unit}{plural} {direction}"


def detect_session_start(db: Database, gap: timedelta = SESSION_GAP) -> dict[str, Any]:
    """Detect the start of the current Devin session by looking at MCP request gaps.

    A session is a contiguous sequence of MCP tool calls. If there's a gap
    larger than `gap` (default 10 minutes) between two consecutive calls,
    the second call starts a new session.

    Returns:
        Dict with:
        - session_start: ISO timestamp of the first call in this session
        - session_start_dt: datetime object
        - elapsed: timedelta since session start
        - elapsed_human: human-readable duration string
        - elapsed_compact: compact duration string
        - tool_call_count: number of MCP calls in this session
        - first_tool: name of the first tool called in this session
        - last_tool: name of the most recent tool called
        - last_tool_time: ISO timestamp of the most recent call
        - idle_time: timedelta since the last tool call (how long agent has been idle)
        - idle_human: human-readable idle time
        - is_active: True if the last call was within the gap threshold
    """
    rows = db.query(
        "SELECT timestamp, name FROM mcp_requests ORDER BY timestamp ASC",
        (),
    )

    if not rows:
        return {
            "session_start": None,
            "session_start_dt": None,
            "elapsed": timedelta(0),
            "elapsed_human": "no MCP calls recorded",
            "elapsed_compact": "0s",
            "tool_call_count": 0,
            "first_tool": None,
            "last_tool": None,
            "last_tool_time": None,
            "idle_time": timedelta(0),
            "idle_human": "n/a",
            "is_active": False,
        }

    # Parse all timestamps
    calls = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            calls.append((ts, row["name"]))
        except (ValueError, KeyError):
            continue

    if not calls:
        return {
            "session_start": None,
            "session_start_dt": None,
            "elapsed": timedelta(0),
            "elapsed_human": "no valid timestamps",
            "elapsed_compact": "0s",
            "tool_call_count": 0,
            "first_tool": None,
            "last_tool": None,
            "last_tool_time": None,
            "idle_time": timedelta(0),
            "idle_human": "n/a",
            "is_active": False,
        }

    # Find session boundaries by looking for gaps
    session_start_idx = 0
    for i in range(1, len(calls)):
        gap_between = calls[i][0] - calls[i - 1][0]
        if gap_between > gap:
            session_start_idx = i

    session_start_dt = calls[session_start_idx][0]
    first_tool = calls[session_start_idx][1]
    last_tool = calls[-1][1]
    last_tool_time = calls[-1][0]
    current = now()

    elapsed = current - session_start_dt
    idle_time = current - last_tool_time
    tool_call_count = len(calls) - session_start_idx

    return {
        "session_start": session_start_dt.isoformat(),
        "session_start_dt": session_start_dt,
        "elapsed": elapsed,
        "elapsed_human": format_duration(elapsed),
        "elapsed_compact": format_duration_compact(elapsed),
        "tool_call_count": tool_call_count,
        "first_tool": first_tool,
        "last_tool": last_tool,
        "last_tool_time": last_tool_time.isoformat(),
        "idle_time": idle_time,
        "idle_human": format_duration(idle_time),
        "idle_compact": format_duration_compact(idle_time),
        "is_active": idle_time < gap,
    }


def session_stats(db: Database, gap: timedelta = SESSION_GAP) -> dict[str, Any]:
    """Compute statistics for the current session.

    Returns:
        Dict with:
        - total_tool_calls: total MCP calls in this session
        - calls_per_minute: average calls per minute of active time
        - active_time: time between first and last call in session
        - active_human: human-readable active time
        - idle_time: time since last call
        - idle_human: human-readable idle time
        - tools_used: list of unique tool names called
        - tool_frequency: dict of tool_name -> call count
        - busiest_minute: the minute with the most calls (if any)
    """
    session = detect_session_start(db, gap)
    if session["tool_call_count"] == 0:
        return {
            "total_tool_calls": 0,
            "calls_per_minute": 0,
            "active_time": timedelta(0),
            "active_human": "n/a",
            "idle_time": timedelta(0),
            "idle_human": "n/a",
            "tools_used": [],
            "tool_frequency": {},
        }

    # Get all calls in this session
    rows = db.query(
        "SELECT timestamp, name FROM mcp_requests WHERE timestamp >= ? ORDER BY timestamp ASC",
        (session["session_start"],),
    )

    calls = []
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            calls.append((ts, row["name"]))
        except (ValueError, KeyError):
            continue

    if not calls:
        return {
            "total_tool_calls": 0,
            "calls_per_minute": 0,
            "active_time": timedelta(0),
            "active_human": "n/a",
            "idle_time": timedelta(0),
            "idle_human": "n/a",
            "tools_used": [],
            "tool_frequency": {},
        }

    first_time = calls[0][0]
    last_time = calls[-1][0]
    active_time = last_time - first_time

    tool_frequency: dict[str, int] = {}
    for _, name in calls:
        tool_frequency[name] = tool_frequency.get(name, 0) + 1

    active_minutes = active_time.total_seconds() / 60
    calls_per_minute = len(calls) / active_minutes if active_minutes > 0 else 0

    current = now()
    idle_time = current - last_time

    return {
        "total_tool_calls": len(calls),
        "calls_per_minute": round(calls_per_minute, 2),
        "active_time": active_time,
        "active_human": format_duration(active_time),
        "idle_time": idle_time,
        "idle_human": format_duration(idle_time),
        "tools_used": sorted(tool_frequency.keys()),
        "tool_frequency": dict(sorted(tool_frequency.items(), key=lambda x: -x[1])),
    }


def build_time_packet(db: Database) -> dict[str, Any]:
    """Build a complete time awareness packet for an AI agent.

    This is the main entry point — it combines all time information into
    a single dict that can be returned as JSON from an MCP tool.

    Returns:
        Dict with:
        - current_time: ISO timestamp
        - current_time_human: human-readable ("3:45 PM PDT, July 17, 2026")
        - timezone: local timezone name
        - day_of_week: e.g. "Thursday"
        - session: session detection result
        - session_stats: session statistics
        - relative: relative time hints ("2 minutes since last tool call")
    """
    current = now()
    session = detect_session_start(db)
    stats = session_stats(db)

    # Get local time for human-readable display
    import time as _time

    local_time = datetime.now()  # local time
    tz_offset = _time.strftime("%z", _time.localtime())
    tz_name = _time.strftime("%Z", _time.localtime())
    if not tz_name or tz_name == "":
        # Derive timezone abbreviation from offset
        if tz_offset:
            sign = "+" if tz_offset[0] != "-" else "-"
            hours = int(tz_offset[1:3])
            tz_name = f"UTC{sign}{hours}"
        else:
            tz_name = "local"
    human = local_time.strftime(f"%-I:%M %p {tz_name}, %B %-d, %Y")
    day = local_time.strftime("%A")

    # Relative hints
    relative_hints = []
    if session["is_active"]:
        relative_hints.append(f"Session is active — {session['elapsed_human']} elapsed, {stats['total_tool_calls']} tool calls")
    else:
        last_time_str = session["last_tool_time"]
        if last_time_str:
            try:
                last_dt = datetime.fromisoformat(last_time_str)
                relative_hints.append(f"Session was idle for {session['idle_human']} — last tool call was {relative_time(last_dt)}")
            except ValueError:
                relative_hints.append(f"Session was idle for {session['idle_human']}")
        else:
            relative_hints.append("No tool calls recorded yet")

    if stats["calls_per_minute"] > 0:
        relative_hints.append(f"Average {stats['calls_per_minute']} tool calls per minute of active time")

    return {
        "current_time": current.isoformat(),
        "current_time_human": human,
        "timezone": tz_name,
        "day_of_week": day,
        "unix_timestamp": int(current.timestamp()),
        "session": {
            "start_time": session["session_start"],
            "elapsed": session["elapsed_human"],
            "elapsed_compact": session["elapsed_compact"],
            "tool_call_count": session["tool_call_count"],
            "first_tool": session["first_tool"],
            "last_tool": session["last_tool"],
            "last_tool_time": session["last_tool_time"],
            "idle_time": session["idle_human"],
            "is_active": session["is_active"],
        },
        "session_stats": {
            "total_tool_calls": stats["total_tool_calls"],
            "calls_per_minute": stats["calls_per_minute"],
            "active_time": stats["active_human"],
            "tools_used": stats["tools_used"],
            "tool_frequency": stats["tool_frequency"],
        },
        "relative_hints": relative_hints,
    }


def format_time_packet(packet: dict[str, Any]) -> str:
    """Format a time packet as a human-readable text block for CLI output."""
    lines = []
    lines.append(f"Current time: {packet['current_time_human']}")
    lines.append(f"Day: {packet['day_of_week']}")
    lines.append(f"Timezone: {packet['timezone']}")
    lines.append("")
    lines.append("Session:")
    sess = packet["session"]
    if sess["start_time"]:
        lines.append(f"  Started: {sess['start_time']}")
        lines.append(f"  Elapsed: {sess['elapsed']} ({sess['tool_call_count']} tool calls)")
        lines.append(f"  First tool: {sess['first_tool']}")
        lines.append(f"  Last tool: {sess['last_tool']} ({sess['idle_time']} ago)")
        lines.append(f"  Active: {'yes' if sess['is_active'] else 'no (idle)'}")
    else:
        lines.append("  No MCP calls recorded yet.")
    lines.append("")
    lines.append("Stats:")
    stats = packet["session_stats"]
    if stats["total_tool_calls"] > 0:
        lines.append(f"  Tool calls: {stats['total_tool_calls']}")
        lines.append(f"  Calls/min: {stats['calls_per_minute']}")
        lines.append(f"  Active time: {stats['active_time']}")
        lines.append(f"  Tools used: {', '.join(stats['tools_used'][:10])}")
        if len(stats["tools_used"]) > 10:
            lines.append(f"    ... and {len(stats['tools_used']) - 10} more")
    else:
        lines.append("  No stats available.")
    lines.append("")
    lines.append("Hints:")
    for hint in packet["relative_hints"]:
        lines.append(f"  • {hint}")
    return "\n".join(lines)
