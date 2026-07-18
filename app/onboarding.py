"""Onboarding — assemble a briefing for a new AI session.

The problem this solves: every new Devin/Claude/agent session starts cold.
It has no memory of who the user is, how they like to work, what went wrong
last time, or what was in progress. This leads to avoidable failures —
the agent opens a PR with an unverified fix, uses the wrong commit style,
ignores a hard-won lesson, etc.

The onboarding packet is the antidote. It pulls together:

  1. User identity & preferences (high-confidence user_model entries)
  2. Hard-won lessons (lesson.* entries in user_model)
  3. Recent session summary (what was just being worked on)
  4. Active workspace + open tasks
  5. Known failure patterns (common_errors from the learning system)

The packet is returned as structured JSON. The calling agent is expected
to read it at the start of every session, before exploring any codebase.

Lessons are stored in the user_model table with keys prefixed `lesson.`.
Each lesson has:
  - key: lesson.<slug>  (e.g. lesson.verify_before_pr)
  - value: the lesson text (what to do / what not to do)
  - confidence: how strongly the lesson is held (starts at 0.9)
  - source: "incident" (learned from a failure) or "observed"
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import Database
from app.log import get_logger
from app.sessions import detect_sessions, summarize_session
from app.workspace import detect_active_workspace

log = get_logger(__name__)

# Keys in user_model that count as user-facing preferences (not internal).
_PREFERENCE_PREFIXES = (
    "user_", "coding_", "workflow", "knowledge_profile",
    "comm_", "git_", "review_",
)


def build_onboarding_packet(db: Database, *, lookback_hours: int = 48) -> dict[str, Any]:
    """Assemble a briefing for a new AI session.

    Args:
        db: MAOS database.
        lookback_hours: How far back to look for session/activity data.

    Returns:
        A dict with user, lessons, recent_session, workspace, tasks,
        and failure_patterns sections.
    """
    now = datetime.now(UTC)
    since = (now - timedelta(hours=lookback_hours)).isoformat()

    return {
        "timestamp": now.isoformat(),
        "user": _user_info(db),
        "lessons": _lessons(db),
        "recent_session": _recent_session(db),
        "workspace": _workspace_info(db),
        "open_tasks": _open_tasks(db),
        "failure_patterns": _failure_patterns(db),
        "recent_activity": _recent_activity(db, since),
        "time_awareness": _time_awareness(db),
    }


def format_onboarding_briefing(packet: dict[str, Any]) -> str:
    """Render the onboarding packet as a readable text briefing.

    This is what an AI agent sees — concise, scannable, actionable.
    """
    lines: list[str] = []
    user = packet.get("user", {})

    # --- User ---
    lines.append("=== USER ===")
    if user.get("bio"):
        lines.append(f"Bio: {user['bio']}")
    if user.get("communication_style"):
        lines.append(f"Communication: {user['communication_style']}")
    if user.get("coding_style"):
        lines.append(f"Coding: {user['coding_style']}")
    if user.get("workflow"):
        lines.append(f"Workflow: {user['workflow']}")
    if user.get("github"):
        lines.append(f"GitHub: {user['github']}")

    # --- Lessons (the critical part) ---
    lessons = packet.get("lessons", [])
    lines.append("")
    lines.append(f"=== LESSONS ({len(lessons)}) ===")
    if not lessons:
        lines.append("(no lessons recorded yet)")
    else:
        for lesson in lessons:
            lines.append(f"  [{lesson['confidence']:.0%}] {lesson['slug']}: {lesson['text']}")

    # --- Recent session ---
    session = packet.get("recent_session", {})
    lines.append("")
    lines.append("=== RECENT SESSION ===")
    if session.get("active"):
        lines.append(f"  Project: {session.get('project', 'unknown')}")
        lines.append(f"  Workspace: {session.get('workspace', 'unknown')}")
        lines.append(f"  Events: {session.get('event_count', 0)}")
        if session.get("summary"):
            lines.append(f"  Summary: {session['summary'][:200]}")
        if session.get("commits"):
            lines.append("  Recent commits:")
            for c in session["commits"][:3]:
                lines.append(f"    {c.get('hash', '')[:8]} {c.get('message', '')[:60]}")
    else:
        lines.append("  (no recent session detected)")

    # --- Workspace ---
    ws = packet.get("workspace", {})
    lines.append("")
    lines.append("=== ACTIVE WORKSPACE ===")
    if ws.get("name"):
        lines.append(f"  {ws['name']} ({ws.get('path', '')})")
        if ws.get("language"):
            lines.append(f"  Language: {ws['language']}")
    else:
        lines.append("  (none detected)")

    # --- Open tasks ---
    tasks = packet.get("open_tasks", [])
    lines.append("")
    lines.append(f"=== OPEN TASKS ({len(tasks)}) ===")
    if tasks:
        for t in tasks[:5]:
            lines.append(f"  [{t['priority']}] {t['title']}")
    else:
        lines.append("  (none)")

    # --- Failure patterns ---
    failures = packet.get("failure_patterns", [])
    lines.append("")
    lines.append(f"=== KNOWN FAILURE PATTERNS ({len(failures)}) ===")
    if failures:
        for f in failures:
            lines.append(f"  {f['action']}: {f['errors']}")
    else:
        lines.append("  (none recorded)")

    # --- Time awareness ---
    time_info = packet.get("time_awareness", {})
    lines.append("")
    lines.append("=== TIME AWARENESS ===")
    if time_info.get("current_time_human"):
        lines.append(f"  Current time: {time_info['current_time_human']}")
        lines.append(f"  Day: {time_info.get('day_of_week', '?')}")
        sess = time_info.get("session", {})
        if sess.get("start_time"):
            lines.append(f"  Session elapsed: {sess.get('elapsed', '?')} ({sess.get('tool_call_count', 0)} tool calls)")
            lines.append(f"  Session active: {'yes' if sess.get('is_active') else 'no (idle ' + sess.get('idle_time', '?') + ')'}")
            lines.append(f"  First tool: {sess.get('first_tool', '?')}")
        else:
            lines.append("  (no previous tool calls — this is the first call in a new session)")
        hints = time_info.get("relative_hints", [])
        if hints:
            lines.append("  Hints:")
            for h in hints:
                lines.append(f"    • {h}")
    else:
        lines.append("  (time awareness unavailable)")

    return "\n".join(lines)


def record_lesson(
    db: Database,
    slug: str,
    text: str,
    *,
    confidence: float = 0.9,
    source: str = "incident",
) -> dict[str, Any]:
    """Record a hard-won lesson in the user_model.

    Args:
        db: MAOS database.
        slug: Short identifier (e.g. "verify_before_pr").
        text: The lesson text — what to do or what not to do.
        confidence: How strongly to hold this lesson (0-1).
        source: Where the lesson came from ("incident", "observed", "manual").

    Returns:
        The stored lesson as a dict.
    """
    key = f"lesson.{slug}"
    db.set_user_model_entry(key, text, confidence=confidence, source=source)
    log.info("onboarding.lesson.recorded", slug=slug, confidence=confidence, source=source)
    return {"slug": slug, "text": text, "confidence": confidence, "source": source}


def list_lessons(db: Database) -> list[dict[str, Any]]:
    """List all recorded lessons from the user_model."""
    entries = db.all_user_model_entries()
    lessons = []
    for e in entries:
        if not e["key"].startswith("lesson."):
            continue
        lessons.append({
            "slug": e["key"].removeprefix("lesson."),
            "text": e["value"],
            "confidence": e["confidence"],
            "source": e["source"],
            "updated_at": e["updated_at"],
        })
    # Sort by confidence descending, then alphabetically.
    lessons.sort(key=lambda lesson: (-lesson["confidence"], lesson["slug"]))
    return lessons


# --- internal helpers ------------------------------------------------------


def _user_info(db: Database) -> dict[str, Any]:
    """Extract user-facing preferences from user_model."""
    entries = db.all_user_model_entries()
    info: dict[str, Any] = {}
    for e in entries:
        key = e["key"]
        # Skip internal/system keys and action stats.
        if key.startswith("_") or key.startswith("action.") or key.startswith("workflow."):
            continue
        if key.startswith("lesson."):
            continue  # lessons are separate section
        # Map known keys to friendly names.
        if key == "user_bio":
            info["bio"] = e["value"]
        elif key == "user_communication_style":
            info["communication_style"] = e["value"]
        elif key == "coding_style":
            info["coding_style"] = e["value"]
        elif key == "workflow":
            info["workflow"] = e["value"]
        elif key == "user_github":
            info["github"] = e["value"]
        elif key == "user_email":
            info["email"] = e["value"]
        elif key == "user_twitter":
            info["twitter"] = e["value"]
        elif key == "knowledge_profile_path":
            info["knowledge_profile"] = e["value"]
        elif any(key.startswith(p) for p in _PREFERENCE_PREFIXES):
            # Catch-all for other preference-style keys.
            info[key] = e["value"]
    return info


def _lessons(db: Database) -> list[dict[str, Any]]:
    """Get all recorded lessons."""
    return list_lessons(db)


def _recent_session(db: Database) -> dict[str, Any]:
    """Get the most recent session with context."""
    try:
        # Ensure sessions are detected.
        if db.session_count() == 0 or db.event_count() > 0:
            detect_sessions(db)
        latest = db.latest_session()
        if not latest:
            return {"active": False}
        s = dict(latest)
        # Try to get a richer summary.
        try:
            summary_data = summarize_session(db, s["id"])
            commits = summary_data.get("commits", [])[:5]
        except Exception:
            commits = []
        return {
            "active": True,
            "id": s.get("id"),
            "start_time": s.get("start_time", ""),
            "end_time": s.get("end_time", ""),
            "project": s.get("project", ""),
            "workspace": s.get("workspace", ""),
            "event_count": s.get("event_count", 0),
            "summary": s.get("summary", ""),
            "commits": [
                {"hash": c.get("hash", ""), "message": c.get("message", "")}
                for c in commits
            ],
        }
    except Exception as e:
        log.warning("onboarding.session_failed", error=str(e))
        return {"active": False, "error": str(e)}


def _workspace_info(db: Database) -> dict[str, Any]:
    """Get the active workspace."""
    try:
        ws = detect_active_workspace(db)
        if not ws:
            return {"name": None, "path": None}
        return {
            "name": ws.get("name", ""),
            "path": ws.get("path", ""),
            "language": ws.get("language", ""),
            "git_repo": ws.get("git_repo", False),
        }
    except Exception as e:
        log.warning("onboarding.workspace_failed", error=str(e))
        return {"name": None, "path": None, "error": str(e)}


def _open_tasks(db: Database) -> list[dict[str, Any]]:
    """Get pending tasks."""
    try:
        rows = db.pending_tasks()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "priority": r["priority"],
                "workspace": r["related_workspace"] or "",
            }
            for r in rows
        ]
    except Exception:
        return []


def _failure_patterns(db: Database) -> list[dict[str, Any]]:
    """Extract known failure patterns from the learning system."""
    entries = db.all_user_model_entries()
    patterns = []
    for e in entries:
        if e["key"].startswith("action.") and e["key"].endswith(".common_errors"):
            action = e["key"].split(".")[1]
            patterns.append({"action": action, "errors": e["value"]})
    return patterns


def _recent_activity(db: Database, since: str) -> dict[str, Any]:
    """Summary of recent events by type."""
    try:
        rows = db.query(
            "SELECT type, COUNT(*) as count FROM events "
            "WHERE timestamp >= ? GROUP BY type ORDER BY count DESC LIMIT 10",
            (since,),
        )
        total_row = db.query_one(
            "SELECT COUNT(*) as n FROM events WHERE timestamp >= ?", (since,)
        )
        return {
            "total": total_row["n"] if total_row else 0,
            "by_type": {r["type"]: r["count"] for r in rows},
        }
    except Exception:
        return {"total": 0, "by_type": {}}


def _time_awareness(db: Database) -> dict[str, Any]:
    """Time awareness for the onboarding briefing.

    Gives the agent a sense of time at the start of a session —
    current time, how long the previous session lasted, and whether
    this is a continuation of an existing session or a fresh start.
    """
    try:
        from app.time import build_time_packet
        return build_time_packet(db)
    except Exception as e:
        log.warning("onboarding.time_failed", error=str(e))
        return {"available": False, "error": str(e)}
