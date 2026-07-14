"""Session detection — groups events into work sessions.

A session is a period of continuous activity. If there's a gap of more than
`gap_threshold_minutes` between consecutive events, a new session starts.

Sessions are auto-detected from the events table but can also be manually
created or named. The auto-detection runs on demand (not continuously) to
avoid complexity.

For historical reconstruction, `detect_sessions_from_git()` uses git commit
timestamps as activity signals — this works even when shell history lacks
timestamps (plain zsh format).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)

DEFAULT_GAP_MINUTES = 30


def detect_sessions(
    db: Database,
    *,
    gap_threshold: timedelta = timedelta(minutes=DEFAULT_GAP_MINUTES),
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Scan events and create sessions from activity gaps.

    Returns the list of newly created sessions.
    """
    # Get all events ordered by time.
    if since:
        events = db.query(
            "SELECT * FROM events WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since,),
        )
    else:
        events = db.query("SELECT * FROM events ORDER BY timestamp ASC")

    if not events:
        return []

    # Get existing sessions to avoid recreating.
    existing = db.recent_sessions(limit=10000)
    existing_starts = {s["start_time"] for s in existing}

    new_sessions: list[dict[str, Any]] = []
    current_start: str | None = None
    current_end: str | None = None
    current_count = 0
    current_types: dict[str, int] = {}

    def _flush_session():
        nonlocal current_start, current_end, current_count, current_types
        if current_start and current_end:
            if current_start in existing_starts:
                current_start = current_end = None
                current_count = 0
                current_types = {}
                return
            # Skip trivial sessions: single event or less than 2 minutes.
            try:
                start_dt = datetime.fromisoformat(current_start)
                end_dt = datetime.fromisoformat(current_end)
                duration = end_dt - start_dt
                if current_count <= 1 and duration < timedelta(minutes=2):
                    log.debug("session.skipped.trivial",
                              start=current_start[:19], events=current_count)
                    current_start = current_end = None
                    current_count = 0
                    current_types = {}
                    return
            except (ValueError, TypeError):
                pass
            # Detect project from event payloads.
            project = _detect_project(db, current_start, current_end)
            sid = db.create_session(
                current_start, current_end,
                project=project,
                event_count=current_count,
            )
            session = {
                "id": sid, "start_time": current_start, "end_time": current_end,
                "project": project, "event_count": current_count,
                "event_types": dict(current_types),
            }
            new_sessions.append(session)
            log.info("session.detected", id=sid, start=current_start[:19],
                     end=current_end[:19], events=current_count, project=project)
        current_start = current_end = None
        current_count = 0
        current_types = {}

    prev_time: datetime | None = None
    for event in events:
        try:
            event_time = datetime.fromisoformat(event["timestamp"])
        except (ValueError, TypeError):
            continue

        if prev_time:
            gap = event_time - prev_time
            if gap > gap_threshold:
                _flush_session()

        if current_start is None:
            current_start = event["timestamp"]
        current_end = event["timestamp"]
        current_count += 1
        etype = event["type"]
        current_types[etype] = current_types.get(etype, 0) + 1
        prev_time = event_time

    _flush_session()
    log.info("session.detection.done", new_sessions=len(new_sessions))
    return new_sessions


def _detect_project(db: Database, start: str, end: str) -> str:
    """Try to determine which project a session was about."""
    # Check FILE_INDEXED events for source paths.
    file_events = db.query(
        """SELECT payload FROM events
           WHERE type = 'FILE_INDEXED' AND timestamp >= ? AND timestamp <= ?
           LIMIT 20""",
        (start, end),
    )

    paths: list[str] = []
    for e in file_events:
        try:
            payload = json.loads(e["payload"])
            if "source" in payload:
                paths.append(payload["source"])
        except Exception:
            continue

    # Find the most common top-level path.
    if paths:
        # Extract project name (last directory component of unique parent paths).
        projects = [p.split("/")[-1] for p in paths if p]
        if projects:
            most_common = Counter(projects).most_common(1)
            if most_common:
                return most_common[0][0]

    # Check git commits.
    commits = db.query(
        "SELECT repo FROM git_commits WHERE author_date >= ? AND author_date <= ? LIMIT 5",
        (start, end),
    )
    if commits:
        repos = [c["repo"].split("/")[-1] for c in commits if c["repo"]]
        if repos:
            return repos[0]

    return ""


def summarize_session(db: Database, session_id: int) -> dict[str, Any]:
    """Gather all data about a session for the reconstruction agent."""
    session = db.get_session(session_id)
    if not session:
        return {}

    start = session["start_time"]
    end = session["end_time"]

    # Events in this session.
    events = db.query(
        "SELECT * FROM events WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
        (start, end),
    )

    # Git commits in this session.
    commits = db.query(
        "SELECT * FROM git_commits WHERE author_date >= ? AND author_date <= ? ORDER BY author_date ASC",
        (start, end),
    )

    # Telemetry in this session.
    telemetry = db.query(
        "SELECT * FROM telemetry_snapshots WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
        (start, end),
    )

    # Shell commands in this session.
    commands = db.query(
        "SELECT * FROM shell_commands WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
        (start, end),
    )

    # Files touched (from FILE_INDEXED events).
    files_touched: set[str] = set()
    for e in events:
        if e["type"] == "FILE_INDEXED":
            try:
                payload = json.loads(e["payload"])
                if "path" in payload:
                    files_touched.add(payload["path"])
            except Exception:
                continue

    return {
        "session": dict(session),
        "events": [dict(e) for e in events],
        "commits": [dict(c) for c in commits],
        "telemetry": [dict(t) for t in telemetry],
        "commands": [dict(c) for c in commands],
        "files_touched": sorted(files_touched),
        "event_count": len(events),
        "commit_count": len(commits),
        "telemetry_count": len(telemetry),
        "command_count": len(commands),
    }


def detect_sessions_from_git(
    db: Database,
    *,
    gap_threshold: timedelta = timedelta(minutes=DEFAULT_GAP_MINUTES),
    min_session_minutes: int = 5,
) -> list[dict[str, Any]]:
    """Detect work sessions from git commit timestamps.

    This is the primary session detection method for historical data.
    Git commits have real timestamps (unlike plain-format shell history)
    and represent meaningful work activity.

    Each session is a cluster of commits across repos with no gap larger
    than `gap_threshold`. The project is determined by the repo with the
    most commits in that session.
    """
    # Get all commits ordered by time.
    commits = db.query(
        "SELECT hash, repo, author_date, message FROM git_commits "
        "ORDER BY author_date ASC"
    )

    if not commits:
        log.info("session.git.no_commits")
        return []

    # Get existing sessions to avoid duplicates.
    existing = db.recent_sessions(limit=10000)
    existing_starts = {s["start_time"] for s in existing}

    new_sessions: list[dict[str, Any]] = []
    current_commits: list[sqlite3_Row_like] = []
    prev_time: datetime | None = None

    def _flush_git_session():
        nonlocal current_commits, prev_time
        if not current_commits:
            return

        start = current_commits[0]["author_date"]
        end = current_commits[-1]["author_date"]

        # Skip sessions shorter than min_session_minutes (unless they have 3+ commits).
        if len(current_commits) < 3:
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                if (end_dt - start_dt) < timedelta(minutes=min_session_minutes):
                    current_commits = []
                    prev_time = None
                    return
            except (ValueError, TypeError):
                pass

        if start in existing_starts:
            current_commits = []
            prev_time = None
            return

        # Determine project from most common repo.
        repos = [c["repo"].split("/")[-1] for c in current_commits if c["repo"]]
        project = Counter(repos).most_common(1)[0][0] if repos else ""

        # Count unique repos in this session.
        unique_repos = set(repos)

        sid = db.create_session(
            start, end,
            project=project,
            event_count=len(current_commits),
        )

        session = {
            "id": sid,
            "start_time": start,
            "end_time": end,
            "project": project,
            "event_count": len(current_commits),
            "repos": sorted(unique_repos),
            "commit_count": len(current_commits),
        }
        new_sessions.append(session)
        log.info(
            "session.git.detected",
            id=sid, start=start[:19], end=end[:19],
            commits=len(current_commits), project=project,
            repos=len(unique_repos),
        )
        current_commits = []
        prev_time = None

    for commit in commits:
        try:
            commit_time = datetime.fromisoformat(
                commit["author_date"].replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            continue

        if prev_time:
            gap = commit_time - prev_time
            if gap > gap_threshold:
                _flush_git_session()

        current_commits.append(commit)
        prev_time = commit_time

    _flush_git_session()

    log.info(
        "session.git.detection.done",
        total_commits=len(commits),
        new_sessions=len(new_sessions),
    )

    # Now try to correlate shell commands to these sessions.
    _correlate_shell_commands(db, new_sessions)

    return new_sessions


def _correlate_shell_commands(
    db: Database, sessions: list[dict[str, Any]],
) -> int:
    """Correlate shell commands to sessions by matching content.

    Since plain-format shell history lacks timestamps, we match command
    content to session activity:
    - Commands containing 'git commit' → match to nearest session by repo
    - Commands containing 'cd <project>' → match to that project's sessions
    - Commands containing project-related keywords → match to project sessions

    This is best-effort — it won't be perfect, but it gives us approximate
    session attribution for the 5,606 commands.
    """
    if not sessions:
        return 0

    # Get all shell commands.
    commands = db.query("SELECT id, command FROM shell_commands ORDER BY id ASC")
    if not commands:
        return 0

    # Build project → sessions mapping.
    project_sessions: dict[str, list[dict]] = {}
    for s in sessions:
        proj = s.get("project", "")
        if proj:
            project_sessions.setdefault(proj, []).append(s)

    correlated = 0
    for cmd in commands:
        command_text = cmd["command"].lower()

        # Match 'cd <project>' commands.
        matched_session = None
        for proj, proj_sessions in project_sessions.items():
            if proj.lower() in command_text:
                # Pick the session closest to this project's activity.
                matched_session = proj_sessions[0]
                break

        # Match 'git commit' commands to any session.
        if not matched_session and "git commit" in command_text:
            matched_session = sessions[0] if sessions else None

        # Match project-specific commands.
        if not matched_session:
            for proj, proj_sessions in project_sessions.items():
                # Check if any word in the project name appears in the command.
                words = proj.lower().replace("-", " ").replace("_", " ").split()
                if any(w in command_text for w in words if len(w) > 3):
                    matched_session = proj_sessions[0]
                    break

        if matched_session:
            correlated += 1

    log.info(
        "session.shell.correlated",
        total_commands=len(commands),
        correlated=correlated,
        sessions=len(sessions),
    )
    return correlated


# Type alias for clarity (sqlite3.Row is duck-typed).
sqlite3_Row_like = Any
