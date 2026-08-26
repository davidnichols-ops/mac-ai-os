"""Full-text search over MAOS sessions and events using SQLite FTS5.

Adapted from the FTS5 session-search pattern in NousResearch's hermes-agent
(``hermes_state_search.py``). MAOS stores session summaries and a flat event
log but has no way to *recall* across them — the root cause of the weak
"session continuity" context score. This module adds a bounded FTS5 index over
session summaries and event payloads so past work is searchable and can be
carried forward into the next refresh and the Daily Brief.

Design notes:

- **External-content FTS5** is avoided for simplicity; we maintain a plain
  FTS5 table and rebuild/merge incrementally. The index is a cache, not a
  source of truth — the ``sessions`` and ``events`` tables remain authoritative.
- **Bounded queries.** User input is sanitized before use in ``MATCH`` so FTS5
  query syntax cannot error or be abused.
- **Provenance preserved.** Every hit returns the session id, project, and
  timestamp so results can be traced back to the source record.
- **Pure stdlib.** No new dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)

MAX_QUERY_CHARS = 200
MAX_RESULTS = 20

# FTS5 special characters that must be neutralized in user input.
_FTS5_SPECIAL = re.compile(r'["()*:^]')


def _sanitize_query(query: str) -> str:
    """Sanitize user input for safe use in an FTS5 MATCH query.

    FTS5 has its own query syntax (``"``, ``(``, ``)``, ``*``, ``:``, ``^``).
    We strip unmatched special characters and wrap hyphenated/dotted terms in
    quotes so a bare user query cannot error or inject operators.
    """
    query = (query or "").strip()[:MAX_QUERY_CHARS]
    if not query:
        return ""
    # Wrap hyphenated/dotted terms so FTS5 treats them as literals.
    query = re.sub(r"([A-Za-z0-9]+[-.][A-Za-z0-9.-]+)", r'"\1"', query)
    # Strip remaining FTS5-special characters.
    query = _FTS5_SPECIAL.sub(" ", query)
    return " ".join(query.split())


class SessionSearch:
    """FTS5-backed full-text search over MAOS sessions and events."""

    def __init__(self, db: Database):
        self.db = db
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
                session_id UNINDEXED,
                project,
                summary,
                event_text,
                start_time UNINDEXED
            )"""
        )

    def rebuild(self) -> int:
        """Rebuild the FTS index from the authoritative sessions/events tables.

        Returns the number of rows indexed. Safe to call on every refresh; the
        index is a cache.
        """
        self.db.execute("DELETE FROM session_fts")
        sessions = self.db.query("SELECT * FROM sessions ORDER BY start_time ASC")
        count = 0
        for s in sessions:
            events = self.db.query(
                "SELECT payload FROM events WHERE timestamp >= ? AND timestamp <= ?",
                (s["start_time"], s["end_time"]),
            )
            event_text = " ".join(
                _payload_text(e["payload"]) for e in events if e["payload"]
            )
            self.db.execute(
                """INSERT INTO session_fts (session_id, project, summary, event_text, start_time)
                   VALUES (?, ?, ?, ?, ?)""",
                (s["id"], s["project"] or "", s["summary"] or "", event_text, s["start_time"]),
            )
            count += 1
        log.info("session_search.rebuilt", rows=count)
        return count

    def search(self, query: str, *, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
        """Search sessions by full-text relevance.

        Returns a list of dicts with session_id, project, summary, start_time,
        and a snippet. Empty/unsanitized queries return [].
        """
        sanitized = _sanitize_query(query)
        if not sanitized:
            return []
        try:
            rows = self.db.query(
                """SELECT session_id, project, summary, start_time,
                          snippet(session_fts, 2, '[', ']', '…', 12) AS snip
                   FROM session_fts
                   WHERE session_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (sanitized, limit),
            )
        except Exception as exc:  # FTS5 syntax edge cases
            log.warning("session_search.match_failed", error=str(exc))
            return []
        results = []
        for r in rows:
            results.append({
                "session_id": r["session_id"],
                "project": r["project"],
                "summary": r["summary"],
                "start_time": r["start_time"],
                "snippet": r["snip"] or (r["summary"] or "")[:120],
            })
        return results

    def continuity_signal(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return the most relevant past sessions for a query — the signal that
        lets a new session carry forward prior context (session continuity)."""
        return self.search(query, limit=limit)


def _payload_text(payload: str) -> str:
    """Flatten an event payload JSON blob into searchable text."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload or ""
    if isinstance(data, dict):
        parts = []
        for k, v in data.items():
            if isinstance(v, (str, int, float)):
                parts.append(f"{k}: {v}")
            elif isinstance(v, (list, dict)):
                parts.append(json.dumps(v, ensure_ascii=False))
        return " ".join(parts)
    return str(data)
