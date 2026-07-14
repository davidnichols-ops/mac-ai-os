"""Context packet builder — assembles everything an AI client needs.

This is the most important endpoint. Every AI client wants the same thing:
"Give me everything relevant right now." Instead of making clients query
five different endpoints, the context packet combines workspace, session,
tasks, decisions, knowledge, telemetry, and blackboard state into one
response.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.blackboard import Blackboard
from app.db import Database
from app.log import get_logger
from app.retrieval.base import Retriever
from app.workspace import detect_active_workspace

log = get_logger(__name__)


def build_context_packet(
    db: Database,
    blackboard: Blackboard,
    *,
    retriever: Retriever | None = None,
    lookback_hours: int = 24,
    rag_query: str | None = None,
) -> dict[str, Any]:
    """Assemble a universal context packet for AI clients.

    Args:
        db: MAOS database.
        blackboard: Shared working memory.
        retriever: Optional vector store for RAG results.
        lookback_hours: How far back to look for events, telemetry, etc.
        rag_query: Optional query to include RAG results. If None, uses
            the blackboard's current goal as the query.

    Returns:
        A dict with workspace, session, tasks, decisions, knowledge,
        system, and blackboard state.
    """
    now = datetime.now(UTC)
    since = (now - timedelta(hours=lookback_hours)).isoformat()

    # --- Workspace ---
    workspace = _workspace_info(db)

    # --- Workspace profiles (all understood projects) ---
    workspace_profiles = _workspace_profiles(db)

    # --- Session ---
    session = _session_info(db)

    # --- Tasks ---
    tasks = _tasks_info(db)

    # --- Recent decisions ---
    decisions = _decisions_info(db, since)

    # --- Related knowledge (graph entities) ---
    knowledge = _knowledge_info(db, blackboard, rag_query)

    # --- RAG results ---
    rag_results: list[dict[str, Any]] = []
    query = rag_query or blackboard.current_goal or ""
    if retriever and query:
        try:
            results = retriever.search(query, top_k=5)
            rag_results = [
                {
                    "text": r.text[:500],
                    "source": r.source,
                    "score": round(r.score, 3),
                }
                for r in results
            ]
        except Exception as e:
            log.warning("context_packet.rag_failed", error=str(e))

    # --- System health ---
    system = _system_info(db, since)

    # --- Blackboard ---
    bb = blackboard.snapshot()

    # --- Recent events summary ---
    events_summary = _events_summary(db, since)

    return {
        "timestamp": now.isoformat(),
        "workspace": workspace,
        "workspace_profiles": workspace_profiles,
        "session": session,
        "tasks": tasks,
        "recent_decisions": decisions,
        "related_knowledge": knowledge,
        "rag_results": rag_results,
        "system": system,
        "blackboard": {
            "current_goal": bb["current_goal"],
            "current_workspace": bb["current_workspace"],
            "open_problems": bb["open_problems"],
            "current_plan": bb["current_plan"],
            "suggested_actions": bb["suggested_actions"],
            "confidence": bb["confidence"],
            "last_updated": bb["last_updated"],
        },
        "recent_activity": events_summary,
    }


def _workspace_info(db: Database) -> dict[str, Any]:
    """Active workspace with git status."""
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
        log.warning("context_packet.workspace_failed", error=str(e))
        return {"name": None, "path": None, "error": str(e)}


def _workspace_profiles(db: Database) -> list[dict[str, Any]]:
    """Profiles of all understood workspaces.

    Returns a list of {name, path, status, profile} for every workspace
    that has been profiled. Sorted by status (active first), then by
    last_activity.
    """
    try:
        rows = db.query(
            "SELECT name, path, project_status, profile, language, last_activity "
            "FROM workspaces WHERE profile != '' "
            "ORDER BY CASE project_status "
            "  WHEN 'active' THEN 0 WHEN 'maintenance' THEN 1 "
            "  WHEN 'complete' THEN 2 WHEN 'abandoned' THEN 3 ELSE 4 END, "
            "last_activity DESC"
        )
        return [
            {
                "name": r["name"],
                "path": r["path"],
                "status": r["project_status"],
                "language": r["language"] or "",
                "profile": r["profile"][:500],  # truncate for context budget
            }
            for r in rows
        ]
    except Exception as e:
        log.warning("context_packet.profiles_failed", error=str(e))
        return []


def _session_info(db: Database) -> dict[str, Any]:
    """Latest work session."""
    try:
        latest = db.latest_session()
        if not latest:
            return {"active": False}
        s = dict(latest)
        return {
            "active": True,
            "id": s.get("id"),
            "start_time": s.get("start_time", ""),
            "end_time": s.get("end_time", ""),
            "project": s.get("project", ""),
            "workspace": s.get("workspace", ""),
            "event_count": s.get("event_count", 0),
            "summary": s.get("summary", ""),
        }
    except Exception as e:
        log.warning("context_packet.session_failed", error=str(e))
        return {"active": False, "error": str(e)}


def _tasks_info(db: Database) -> list[dict[str, Any]]:
    """Pending tasks."""
    try:
        rows = db.pending_tasks()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "priority": r["priority"],
                "related_workspace": r["related_workspace"] or "",
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception as e:
        log.warning("context_packet.tasks_failed", error=str(e))
        return []


def _decisions_info(db: Database, since: str) -> list[dict[str, Any]]:
    """Recent routing/action decisions."""
    try:
        rows = db.query(
            "SELECT * FROM decisions WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 10",
            (since,),
        )
        return [
            {
                "agent": r["agent"],
                "decision": r["decision"],
                "confidence": r["confidence"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    except Exception:
        return []


def _knowledge_info(
    db: Database, blackboard: Blackboard, query: str | None,
) -> list[dict[str, Any]]:
    """Entities from the knowledge graph related to current context."""
    try:
        # Search by goal keywords or the provided query.
        search_term = query or blackboard.current_goal or ""
        if not search_term:
            # Fall back to recent entities.
            rows = db.query(
                "SELECT name, type, description FROM entities ORDER BY created_at DESC LIMIT 10"
            )
        else:
            words = [w for w in search_term.split() if len(w) > 2]
            entities = []
            seen = set()
            for word in words[:3]:
                for e in db.search_entities(word, limit=5):
                    name = e["name"] if isinstance(e, dict) else e["name"]
                    if name not in seen:
                        seen.add(name)
                        entities.append(e)
            rows = entities[:10]

        return [
            {
                "name": r["name"] if hasattr(r, "__getitem__") else r.get("name", ""),
                "type": r["type"] if hasattr(r, "__getitem__") else r.get("type", ""),
                "description": r["description"] if hasattr(r, "__getitem__") else r.get("description", ""),
            }
            for r in rows
        ]
    except Exception as e:
        log.warning("context_packet.knowledge_failed", error=str(e))
        return []


def _system_info(db: Database, since: str) -> dict[str, Any]:
    """Latest telemetry snapshot."""
    try:
        rows = db.telemetry_since(since, limit=1)
        if not rows:
            return {"available": False}
        t = dict(rows[-1])
        mem_percent = t.get("memory_percent", 0)
        return {
            "available": True,
            "cpu_percent": round(t.get("cpu_percent", 0), 1),
            "memory_used_gb": round(t.get("memory_used_gb", 0), 1),
            "memory_total_gb": round(t.get("memory_total_gb", 0), 1),
            "memory_pressure": "high" if mem_percent > 85 else "normal",
            "process_count": t.get("process_count", 0),
            "thermal_state": t.get("thermal_state"),
            "timestamp": t.get("timestamp", ""),
        }
    except Exception:
        return {"available": False}


def _events_summary(db: Database, since: str) -> dict[str, Any]:
    """Summary of recent events by type."""
    try:
        rows = db.query(
            "SELECT type, COUNT(*) as count FROM events WHERE timestamp >= ? GROUP BY type ORDER BY count DESC LIMIT 10",
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
