"""Context quality evaluator — measures how much useful context MAOS has.

Evaluates six dimensions:
  1. Workspace awareness — are workspaces registered with metadata?
  2. Session continuity — are sessions detected and summarized?
  3. Task awareness — are there pending tasks with metadata?
  4. Knowledge retrieval — are there indexed files and entities?
  5. Preference awareness — does the user model have entries?
  6. Activity timeline coverage — are events recorded across types?

Each dimension returns a 0-100 score. The overall score is a weighted
average. Results are stored in the context_evaluations table for
tracking over time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)

# Weights for the overall score (must sum to 1.0).
_WEIGHTS = {
    "workspace": 0.20,
    "session": 0.15,
    "tasks": 0.15,
    "knowledge": 0.25,
    "preferences": 0.10,
    "activity": 0.15,
}


def evaluate_context(db: Database) -> dict[str, Any]:
    """Evaluate context quality across all dimensions.

    Returns a dict with per-dimension scores (0-100), an overall score,
    and details explaining each score.
    """
    workspace_score, workspace_details = _eval_workspace(db)
    session_score, session_details = _eval_session(db)
    tasks_score, tasks_details = _eval_tasks(db)
    knowledge_score, knowledge_details = _eval_knowledge(db)
    preferences_score, preferences_details = _eval_preferences(db)
    activity_score, activity_details = _eval_activity(db)

    overall = round(
        workspace_score * _WEIGHTS["workspace"]
        + session_score * _WEIGHTS["session"]
        + tasks_score * _WEIGHTS["tasks"]
        + knowledge_score * _WEIGHTS["knowledge"]
        + preferences_score * _WEIGHTS["preferences"]
        + activity_score * _WEIGHTS["activity"]
    )

    result = {
        "workspace": workspace_score,
        "session": session_score,
        "tasks": tasks_score,
        "knowledge": knowledge_score,
        "preferences": preferences_score,
        "activity": activity_score,
        "overall": overall,
        "details": {
            "workspace": workspace_details,
            "session": session_details,
            "tasks": tasks_details,
            "knowledge": knowledge_details,
            "preferences": preferences_details,
            "activity": activity_details,
        },
    }

    # Persist to database.
    db.save_context_evaluation(
        workspace=workspace_score,
        session=session_score,
        tasks=tasks_score,
        knowledge=knowledge_score,
        preferences=preferences_score,
        activity=activity_score,
        overall=overall,
        details=json.dumps(result["details"]),
    )

    log.info(
        "context.evaluated",
        overall=overall,
        workspace=workspace_score,
        session=session_score,
        tasks=tasks_score,
        knowledge=knowledge_score,
        preferences=preferences_score,
        activity=activity_score,
    )

    return result


def _eval_workspace(db: Database) -> tuple[int, dict[str, Any]]:
    """Workspace awareness: registered workspaces with metadata + profiles."""
    workspaces = db.all_workspaces()
    count = len(workspaces)

    if count == 0:
        return 0, {"count": 0, "reason": "no workspaces registered"}

    # Score based on count and metadata completeness.
    with_language = sum(1 for w in workspaces if w["language"])
    with_git = sum(1 for w in workspaces if w["git_repo"])
    with_activity = sum(1 for w in workspaces if w["last_activity"])
    with_profile = sum(1 for w in workspaces if w["profile"])

    completeness = (
        (with_language / count * 20)
        + (with_git / count * 20)
        + (with_activity / count * 15)
        + (with_profile / count * 25)  # profiles are the biggest contributor
    )
    count_score = min(count * 20, 20)  # 1 workspace = 20, 5+ = 100
    score = round(count_score + completeness)

    return score, {
        "count": count,
        "with_language": with_language,
        "with_git": with_git,
        "with_activity": with_activity,
        "with_profile": with_profile,
    }


def _eval_session(db: Database) -> tuple[int, dict[str, Any]]:
    """Session continuity: detected sessions with summaries."""
    total = db.session_count()
    recent = db.recent_sessions(limit=10)

    if total == 0:
        return 0, {"count": 0, "reason": "no sessions detected"}

    with_summary = sum(1 for s in recent if s["summary"])
    with_project = sum(1 for s in recent if s["project"])

    count_score = min(total * 10, 50)  # 5+ sessions = 50
    summary_score = (with_summary / len(recent) * 25) if recent else 0
    project_score = (with_project / len(recent) * 25) if recent else 0
    score = round(count_score + summary_score + project_score)

    return score, {
        "count": total,
        "recent": len(recent),
        "with_summary": with_summary,
        "with_project": with_project,
    }


def _eval_tasks(db: Database) -> tuple[int, dict[str, Any]]:
    """Task awareness: pending tasks with metadata."""
    pending = db.pending_tasks()
    all_tasks = db.all_tasks(limit=100)

    if not all_tasks:
        return 0, {"count": 0, "reason": "no tasks created"}

    with_workspace = sum(1 for t in all_tasks if t["related_workspace"])
    with_priority = sum(1 for t in all_tasks if t["priority"] > 0)
    ai_sourced = sum(1 for t in all_tasks if t["source"] == "ai")

    count_score = min(len(all_tasks) * 15, 40)
    metadata_score = (with_workspace / len(all_tasks) * 20) if all_tasks else 0
    priority_score = (with_priority / len(all_tasks) * 20) if all_tasks else 0
    ai_score = (ai_sourced / len(all_tasks) * 20) if all_tasks else 0
    score = round(count_score + metadata_score + priority_score + ai_score)

    return score, {
        "total": len(all_tasks),
        "pending": len(pending),
        "with_workspace": with_workspace,
        "with_priority": with_priority,
        "ai_sourced": ai_sourced,
    }


def _eval_knowledge(db: Database) -> tuple[int, dict[str, Any]]:
    """Knowledge retrieval: indexed files and knowledge graph entities."""
    file_count = db.file_hash_count()
    entity_row = db.query_one("SELECT COUNT(*) as n FROM entities")
    rel_row = db.query_one("SELECT COUNT(*) as n FROM relationships")
    commit_count = db.commit_count()

    entities = entity_row["n"] if entity_row else 0
    relationships = rel_row["n"] if rel_row else 0

    file_score = min(file_count * 2, 40)  # 20+ files = 40
    entity_score = min(entities * 5, 30)  # 6+ entities = 30
    rel_score = min(relationships * 5, 15)  # 3+ relationships = 15
    commit_score = min(commit_count * 2, 15)  # 8+ commits = 15
    score = round(file_score + entity_score + rel_score + commit_score)

    return score, {
        "indexed_files": file_count,
        "entities": entities,
        "relationships": relationships,
        "git_commits": commit_count,
    }


def _eval_preferences(db: Database) -> tuple[int, dict[str, Any]]:
    """Preference awareness: user model entries."""
    entries = db.all_user_model_entries()
    count = len(entries)

    if count == 0:
        return 0, {"count": 0, "reason": "no preferences recorded"}

    # Exclude system entries (like _blackboard_state).
    user_entries = [e for e in entries if not e["key"].startswith("_")]
    user_count = len(user_entries)

    with_high_confidence = sum(1 for e in user_entries if e["confidence"] > 0.7)

    count_score = min(user_count * 20, 60)
    confidence_score = (with_high_confidence / max(user_count, 1) * 40)
    score = round(count_score + confidence_score)

    return score, {
        "total": count,
        "user_entries": user_count,
        "with_high_confidence": with_high_confidence,
    }


def _eval_activity(db: Database) -> tuple[int, dict[str, Any]]:
    """Activity timeline coverage: events across multiple types over time."""
    since = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    recent = db.query_events(since=since, limit=500)

    if not recent:
        return 0, {"count": 0, "reason": "no events in the last 7 days"}

    types = {e["type"] for e in recent}
    type_count = len(types)

    # Score based on event volume and type diversity.
    volume_score = min(len(recent) * 2, 50)  # 25+ events = 50
    diversity_score = min(type_count * 10, 50)  # 5+ types = 50
    score = round(volume_score + diversity_score)

    return score, {
        "recent_events": len(recent),
        "event_types": type_count,
        "types": sorted(types),
    }
