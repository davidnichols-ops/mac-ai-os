"""MCP resources — thin adapters over the existing Context API.

Each resource handler delegates to app/server/context.py or direct
database queries. No business logic here — just protocol translation.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from app.db import Database
from app.log import get_logger
from app.runtime import Runtime
from app.server.context import build_context_packet
from app.workspace import detect_active_workspace

log = get_logger(__name__)


def read_context_resource(runtime: Runtime) -> str:
    """maos://context/current — the universal context packet.

    Delegates to build_context_packet() in the Context API.
    """
    packet = build_context_packet(
        runtime.db,
        runtime.blackboard,
        retriever=runtime.retriever,
    )
    return json.dumps(packet, default=str, indent=2)


def read_workspace_resource(runtime: Runtime) -> str:
    """maos://workspace/current — active workspace with git status."""
    ws = detect_active_workspace(runtime.db)
    if not ws:
        return json.dumps({"error": "no active workspace"})

    result: dict[str, Any] = {
        "name": ws.get("name", ""),
        "path": ws.get("path", ""),
        "language": ws.get("language", ""),
        "git_repo": bool(ws.get("git_repo", False)),
    }

    # Enrich with live git status (read-only, no side effects).
    path = ws.get("path", "")
    if path:
        result.update(_git_status(path))

    return json.dumps(result, default=str, indent=2)


def read_session_resource(runtime: Runtime) -> str:
    """maos://session/latest — most recent session with details."""
    db = runtime.db
    latest = db.latest_session()
    if not latest:
        return json.dumps({"error": "no sessions recorded"})

    session = dict(latest)

    # Enrich with related data from the same time window.
    start = session.get("start_time", "")
    end = session.get("end_time", "")

    # Commands run during this session.
    if start and end:
        commands = db.query(
            "SELECT * FROM shell_commands WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC LIMIT 50",
            (start, end),
        )
        session["commands"] = [dict(c) for c in commands]

        # Commits during this session.
        commits = db.query(
            "SELECT * FROM git_commits WHERE author_date >= ? AND author_date <= ? "
            "ORDER BY author_date DESC LIMIT 20",
            (start, end),
        )
        session["commits"] = [dict(c) for c in commits]

        # Events during this session.
        events = db.query(
            "SELECT * FROM events WHERE timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp ASC LIMIT 100",
            (start, end),
        )
        session["events"] = [dict(e) for e in events]
    else:
        session["commands"] = []
        session["commits"] = []
        session["events"] = []

    # Pending tasks (potential "unfinished work").
    tasks = db.pending_tasks()
    session["unfinished_tasks"] = [dict(t) for t in tasks]

    return json.dumps(session, default=str, indent=2)


def read_tasks_resource(runtime: Runtime) -> str:
    """maos://tasks/open — pending tasks with dependencies and priority."""
    db = runtime.db
    rows = db.pending_tasks()
    tasks = []
    for r in rows:
        task = dict(r)
        # Check if task is blocked by dependencies.
        deps_str = task.get("dependencies", "")
        deps = [d.strip() for d in deps_str.split(",") if d.strip()] if deps_str else []
        task["dependencies"] = deps
        task["is_blocked"] = _is_task_blocked(db, task["id"], deps)
        tasks.append(task)
    return json.dumps(
        {"count": len(tasks), "tasks": tasks}, default=str, indent=2
    )


def read_knowledge_resource(runtime: Runtime, query: str = "") -> str:
    """maos://knowledge/search — semantic search over indexed knowledge.

    Combines vector store search with knowledge graph entity lookup.
    """
    results: dict[str, Any] = {"query": query}

    # Vector search.
    if query and runtime.retriever:
        try:
            rag = runtime.retriever.search(query, top_k=5)
            results["vector_results"] = [
                {
                    "text": r.text[:500],
                    "source": r.source,
                    "score": round(r.score, 3),
                }
                for r in rag
            ]
        except Exception as e:
            results["vector_error"] = str(e)
    else:
        results["vector_results"] = []

    # Knowledge graph entity search.
    if query:
        words = [w for w in query.split() if len(w) > 2]
        entities = []
        seen = set()
        for word in words[:3]:
            for e in runtime.db.search_entities(word, limit=5):
                name = e["name"]
                if name not in seen:
                    seen.add(name)
                    entities.append(dict(e))
        results["graph_entities"] = entities[:10]
    else:
        results["graph_entities"] = []

    return json.dumps(results, default=str, indent=2)


# --- helpers ---------------------------------------------------------------


def _git_status(path: str) -> dict[str, Any]:
    """Get git status for a workspace path (read-only)."""
    info: dict[str, Any] = {}
    try:
        branch = subprocess.run(
            ["git", "-C", path, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.stdout.strip():
            info["git_branch"] = branch.stdout.strip()

        status = subprocess.run(
            ["git", "-C", path, "status", "--short"],
            capture_output=True, text=True, timeout=5,
        )
        if status.stdout.strip():
            modified = [
                line.strip()
                for line in status.stdout.strip().split("\n")
                if line.strip()
            ]
            info["modified_files"] = modified
            info["modified_file_count"] = len(modified)
        else:
            info["modified_files"] = []
            info["modified_file_count"] = 0

        info["git_status"] = "clean" if not status.stdout.strip() else "modified"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        info["git_status"] = "unavailable"
    return info


def _is_task_blocked(db: Database, task_id: int, deps: list[str]) -> bool:
    """Check if a task is blocked by incomplete dependencies."""
    if not deps:
        return False
    for dep_str in deps:
        try:
            dep_id = int(dep_str)
            row = db.query_one(
                "SELECT status FROM tasks WHERE id = ?", (dep_id,)
            )
            if row and row["status"] != "completed":
                return True
        except (ValueError, TypeError):
            continue
    return False
