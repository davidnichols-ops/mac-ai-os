"""Daily brief generator — Nemotron synthesizes MAOS tasks into clear assignments.

The core problem: 49 pending tasks across 28 workspaces is paralyzing.
The solution: Nemotron Ultra reads all tasks, groups them by business unit,
and produces 3-5 clear, actionable assignments for the day.

Each assignment is:
  - One sentence describing what to do
  - The expected deliverable
  - The workspace/repo to work in
  - Estimated effort (S/M/L)
  - Why it matters now

No fluff. No paralysis. Just clear work.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from app.db import Database
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message
from app.retrieval.base import Retriever

log = get_logger(__name__)

# Business unit mapping — workspace → department.
BUSINESS_UNITS: dict[str, str] = {
    "X-MaC": "Product Engineering",
    "X-MaC-digital-twin": "Product Engineering",
    "aafp": "R&D / Networking",
    "aafp-go": "R&D / Networking",
    "AAFP-research": "R&D / Networking",
    "mac-ai-os": "Platform / Trajectory OS",
    "repo-archaeologist": "Internal Tools",
    "linkedin-dev-pipeline": "Growth",
    "elonwatch": "Ops / Monitoring",
    "elonwatch-mac": "Ops / Monitoring",
    "trustcard": "Security",
    "jarword": "Client Work",
    "cvconform": "R&D / CV",
    "apple-quality-recognition-engine": "R&D / CV",
    "apple_quality_recognition_engine": "R&D / CV",
    "Gargantua": "R&D / Graphics",
    "word_proc": "Product",
    "dnichols-ops-": "Core Infrastructure",
}

# Tasks with these prefixes are GitHub PRs/issues — separate bucket.
GITHUB_PREFIX = "GitHub:"


def _task_get(task: Any, key: str, default: str = "") -> Any:
    """Get a field from a task that might be a dict or sqlite3.Row."""
    try:
        val = task[key]
        return val if val is not None else default
    except (KeyError, IndexError):
        return default


def _classify_task(task: Any) -> str:
    """Classify a task into a business unit."""
    title = _task_get(task, "title", "")
    ws = _task_get(task, "related_workspace", "") or _task_get(task, "workspace", "")

    if title.startswith(GITHUB_PREFIX):
        return "GitHub Backlog"

    if ws in BUSINESS_UNITS:
        return BUSINESS_UNITS[ws]

    # Infer from title keywords.
    title_lower = title.lower()
    if any(w in title_lower for w in ["x-mac", "xmac", "dedup", "disk", "cleaner"]):
        return "Product Engineering"
    if any(w in title_lower for w in ["aafp", "post-quantum", "quic", "p2p"]):
        return "R&D / Networking"
    if any(w in title_lower for w in ["nemotron", "trajectory", "maos", "operator"]):
        return "Platform / Trajectory OS"
    if any(w in title_lower for w in ["slack", "bridge", "notifier"]):
        return "Platform / Trajectory OS"
    if any(w in title_lower for w in ["ultralytics", "yolo", "detection", "tracking"]):
        return "External Contributions"

    return "Unassigned"


def _group_tasks(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group tasks by business unit."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        unit = _classify_task(t)
        groups.setdefault(unit, []).append(t)
    # Sort each group by priority (descending).
    for unit in groups:
        groups[unit].sort(key=lambda t: _task_get(t, "priority", 0), reverse=True)
    return groups


def _format_tasks_for_model(tasks: list[dict[str, Any]]) -> str:
    """Format tasks as a compact list for the model prompt."""
    lines = []
    for t in tasks:
        tid = _task_get(t, "id", "?")
        title = _task_get(t, "title", "?")[:120]
        priority = _task_get(t, "priority", 0)
        ws = _task_get(t, "related_workspace", "") or _task_get(t, "workspace", "")
        lines.append(f"  [{tid}] (p={priority}) {title}" + (f"  ws={ws}" if ws else ""))
    return "\n".join(lines)


def generate_brief(
    db: Database,
    *,
    model: Model | None = None,
    retriever: Retriever | None = None,
    max_assignments: int = 5,
) -> dict[str, Any]:
    """Generate a daily brief — clear, non-paralyzing work assignments.

    Returns:
        {"ok": bool, "text": str, "assignments": [...], "stats": {...}}
    """
    tasks = db.pending_tasks()
    if not tasks:
        return {"ok": True, "text": "No pending tasks. Clear day.", "assignments": [], "stats": {"total": 0}}

    groups = _group_tasks(tasks)
    total = len(tasks)
    github_count = len(groups.get("GitHub Backlog", []))

    # Stats by business unit.
    stats = {
        "total": total,
        "github": github_count,
        "by_unit": {u: len(ts) for u, ts in groups.items()},
    }

    # If no model, produce a structured fallback (still useful, just not AI-synthesized).
    if model is None:
        return {
            "ok": True,
            "text": _fallback_brief(groups, total, max_assignments),
            "assignments": [],
            "stats": stats,
        }

    # Build the prompt for Nemotron Ultra.
    today = datetime.now(UTC).strftime("%A, %B %d, %Y")
    task_list = _format_tasks_for_model(tasks)

    # Pull recent context from RAG if available.
    rag_context = ""
    if retriever:
        try:
            results = retriever.search("current work priorities and active projects", top_k=3)
            if results:
                rag_context = "\n\nRecent context from indexed work:\n"
                for r in results:
                    rag_context += f"  [{r.source}] {r.text[:300]}\n"
        except Exception as e:
            log.warning("brief.rag_failed", error=str(e))

    system = f"""You are the chief of staff for a solo systems engineer running a portfolio of projects.
Today is {today}.

The engineer has {total} pending tasks across these business units:
{json.dumps(stats['by_unit'], indent=2)}

Here are ALL pending tasks (ID, priority, title, workspace):
{task_list}
{rag_context}

Your job: produce a DAILY BRIEF with exactly {max_assignments} assignments.

Rules:
1. Each assignment is ONE clear action — no vague "work on X" or "continue Y".
2. Pick the highest-impact tasks. Prioritize by: unblocking other work > user-facing impact > technical debt.
3. Group related tasks (e.g. multiple GitHub PRs for the same repo = one assignment).
4. Each assignment has: a one-line action, the deliverable, where to work, effort estimate (S/M/L), and why now.
5. Be direct and concise. No motivational language. No "consider" or "you might want to". Just clear work.
6. If a task is stale (>2 weeks old), call it out as "stale — decide: do or drop".
7. End with a one-line "Focus today:" that names the single most important thing.

Format each assignment as:
**N. [Action]**
   Deliverable: [what done looks like]
   Where: [workspace/repo]
   Effort: [S/M/L]
   Why now: [one sentence]

This is for a Slack message. Keep it tight. No preamble. No summary paragraph at the top. Just the assignments and the focus line."""

    try:
        resp = model.chat([
            Message("system", system),
            Message("user", f"Generate today's brief. {total} tasks, {max_assignments} assignments max."),
        ])
        text = resp.text.strip() if resp.text else _fallback_brief(groups, total, max_assignments)
        return {"ok": True, "text": text, "assignments": [], "stats": stats}
    except Exception as e:
        log.error("brief.model_failed", error=str(e))
        return {
            "ok": False,
            "text": f"Brief generation failed: {e}\n\n" + _fallback_brief(groups, total, max_assignments),
            "assignments": [],
            "stats": stats,
        }


def _fallback_brief(groups: dict[str, list[dict[str, Any]]], total: int, max_assignments: int) -> str:
    """Produce a structured brief without AI — just priority-sorted tasks."""
    lines = [f"*Daily Brief — {total} tasks across {len(groups)} units*\n"]

    # Flatten and sort by priority.
    all_tasks = []
    for unit, tasks in groups.items():
        for t in tasks:
            all_tasks.append((unit, t))
    all_tasks.sort(key=lambda x: _task_get(x[1], "priority", 0), reverse=True)

    # Pick top N.
    for i, (unit, t) in enumerate(all_tasks[:max_assignments], 1):
        tid = _task_get(t, "id", "?")
        title = _task_get(t, "title", "?")[:80]
        lines.append(f"*{i}. [{unit}] {title}*")
        lines.append(f"   Task ID: {tid}")
        lines.append("")

    if len(all_tasks) > max_assignments:
        lines.append(f"_+{len(all_tasks) - max_assignments} more tasks. Send 'tasks' for full list._")

    return "\n".join(lines)


def generate_project_brief(
    db: Database,
    workspace: str,
    *,
    model: Model | None = None,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    """Dynamically load and summarize a specific project's state.

    Pulls: tasks, workspace profile, recent git commits, RAG context.
    Synthesizes with Nemotron into a project status + next action.
    """
    # Get tasks for this workspace.
    all_tasks = db.pending_tasks()
    ws_tasks = [t for t in all_tasks if (_task_get(t, "related_workspace", "") or _task_get(t, "workspace", "")) == workspace]

    # Get workspace profile from DB.
    profile = ""
    try:
        rows = db.query(
            "SELECT profile FROM workspaces WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
            (workspace,),
        )
        if rows:
            profile = _task_get(rows[0], "profile", "")
    except Exception:
        pass

    # Get recent git commits for this workspace.
    commits = []
    try:
        rows = db.query(
            "SELECT hash, message, author, date FROM git_commits "
            "WHERE repo LIKE ? ORDER BY date DESC LIMIT 5",
            (f"%{workspace}%",),
        )
        commits = rows or []
    except Exception:
        pass

    # RAG search for project context.
    rag_context = ""
    if retriever:
        try:
            results = retriever.search(f"{workspace} project status recent work", top_k=5)
            if results:
                chunks = [f"[{r.source}] {r.text[:500]}" for r in results]
                rag_context = "\n\nIndexed context:\n" + "\n\n".join(chunks)
        except Exception:
            pass

    # Build summary data.
    task_list = _format_tasks_for_model(ws_tasks) if ws_tasks else "  (no pending tasks)"
    commit_list = "\n".join(
        f"  {_task_get(c, 'date', '?')[:10]} {_task_get(c, 'message', '?')[:80]}"
        for c in commits
    ) if commits else "  (no recent commits indexed)"

    if model is None:
        # Fallback: structured dump.
        lines = [f"*Project: {workspace}*\n"]
        if profile:
            lines.append(f"Profile: {profile[:200]}\n")
        lines.append(f"Tasks ({len(ws_tasks)}):")
        lines.append(task_list)
        lines.append("\nRecent commits:")
        lines.append(commit_list)
        return {"ok": True, "text": "\n".join(lines)}

    system = f"""You are a project analyst. Summarize the current state of the '{workspace}' project and recommend the single next action.

Project profile:
{profile[:1000]}

Pending tasks:
{task_list}

Recent commits:
{commit_list}
{rag_context}

Produce a concise project brief:
1. **Status** — one sentence: is this active, stalled, complete, or needs decision?
2. **Open work** — list the pending tasks (if any) with priority
3. **Next action** — the ONE thing to do next, with clear deliverable
4. **Blockers** — anything blocking progress (or 'none')

Keep it under 15 lines. This is for Slack. No fluff."""

    try:
        resp = model.chat([
            Message("system", system),
            Message("user", f"Summarize project: {workspace}"),
        ])
        text = resp.text.strip() if resp.text else f"Project {workspace}: no response from model"
        return {"ok": True, "text": text}
    except Exception as e:
        log.error("project_brief.model_failed", error=str(e))
        return {"ok": False, "text": f"Project brief failed: {e}"}


def generate_trajectory_brief(
    db: Database,
    *,
    model: Model | None = None,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    """Dynamically load Trajectory OS / MAOS state and synthesize next steps.

    Pulls: Trajectory OS findings plus MAOS tasks, blackboard goal, session
    history, and RAG context. Synthesizes with Nemotron into a trajectory
    status + next milestone.
    """
    # Get MAOS-specific tasks.
    all_tasks = db.pending_tasks()
    maos_tasks = [
        t for t in all_tasks
        if any(w in (_task_get(t, "title", "") + _task_get(t, "related_workspace", "")).lower()
               for w in ["maos", "trajectory", "operator", "slack", "nemotron", "guard", "blackboard"])
    ] if all_tasks else []

    # Get blackboard state.
    blackboard_info = ""
    try:
        rows = db.query(
            "SELECT key, value FROM blackboard ORDER BY updated_at DESC LIMIT 10"
        )
        if rows:
            blackboard_info = "\n".join(f"  {r['key']}: {str(r['value'])[:100]}" for r in rows)
    except Exception:
        pass

    # Get recent sessions.
    sessions = []
    try:
        rows = db.query(
            "SELECT id, summary, start_time FROM sessions ORDER BY start_time DESC LIMIT 3"
        )
        sessions = rows or []
    except Exception:
        pass

    # RAG search for trajectory OS context.
    rag_context = ""
    if retriever:
        try:
            results = retriever.search("trajectory OS MAOS agent loop architecture roadmap", top_k=5)
            if results:
                chunks = [f"[{r.source}] {r.text[:500]}" for r in results]
                rag_context = "\n\nIndexed context:\n" + "\n\n".join(chunks)
        except Exception:
            pass

    task_list = _format_tasks_for_model(maos_tasks) if maos_tasks else "  (no MAOS-specific tasks)"
    session_list = "\n".join(
        f"  Session {_task_get(s, 'id', '?')}: {str(_task_get(s, 'summary', ''))[:80]}"
        for s in sessions
    ) if sessions else "  (no recent sessions)"

    trajectory_findings = ""
    try:
        from app.integrations.trajectory import TrajectoryContextProvider

        trajectory_findings = TrajectoryContextProvider.from_environment().get("findings") or ""
    except Exception:
        pass  # Trajectory OS is optional; MAOS state remains useful by itself.

    if model is None:
        lines = ["*Trajectory OS / MAOS Status*\n"]
        if trajectory_findings:
            lines.append(trajectory_findings[:1800])
        lines.append(f"MAOS tasks: {len(maos_tasks)}")
        lines.append(f"Blackboard:\n{blackboard_info}")
        lines.append(f"\nRecent sessions:\n{session_list}")
        lines.append(f"\nTasks:\n{task_list}")
        return {"ok": True, "text": "\n".join(lines)}

    system = f"""You are the trajectory OS architect. Assess the current state of MAOS (Mac AI OS) and recommend the next milestone.

MAOS-specific pending tasks:
{task_list}

Blackboard state:
{blackboard_info}

Recent sessions:
{session_list}
\nTrajectory OS findings (provider-neutral, read-only):
{trajectory_findings[:6000] if trajectory_findings else "  (Trajectory OS findings unavailable)"}
{rag_context}

Produce a trajectory brief:
1. **Current phase** — one sentence: what phase is MAOS in?
2. **Completed** — what's done (from sessions and context)
3. **Next milestone** — the ONE milestone that matters next, with clear definition of done
4. **Critical path** — what's blocking it (or 'clear')

Keep it under 15 lines. This is for Slack. Be direct."""

    try:
        resp = model.chat([
            Message("system", system),
            Message("user", "Generate trajectory OS brief."),
        ])
        text = resp.text.strip() if resp.text else "Trajectory OS: no response from model"
        return {"ok": True, "text": text}
    except Exception as e:
        log.error("trajectory_brief.model_failed", error=str(e))
        return {"ok": False, "text": f"Trajectory brief failed: {e}"}
