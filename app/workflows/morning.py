"""Morning routine workflow — "Get me ready to continue where I left off."

This is the first real workflow. It:
1. Reconstructs the last session
2. Summarizes workspace changes
3. Checks pending tasks
4. Checks system health
5. Prepares the workspace (opens it)
6. Creates a daily plan

Each step gathers data and the final output is a synthesized summary.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from app.actions.models import ActionRequest
from app.db import Database
from app.log import get_logger
from app.workflows.models import VerificationResult, Workflow, WorkflowStep

log = get_logger(__name__)


def _step_reconstruct_session(ctx: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the last meaningful work session."""
    db: Database = ctx["db"]
    from app.sessions import detect_sessions, summarize_session

    detect_sessions(db)

    # Find the most substantial recent session — not just the latest one.
    # Look at the last 10 sessions and pick the one with the most events
    # that's older than the current moment (so we don't pick the session
    # that's being created right now by this workflow run).
    sessions = db.recent_sessions(limit=10)
    if not sessions:
        return {"ok": True, "session": None, "summary": "No previous session found."}

    # Filter out trivial sessions and pick the most substantial.
    from datetime import datetime

    now = datetime.now(UTC)
    best = None
    best_score = -1
    for s in sessions:
        try:
            start = datetime.fromisoformat(s["start_time"])
        except (ValueError, TypeError):
            continue
        # Skip sessions that started in the last 5 minutes (likely this run).
        if (now - start).total_seconds() < 300:
            continue
        # Score by event count.
        score = s["event_count"]
        if score > best_score:
            best_score = score
            best = s

    if not best:
        # Fall back to the latest non-trivial session.
        for s in sessions:
            if s["event_count"] >= 2:
                best = s
                break

    if not best:
        return {"ok": True, "session": None, "summary": "No meaningful previous session found."}

    data = summarize_session(db, best["id"])
    return {
        "ok": True,
        "session": data["session"],
        "event_count": data["event_count"],
        "commit_count": data["commit_count"],
        "files_touched": data["files_touched"][:10],
        "project": data["session"].get("project", ""),
    }


def _verify_session(ctx: dict[str, Any]) -> VerificationResult:
    session = ctx.get("session")
    if session:
        return VerificationResult(
            success=True,
            evidence=[f"session found: {session['start_time'][:19]} to {session['end_time'][:19]}"],
        )
    return VerificationResult(success=True, evidence=["no previous session (first run)"])


def _step_summarize_changes(ctx: dict[str, Any]) -> dict[str, Any]:
    """Summarize current workspace changes."""
    db: Database = ctx["db"]
    from app.workspace import detect_active_workspace

    ws = detect_active_workspace(db)
    if not ws:
        return {"ok": True, "workspace": None, "changes": "No active workspace detected."}

    # Convert to dict if it's a sqlite3.Row.
    if not isinstance(ws, dict):
        ws = dict(ws)

    # Get latest snapshot.
    snapshot = db.latest_snapshot(workspace=ws.get("name") or ws.get("path", ""))
    if snapshot:
        snapshot = dict(snapshot)
        modified = snapshot["modified_files"]
        return {
            "ok": True,
            "workspace": ws.get("name", "?"),
            "branch": snapshot.get("git_branch"),
            "modified_files": modified.split("\n")[:10] if modified else [],
            "changes": f"{len(modified.split(chr(10))) if modified else 0} modified files"
                if modified else "clean working tree",
        }
    return {"ok": True, "workspace": ws.get("name", "?"), "changes": "no snapshot available"}


def _step_check_tasks(ctx: dict[str, Any]) -> dict[str, Any]:
    """Check pending tasks."""
    db: Database = ctx["db"]
    tasks = db.pending_tasks()
    return {
        "ok": True,
        "pending_count": len(tasks),
        "tasks": [{"id": t["id"], "title": t["title"]} for t in tasks[:5]],
    }


def _step_check_system_health(ctx: dict[str, Any]) -> dict[str, Any]:
    """Check system health from latest telemetry, collecting if needed."""
    db: Database = ctx["db"]
    from datetime import datetime, timedelta

    since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    telemetry = db.telemetry_since(since, limit=1)

    if not telemetry:
        # Collect a fresh telemetry snapshot so health always reports.
        try:
            from app.telemetry.collector import TelemetryCollector

            collector = TelemetryCollector(db, interval_seconds=1.0)
            collector.collect_once()
            collector.stop()
            telemetry = db.telemetry_since(since, limit=1)
        except Exception as e:
            log.warning("morning.telemetry.collect_failed", error=str(e))

    if not telemetry:
        return {"ok": True, "health": "unavailable", "cpu": None, "memory": None}

    t = telemetry[-1]
    mem_pct = (t["memory_used_gb"] / t["memory_total_gb"] * 100) if t["memory_total_gb"] else 0
    health = "normal"
    if t["cpu_percent"] > 80:
        health = "high CPU"
    if mem_pct > 85:
        health = "memory pressure"

    return {
        "ok": True,
        "health": health,
        "cpu": round(t["cpu_percent"], 1),
        "memory_used": round(t["memory_used_gb"], 1),
        "memory_total": round(t["memory_total_gb"], 1),
        "process_count": t["process_count"],
    }


def _step_prepare_workspace(ctx: dict[str, Any]) -> dict[str, Any]:
    """Prepare the workspace — open it if an action executor is available."""
    workspace = ctx.get("workspace")
    if not workspace:
        return {"ok": True, "prepared": False, "reason": "no active workspace"}

    executor = ctx.get("action_executor")
    if not executor:
        return {"ok": True, "prepared": False, "reason": "no executor available",
                "workspace": workspace}

    # Request to open the workspace.
    request = ActionRequest(action="open_workspace", target=workspace)
    result = executor.execute(request, context={"db": ctx["db"]})
    return {
        "ok": result.ok,
        "prepared": result.ok,
        "workspace": workspace,
        "result": result.result if result.ok else result.error,
    }


def _step_create_daily_plan(ctx: dict[str, Any]) -> dict[str, Any]:
    """Create a daily plan from the gathered context."""
    plan_parts: list[str] = []

    session = ctx.get("session")
    if session:
        project = ctx.get("project", "your project")
        plan_parts.append(f"Continue work on {project}")

    tasks = ctx.get("tasks", [])
    if tasks:
        plan_parts.append(f"Address {len(tasks)} pending task(s)")
        for t in tasks[:3]:
            plan_parts.append(f"  - [{t['id']}] {t['title']}")

    # Check for modified files from the summarize_changes step.
    workspace = ctx.get("workspace")
    modified_files = ctx.get("modified_files", [])
    if modified_files:
        plan_parts.append(f"Review {len(modified_files)} modified files in {workspace or 'workspace'}")
    elif workspace:
        plan_parts.append(f"Working tree is clean in {workspace}")

    health = ctx.get("health", "unknown")
    if health and health not in ("normal", "unknown", "no recent telemetry", "unavailable"):
        plan_parts.append(f"Note: system health is {health}")

    if not plan_parts:
        plan_parts.append("No specific tasks identified — consider indexing new content or running tests")

    return {"ok": True, "plan": plan_parts}


def morning_workflow() -> Workflow:
    """Create the morning routine workflow."""
    wf = Workflow(
        name="morning_workspace_restore",
        description="Reconstruct last session, check state, and prepare workspace",
    )
    wf.add_step(WorkflowStep(
        name="reconstruct_last_session",
        description="Reconstruct the last work session from events, commits, and telemetry",
        handler=_step_reconstruct_session,
        verify=_verify_session,
        required=False,  # first run has no session
    ))
    wf.add_step(WorkflowStep(
        name="summarize_changes",
        description="Summarize current workspace changes from latest snapshot",
        handler=_step_summarize_changes,
        required=False,
    ))
    wf.add_step(WorkflowStep(
        name="check_pending_tasks",
        description="Check for pending tasks",
        handler=_step_check_tasks,
        required=False,
    ))
    wf.add_step(WorkflowStep(
        name="check_system_health",
        description="Check system health from recent telemetry",
        handler=_step_check_system_health,
        required=False,
    ))
    wf.add_step(WorkflowStep(
        name="prepare_workspace",
        description="Open the active workspace",
        handler=_step_prepare_workspace,
        required=False,  # opening Finder is optional
    ))
    wf.add_step(WorkflowStep(
        name="create_daily_plan",
        description="Create a daily plan from gathered context",
        handler=_step_create_daily_plan,
        required=True,
    ))
    return wf


def format_morning_report(result) -> str:
    """Format the workflow result as a readable morning report."""
    if not result.ok and result.error:
        return f"Morning routine failed: {result.error}"

    lines: list[str] = ["Good morning.", ""]

    # Session info.
    for step in result.steps:
        if step.step_name == "reconstruct_last_session" and step.ok:
            session = step.data.get("session")
            if session:
                lines.append("Last session:")
                lines.append(f"  {session.get('start_time', '?')[:19]} → {session.get('end_time', '?')[:19]}")
                project = step.data.get("project")
                if project:
                    lines.append(f"  Project: {project}")
                lines.append(f"  Events: {step.data.get('event_count', 0)}")
                lines.append(f"  Commits: {step.data.get('commit_count', 0)}")
                files = step.data.get("files_touched", [])
                if files:
                    lines.append(f"  Files touched: {len(files)}")
                lines.append("")
            else:
                lines.append("No previous session found.")
                lines.append("")

    # Workspace changes.
    for step in result.steps:
        if step.step_name == "summarize_changes" and step.ok:
            ws = step.data.get("workspace")
            if ws:
                lines.append(f"Current workspace: {ws}")
                branch = step.data.get("branch")
                if branch:
                    lines.append(f"  Branch: {branch}")
                modified = step.data.get("modified_files", [])
                if modified:
                    lines.append(f"  Modified files ({len(modified)}):")
                    for f in modified[:5]:
                        lines.append(f"    {f}")
                else:
                    lines.append("  Working tree: clean")
                lines.append("")

    # Tasks.
    for step in result.steps:
        if step.step_name == "check_pending_tasks" and step.ok:
            count = step.data.get("pending_count", 0)
            tasks = step.data.get("tasks", [])
            if count > 0:
                lines.append(f"Pending tasks ({count}):")
                for t in tasks:
                    lines.append(f"  [{t['id']}] {t['title']}")
            else:
                lines.append("No pending tasks.")
            lines.append("")

    # System health.
    for step in result.steps:
        if step.step_name == "check_system_health" and step.ok:
            health = step.data.get("health", "unknown")
            cpu = step.data.get("cpu")
            mem_used = step.data.get("memory_used")
            mem_total = step.data.get("memory_total")
            if cpu is not None:
                lines.append(f"System: {health} (CPU {cpu}%, Memory {mem_used}/{mem_total}GB)")
            else:
                lines.append(f"System: {health}")
            lines.append("")

    # Daily plan.
    for step in result.steps:
        if step.step_name == "create_daily_plan" and step.ok:
            plan = step.data.get("plan", [])
            lines.append("Recommended plan:")
            for item in plan:
                lines.append(f"  {item}")

    return "\n".join(lines)
