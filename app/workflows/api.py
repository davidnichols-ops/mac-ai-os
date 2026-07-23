"""Durable workflow external API — MCP/Context API exposure.

Phase 7: This module exposes the workflow engine through MCP tools,
allowing external clients (LLMs, other agents, CLI) to:

  - List workflow definitions and runs
  - Start/pause/resume/cancel workflows
  - Inspect running workflows (step details, checkpoints, side effects)
  - View workflow history
  - Manage the dead letter queue

These functions are designed to be registered as MCP tools in the
MCP server (app/mcp/server.py).
"""

from __future__ import annotations

import json
from typing import Any

from app.log import get_logger
from app.workflows.engine import WorkflowEngine
from app.workflows.failure import DeadLetterQueue

log = get_logger(__name__)


def _row_get(row: Any, key: str, default: Any = "") -> Any:
    """Safely get a value from a sqlite3.Row, which doesn't support .get()."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def maos_workflow_list_definitions(runtime: Any) -> str:
    """List all registered workflow definitions."""
    db = runtime.db
    defs = db.list_workflow_definitions()
    if not defs:
        return "No workflow definitions found."
    lines = ["Workflow Definitions:"]
    for d in defs:
        created = d["created_at"] if "created_at" in d.keys() else "unknown"
        lines.append(f"  - {d['name']} v{d['version']} (created: {created})")
    return "\n".join(lines)


def maos_workflow_list_runs(runtime: Any, status: str = "", limit: int = 10) -> str:
    """List recent workflow runs, optionally filtered by status."""
    db = runtime.db
    if status:
        runs = db.workflow_runs_by_status(status)[:limit]
    else:
        runs = db.query(
            "SELECT * FROM workflow_runs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
    if not runs:
        return "No workflow runs found."
    lines = [f"Workflow Runs ({'status=' + status if status else 'all'}):"]
    for r in runs:
        lines.append(
            f"  [{r['id']}] {r['workflow']} — {r['status']} "
            f"({r['steps_done']}/{r['steps_total']} steps) "
            f"at {r['timestamp'][:19]}"
        )
    return "\n".join(lines)


def maos_workflow_show(runtime: Any, run_id: int) -> str:
    """Show details of a specific workflow run."""
    db = runtime.db
    engine = WorkflowEngine(db)
    details = engine.get_run_details(run_id)
    if not details:
        return f"Workflow run {run_id} not found."

    run = details["run"]
    steps = details["steps"]
    lines = [
        f"Workflow Run #{run_id}",
        f"  Name: {run['workflow']}",
        f"  Status: {run['status']}",
        f"  Steps: {run['steps_done']}/{run['steps_total']}",
        f"  Workflow ID: {_row_get(run, 'workflow_id', '')}",
        f"  Version: {_row_get(run, 'version', 1)}",
        f"  Started: {run['timestamp'][:19]}",
    ]
    if _row_get(run, "finished_at"):
        lines.append(f"  Finished: {run['finished_at'][:19]}")
    if _row_get(run, "parent_run_id"):
        lines.append(f"  Parent Run: {run['parent_run_id']}")
    if _row_get(run, "heartbeat_at"):
        lines.append(f"  Last Heartbeat: {run['heartbeat_at'][:19]}")

    lines.append("\n  Steps:")
    for s in steps:
        status_icon = "✓" if s["status"] == "completed" else "✗" if s["status"] == "failed" else "→"
        lines.append(
            f"    {status_icon} [{s['step_index']}] {s['step_name']} "
            f"({s['status']}, attempt {s['attempt']}, {_row_get(s, 'duration_ms', 0)}ms)"
        )
        if _row_get(s, "error"):
            lines.append(f"        Error: {s['error'][:100]}")

    # Show checkpoints.
    checkpoints = db.checkpoints_for_workflow(run_id)
    if checkpoints:
        lines.append(f"\n  Checkpoints: {len(checkpoints)}")
        for cp in checkpoints[-3:]:  # last 3
            lines.append(f"    - After step {cp['step_index']} at {cp['created_at'][:19]}")

    # Show side effects.
    effects = db.side_effects_for_workflow(run_id)
    if effects:
        lines.append(f"\n  Side Effects: {len(effects)}")
        for e in effects:
            lines.append(f"    - {e['effect_type']} (key: {e['idempotency_key'][:40]})")

    return "\n".join(lines)


def maos_workflow_cancel(runtime: Any, run_id: int) -> str:
    """Cancel a running workflow."""
    db = runtime.db
    engine = WorkflowEngine(db)
    success = engine.cancel(run_id)
    if success:
        return f"Workflow run {run_id} cancellation requested."
    return f"Cannot cancel run {run_id} (not running or not found)."


def maos_workflow_resume(runtime: Any, run_id: int) -> str:
    """Resume a crashed or interrupted workflow from its last checkpoint."""
    db = runtime.db
    WorkflowEngine(db)

    # Get the workflow definition.
    run = db.get_workflow_run(run_id)
    if not run:
        return f"Run {run_id} not found."

    # Load the workflow definition from the DB.
    def_row = db.get_workflow_definition(run["workflow"])
    if not def_row:
        return f"Workflow definition '{run['workflow']}' not found."

    # Reconstruct the workflow from the definition.
    # Note: handlers can't be serialized, so we can only resume workflows
    # that are registered in code. For now, return an informative message.
    return (
        f"Resume requested for run {run_id} (workflow: {run['workflow']}).\n"
        f"Status: {run['status']}\n"
        f"Note: Workflow resume requires the workflow definition to be "
        f"registered in code. Use the CLI: uv run app workflow resume {run_id}"
    )


def maos_workflow_inspect(runtime: Any, run_id: int) -> str:
    """Inspect a workflow run — alias for maos_workflow_show with more detail."""
    return maos_workflow_show(runtime, run_id)


def maos_workflow_dlq_list(runtime: Any, status: str = "pending") -> str:
    """List entries in the workflow dead letter queue."""
    db = runtime.db
    dlq = DeadLetterQueue(db)
    entries = dlq.list_entries(status=status)
    if not entries:
        return f"No {status} entries in the dead letter queue."
    lines = [f"Dead Letter Queue ({status}):"]
    for e in entries:
        lines.append(
            f"  [{e['id']}] Run #{e['run_id']}: {e['workflow_name']} "
            f"— {e['failure_reason'][:80]} "
            f"(enqueued: {e['enqueued_at'][:19]})"
        )
    return "\n".join(lines)


def maos_workflow_dlq_inspect(runtime: Any, dlq_id: int) -> str:
    """Inspect a specific dead letter queue entry."""
    db = runtime.db
    dlq = DeadLetterQueue(db)
    entry = dlq.get_entry(dlq_id)
    if not entry:
        return f"DLQ entry {dlq_id} not found."

    lines = [
        f"Dead Letter Queue Entry #{dlq_id}",
        f"  Run ID: {entry['run_id']}",
        f"  Workflow: {entry['workflow_name']}",
        f"  Status: {entry['status']}",
        f"  Failure Reason: {entry['failure_reason']}",
        f"  Enqueued: {entry['enqueued_at'][:19]}",
    ]
    if _row_get(entry, "inspected_at"):
        lines.append(f"  Inspected: {entry['inspected_at'][:19]}")
        lines.append(f"  Notes: {entry['notes']}")
    if _row_get(entry, "new_run_id"):
        lines.append(f"  Reprocessed as Run: {entry['new_run_id']}")

    comps = json.loads(_row_get(entry, "compensation_records", "[]"))
    if comps:
        lines.append(f"\n  Compensation Records ({len(comps)}):")
        for c in comps:
            icon = "✓" if c["success"] else "✗"
            lines.append(f"    {icon} {c['step_name']} — {c.get('error', 'success')}")

    ctx = json.loads(_row_get(entry, "context_snapshot", "{}"))
    if ctx:
        lines.append(f"\n  Context Snapshot: {json.dumps(ctx, indent=2)[:500]}")

    return "\n".join(lines)


def maos_workflow_dlq_mark_inspected(runtime: Any, dlq_id: int, notes: str = "") -> str:
    """Mark a DLQ entry as inspected with optional notes."""
    db = runtime.db
    dlq = DeadLetterQueue(db)
    entry = dlq.get_entry(dlq_id)
    if not entry:
        return f"DLQ entry {dlq_id} not found."
    dlq.mark_inspected(dlq_id, notes)
    return f"DLQ entry {dlq_id} marked as inspected."


def maos_workflow_status(runtime: Any) -> str:
    """Get overall workflow engine status."""
    db = runtime.db
    running = db.workflow_runs_by_status("running")
    paused = db.workflow_runs_by_status("paused")
    crashed = db.workflow_runs_by_status("crashed")
    dlq = DeadLetterQueue(db)
    dlq_count = dlq.count()

    lines = [
        "Workflow Engine Status:",
        f"  Running: {len(running)}",
        f"  Paused: {len(paused)}",
        f"  Crashed: {len(crashed)}",
        f"  Dead Letter Queue: {dlq_count}",
    ]
    if running:
        lines.append("\n  Active Runs:")
        for r in running:
            lines.append(f"    [{r['id']}] {r['workflow']} ({r['steps_done']}/{r['steps_total']})")
    if crashed:
        lines.append("\n  Crashed Runs (need recovery):")
        for r in crashed:
            lines.append(f"    [{r['id']}] {r['workflow']} — use 'workflow resume {r['id']}'")

    return "\n".join(lines)


def maos_workflow_recover_crashed(runtime: Any) -> str:
    """Recover crashed workflow runs by marking stale runs as crashed."""
    db = runtime.db
    engine = WorkflowEngine(db)
    crashed = engine.recover_crashed_runs()
    if not crashed:
        return "No stale workflow runs detected."
    lines = [f"Recovered {len(crashed)} crashed workflow runs:"]
    for run_id in crashed:
        lines.append(f"  - Run #{run_id}")
    return "\n".join(lines)
