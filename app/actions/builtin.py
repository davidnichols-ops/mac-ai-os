"""Built-in actions — safe, initial actions for the agency layer.

These are intentionally boring. They build the foundation:
  - open_workspace: open a project directory
  - create_todo: create a task
  - complete_todo: mark a task as done
  - workspace_summary: generate a summary of a workspace's state
  - list_tasks: list pending tasks
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from app.actions.models import Action, ActionRequest, ActionResult
from app.db import Database
from app.log import get_logger

log = get_logger(__name__)


def _open_workspace_action(request: ActionRequest, ctx: dict[str, Any]) -> ActionResult:
    """Open a workspace directory in Finder or the default editor."""
    target = request.target or request.parameters.get("path", "")
    if not target:
        return ActionResult(ok=False, action="open_workspace", target=target,
                            error="no target specified")

    # Resolve workspace path from name if needed.
    db: Database | None = ctx.get("db")
    if db and not target.startswith("/") and not target.startswith("~"):
        ws = db.workspace_by_path(target)
        if ws:
            target = ws["path"]
        else:
            # Try to find by name in registered workspaces.
            workspaces = db.all_workspaces()
            for w in workspaces:
                if w["name"] == target:
                    target = w["path"]
                    break

    try:
        # Open in Finder on macOS.
        subprocess.run(["open", target], check=True, timeout=5)
        return ActionResult(
            ok=True, action="open_workspace", target=target,
            result=f"opened {target} in Finder",
        )
    except subprocess.CalledProcessError as e:
        return ActionResult(
            ok=False, action="open_workspace", target=target,
            error=f"failed to open: {e}",
        )
    except FileNotFoundError:
        return ActionResult(
            ok=False, action="open_workspace", target=target,
            error=f"path not found: {target}",
        )


def _create_todo_action(request: ActionRequest, ctx: dict[str, Any]) -> ActionResult:
    """Create a TODO task."""
    db: Database | None = ctx.get("db")
    if not db:
        return ActionResult(ok=False, action="create_todo", target=request.target,
                            error="no database in context")

    title = request.parameters.get("title", request.target)
    if not title:
        return ActionResult(ok=False, action="create_todo", target=request.target,
                            error="no title specified")

    workspace = request.parameters.get("workspace", "")
    priority = int(request.parameters.get("priority", 0))
    due_date = request.parameters.get("due_date", "")

    task_id = db.create_task(
        title, source="ai", related_workspace=workspace,
        due_date=due_date, priority=priority,
    )
    return ActionResult(
        ok=True, action="create_todo", target=title,
        result=f"created task #{task_id}: {title}",
    )


def _complete_todo_action(request: ActionRequest, ctx: dict[str, Any]) -> ActionResult:
    """Mark a task as completed."""
    db: Database | None = ctx.get("db")
    if not db:
        return ActionResult(ok=False, action="complete_todo", target=request.target,
                            error="no database in context")

    task_id = request.parameters.get("task_id")
    if not task_id:
        return ActionResult(ok=False, action="complete_todo", target=request.target,
                            error="no task_id specified")

    try:
        db.complete_task(int(task_id))
        return ActionResult(
            ok=True, action="complete_todo", target=str(task_id),
            result=f"completed task #{task_id}",
        )
    except (ValueError, TypeError) as e:
        return ActionResult(ok=False, action="complete_todo", target=str(task_id),
                            error=f"invalid task_id: {e}")


def _workspace_summary_action(request: ActionRequest, ctx: dict[str, Any]) -> ActionResult:
    """Generate a summary of a workspace's current state."""
    target = request.target or request.parameters.get("workspace", "")
    db: Database | None = ctx.get("db")
    if not db:
        return ActionResult(ok=False, action="workspace_summary", target=target,
                            error="no database in context")

    # Find workspace.
    ws = None
    if target:
        ws = db.workspace_by_path(target)
        if not ws:
            for w in db.all_workspaces():
                if w["name"] == target:
                    ws = w
                    break

    if not ws:
        return ActionResult(ok=False, action="workspace_summary", target=target,
                            error=f"workspace not found: {target}")

    path = ws["path"]
    import subprocess

    parts: list[str] = [f"Workspace: {ws['name']}", f"Path: {path}"]

    # Git status.
    try:
        status = subprocess.run(
            ["git", "-C", path, "status", "--short"],
            capture_output=True, text=True, timeout=5,
        )
        if status.stdout.strip():
            modified = [line.strip() for line in status.stdout.strip().split("\n") if line.strip()]
            parts.append(f"Modified files ({len(modified)}):")
            for f in modified[:10]:
                parts.append(f"  {f}")
        else:
            parts.append("Modified files: none (clean working tree)")

        branch = subprocess.run(
            ["git", "-C", path, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.stdout.strip():
            parts.append(f"Branch: {branch.stdout.strip()}")

        last_commit = subprocess.run(
            ["git", "-C", path, "log", "-1", "--oneline"],
            capture_output=True, text=True, timeout=5,
        )
        if last_commit.stdout.strip():
            parts.append(f"Last commit: {last_commit.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        parts.append("Git: not available")

    # Pending tasks for this workspace.
    tasks = db.pending_tasks(workspace=path, limit=5)
    if tasks:
        parts.append(f"Pending tasks ({len(tasks)}):")
        for t in tasks[:5]:
            parts.append(f"  [{t['id']}] {t['title']}")
    else:
        parts.append("Pending tasks: none")

    return ActionResult(
        ok=True, action="workspace_summary", target=ws["name"],
        result="\n".join(parts),
    )


def _capture_personal_memory_action(request: ActionRequest, ctx: dict[str, Any]) -> ActionResult:
    """Capture a personal memory through the durable MAOS memory service."""
    memory = ctx.get("memory")
    if memory is None:
        return ActionResult(ok=False, action="capture_personal_memory", target=request.target,
                            error="memory service unavailable")
    result = memory.capture(
        str(request.parameters.get("content", request.target)),
        title=str(request.parameters.get("title", "")),
        tags=tuple(request.parameters.get("tags", ())),
        source=str(request.parameters.get("source", "manual")),
        source_ref=str(request.parameters.get("source_ref", "")),
        author_id=str(request.parameters.get("author_id", "")),
        archive=bool(request.parameters.get("archive", True)),
    )
    if not result.ok:
        return ActionResult(ok=False, action="capture_personal_memory", target=request.target,
                            error=result.error)
    archive = result.archive
    payload = {
        "memory_id": result.memory_id,
        "content_hash": result.content_hash,
        "indexed": result.indexed,
        "duplicate": result.duplicate,
        "archive_status": archive.status if archive else "disabled",
        "archive_path": archive.archive_path if archive else "",
        "archive_verified": archive.verified if archive else False,
        "archive_detail": archive.detail if archive else "",
    }
    return ActionResult(
        ok=True,
        action="capture_personal_memory",
        target=result.memory_id,
        result=json.dumps(payload, sort_keys=True),
    )


def _list_tasks_action(request: ActionRequest, ctx: dict[str, Any]) -> ActionResult:
    """List pending tasks."""
    db: Database | None = ctx.get("db")
    if not db:
        return ActionResult(ok=False, action="list_tasks", target=request.target,
                            error="no database in context")

    workspace = request.parameters.get("workspace", "")
    tasks = db.pending_tasks(workspace=workspace or None, limit=50)
    if not tasks:
        return ActionResult(
            ok=True, action="list_tasks", target=workspace or "all",
            result="No pending tasks.",
        )

    lines = [f"Pending tasks ({len(tasks)}):"]
    for t in tasks:
        lines.append(f"  [{t['id']}] {t['title']}")
    return ActionResult(
        ok=True, action="list_tasks", target=workspace or "all",
        result="\n".join(lines),
    )


def builtin_actions() -> list[tuple[Action, callable]]:
    """Return all built-in actions with their handlers."""
    return [
        (Action(
            name="capture_personal_memory",
            description="Capture, index, and optionally archive a user-authored personal memory",
            permissions=["memory.write"],
            reversible=True,
            parameters={
                "content": "string", "title": "string", "tags": "list[string]",
                "source": "string", "source_ref": "string", "author_id": "string",
                "archive": "bool",
            },
        ), _capture_personal_memory_action),
        (Action(
            name="open_workspace",
            description="Open a project directory in Finder",
            permissions=["filesystem.read", "application.launch"],
            reversible=True,
            parameters={"path": "string", "name": "string"},
        ), _open_workspace_action),
        (Action(
            name="create_todo",
            description="Create a TODO task",
            permissions=["tasks.write"],
            reversible=True,
            parameters={"title": "string", "workspace": "string",
                       "priority": "int", "due_date": "string"},
        ), _create_todo_action),
        (Action(
            name="complete_todo",
            description="Mark a task as completed",
            permissions=["tasks.write"],
            reversible=False,
            parameters={"task_id": "int"},
        ), _complete_todo_action),
        (Action(
            name="workspace_summary",
            description="Generate a summary of a workspace's current state",
            permissions=["filesystem.read", "git.exec"],
            reversible=True,
            parameters={"workspace": "string"},
        ), _workspace_summary_action),
        (Action(
            name="list_tasks",
            description="List pending tasks",
            permissions=["tasks.read"],
            reversible=True,
            parameters={"workspace": "string"},
        ), _list_tasks_action),
    ]
