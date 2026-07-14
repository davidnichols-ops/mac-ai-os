"""Workspace profiler — generates LLM-driven project profiles.

For each workspace, gathers:
  - README content (first 2000 chars)
  - Recent git commits (last 10)
  - Current branch and uncommitted changes
  - TODO/FIXME comments in source files
  - Open feature branches

Then asks the utility LLM to produce a structured profile:
  - What the project is (1-2 sentences)
  - Status: active / abandoned / complete / maintenance
  - Open work: discovered tasks and incomplete features
  - Tech stack summary

Profiles are stored in the workspaces table and included in the
context packet, so any AI client connecting via MCP immediately
understands what each project is.
"""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db import Database
from app.log import get_logger
from app.models.base import Model

log = get_logger(__name__)

# Status options for projects.
STATUSES = ("active", "abandoned", "complete", "maintenance")

# How many TODOs to extract per project.
MAX_TODOS = 10
# How many recent commits to include.
MAX_COMMITS = 10
# Max branches to check.
MAX_BRANCHES = 5


def profile_workspace(
    db: Database, workspace_path: str, model: Model | None = None,
) -> dict[str, Any]:
    """Profile a single workspace.

    Returns a dict with: name, path, profile, status, open_work, branches,
    todos, git_info. Stores profile + status in the workspaces table.
    """
    p = Path(workspace_path)
    if not p.exists():
        return {"error": f"path does not exist: {p}"}

    ws = db.workspace_by_path(str(p))
    name = ws["name"] if ws else p.name

    # Gather raw data.
    readme = _read_readme(p)
    git_info = _git_info(str(p)) if (p / ".git").exists() else {}
    todos = _extract_todos(p)
    branches = _list_branches(str(p)) if (p / ".git").exists() else []

    # Build the LLM prompt.
    prompt_parts = [
        f"Project name: {name}",
        f"Path: {p}",
        f"Language: {ws['language'] if ws else 'unknown'}",
        f"File count: {ws['file_count'] if ws else 0}",
    ]

    if readme:
        prompt_parts.append(f"\nREADME (first 2000 chars):\n{readme[:2000]}")

    if git_info.get("recent_commits"):
        commits_text = "\n".join(
            f"  {c['date'][:10]} {c['message'][:80]}" for c in git_info["recent_commits"]
        )
        prompt_parts.append(f"\nRecent commits:\n{commits_text}")

    if git_info.get("branch"):
        prompt_parts.append(f"\nCurrent branch: {git_info['branch']}")

    if git_info.get("modified_files"):
        prompt_parts.append(
            f"\nUncommitted changes ({len(git_info['modified_files'])} files): "
            + ", ".join(git_info["modified_files"][:10])
        )

    if branches:
        prompt_parts.append(f"\nBranches: {', '.join(branches)}")

    if todos:
        todos_text = "\n".join(f"  - {t}" for t in todos[:MAX_TODOS])
        prompt_parts.append(f"\nTODOs found in code:\n{todos_text}")

    prompt = "\n".join(prompt_parts)

    # Ask the LLM to profile this project.
    system = (
        "You are a project analyst. Given project artifacts, produce a concise profile.\n"
        "Respond in EXACTLY this format (no markdown, no code fences):\n"
        "SUMMARY: <1-2 sentences describing what this project is>\n"
        "STATUS: <one of: active, abandoned, complete, maintenance>\n"
        "OPEN_WORK: <comma-separated list of incomplete features or tasks, or 'none'>\n"
        "STACK: <comma-separated tech stack summary>"
    )

    profile_text = ""
    status = "unknown"
    open_work: list[str] = []

    if model:
        try:
            resp = model.generate(prompt, system=system, max_tokens=300)
            profile_text = resp.text.strip()
            status, open_work = _parse_profile(profile_text)
        except Exception as e:
            log.warning("profiler.llm_failed", workspace=name, error=str(e))
            profile_text = _fallback_profile(name, readme, git_info, todos)
            status = _guess_status(git_info)
    else:
        profile_text = _fallback_profile(name, readme, git_info, todos)
        status = _guess_status(git_info)

    # Extract open work as tasks.
    if open_work and open_work != ["none"]:
        _create_discovered_tasks(db, name, str(p), open_work)

    # Store in DB.
    db.update_workspace_profile(str(p), profile=profile_text, project_status=status)

    log.info(
        "workspace.profiled",
        workspace=name,
        status=status,
        open_work=len(open_work),
        has_readme=bool(readme),
        commits=len(git_info.get("recent_commits", [])),
        todos=len(todos),
    )

    return {
        "name": name,
        "path": str(p),
        "profile": profile_text,
        "status": status,
        "open_work": open_work,
        "branches": branches,
        "todos": todos[:MAX_TODOS],
        "git_info": {
            "branch": git_info.get("branch", ""),
            "modified_files": git_info.get("modified_files", []),
            "commit_count": len(git_info.get("recent_commits", [])),
        },
    }


def profile_all_workspaces(
    db: Database, model: Model | None = None, *, force: bool = False,
) -> list[dict[str, Any]]:
    """Profile all registered workspaces.

    If force=False, skips workspaces that already have profiles.
    """
    workspaces = db.all_workspaces()
    results: list[dict[str, Any]] = []

    for ws in workspaces:
        if not force and ws["profile"]:
            log.info("profiler.skip", workspace=ws["name"], reason="already profiled")
            continue
        result = profile_workspace(db, ws["path"], model=model)
        results.append(result)

    log.info(
        "profiler.complete",
        total=len(workspaces),
        profiled=len(results),
        skipped=len(workspaces) - len(results),
    )

    return results


# --- data gathering helpers -------------------------------------------------


def _read_readme(path: Path) -> str:
    """Read README content (tries multiple extensions)."""
    for name in ("README.md", "README.rst", "README.txt", "README", "readme.md"):
        readme = path / name
        if readme.exists():
            try:
                return readme.read_text(errors="ignore")[:3000]
            except (PermissionError, OSError):
                pass
    return ""


def _git_info(path: str) -> dict[str, Any]:
    """Get git info for a workspace (read-only)."""
    info: dict[str, Any] = {}
    try:
        # Current branch.
        branch = subprocess.run(
            ["git", "-C", path, "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        if branch.stdout.strip():
            info["branch"] = branch.stdout.strip()

        # Recent commits.
        log_result = subprocess.run(
            ["git", "-C", path, "log", "--oneline", "--date=short",
             "--pretty=format:%ad %s", "-10"],
            capture_output=True, text=True, timeout=5,
        )
        if log_result.stdout.strip():
            commits = []
            for line in log_result.stdout.strip().split("\n"):
                parts = line.split(" ", 1)
                commits.append({
                    "date": parts[0] if parts else "",
                    "message": parts[1] if len(parts) > 1 else line,
                })
            info["recent_commits"] = commits

        # Modified files.
        status = subprocess.run(
            ["git", "-C", path, "status", "--short"],
            capture_output=True, text=True, timeout=5,
        )
        if status.stdout.strip():
            modified = [
                line.strip().split(" ", 1)[-1]
                for line in status.stdout.strip().split("\n")
                if line.strip()
            ]
            info["modified_files"] = modified
        else:
            info["modified_files"] = []

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def _extract_todos(path: Path) -> list[str]:
    """Extract TODO/FIXME/HACK comments from source files."""
    todos: list[str] = []
    todo_pattern = re.compile(
        r"#\s*(TODO|FIXME|HACK|XXX|BUG)\s*[:\-]?\s*(.+)", re.IGNORECASE
    )

    # Only scan top-level + one level deep, skip common ignore dirs.
    ignore_dirs = {".git", "node_modules", ".venv", "__pycache__", "venv",
                   ".tox", "dist", "build", ".eggs", "target", ".next"}

    files_to_check: list[Path] = []
    for item in path.iterdir():
        if item.is_dir() and item.name not in ignore_dirs:
            for child in item.iterdir():
                if child.is_file() and child.suffix in (".py", ".js", ".ts",
                    ".rs", ".go", ".swift", ".rb", ".java", ".kt", ".sh"):
                    files_to_check.append(child)
        elif item.is_file() and item.suffix in (".py", ".js", ".ts",
            ".rs", ".go", ".swift", ".rb", ".java", ".kt", ".sh"):
            files_to_check.append(item)

    for f in files_to_check[:50]:  # limit to 50 files
        try:
            content = f.read_text(errors="ignore")
            for match in todo_pattern.finditer(content):
                todo_text = match.group(2).strip()[:100]
                if todo_text:
                    todos.append(f"{match.group(1)}: {todo_text} ({f.name})")
                    if len(todos) >= MAX_TODOS:
                        return todos
        except (PermissionError, OSError, UnicodeDecodeError):
            continue

    return todos


def _list_branches(path: str) -> list[str]:
    """List local branches (limited)."""
    try:
        result = subprocess.run(
            ["git", "-C", path, "branch", "--list"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            branches = []
            for line in result.stdout.strip().split("\n"):
                branch = line.strip().lstrip("*").strip()
                if branch:
                    branches.append(branch)
                    if len(branches) >= MAX_BRANCHES:
                        break
            return branches
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


# --- LLM response parsing ---------------------------------------------------


def _parse_profile(text: str) -> tuple[str, list[str]]:
    """Parse the LLM profile response into status and open work items."""
    status = "unknown"
    open_work: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("STATUS:"):
            raw_status = line.split(":", 1)[1].strip().lower()
            for s in STATUSES:
                if s in raw_status:
                    status = s
                    break
        elif line.upper().startswith("OPEN_WORK:"):
            raw_work = line.split(":", 1)[1].strip()
            if raw_work and raw_work.lower() != "none":
                open_work = [w.strip() for w in raw_work.split(",") if w.strip()]

    return status, open_work


def _fallback_profile(
    name: str, readme: str, git_info: dict[str, Any], todos: list[str],
) -> str:
    """Generate a profile without LLM (heuristic-based)."""
    parts = [f"SUMMARY: {name}"]

    if readme:
        # First non-empty, non-header line of README.
        for line in readme.split("\n"):
            line = line.strip().lstrip("#").strip()
            if line and not line.startswith("!["):
                parts[0] = f"SUMMARY: {line[:200]}"
                break

    status = _guess_status(git_info)
    parts.append(f"STATUS: {status}")

    if todos:
        parts.append(f"OPEN_WORK: {', '.join(t[:60] for t in todos[:5])}")
    elif git_info.get("modified_files"):
        parts.append(f"OPEN_WORK: {len(git_info['modified_files'])} uncommitted files")
    else:
        parts.append("OPEN_WORK: none")

    return "\n".join(parts)


def _guess_status(git_info: dict[str, Any]) -> str:
    """Heuristic status guess from git info."""
    commits = git_info.get("recent_commits", [])
    if not commits:
        return "abandoned"

    # Check most recent commit date.
    latest = commits[0] if commits else {}
    latest_date = latest.get("date", "")

    if latest_date:
        try:
            latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
            days_ago = (datetime.now(UTC).date() - latest_dt.date()).days
            if days_ago < 30:
                return "active"
            elif days_ago < 180:
                return "maintenance"
            else:
                return "abandoned"
        except (ValueError, TypeError):
            pass

    return "unknown"


def _create_discovered_tasks(
    db: Database, workspace_name: str, workspace_path: str, work_items: list[str],
) -> None:
    """Create tasks from discovered open work items.

    Checks for duplicates — won't create a task if one with the same
    title already exists.
    """
    existing_titles = {t["title"] for t in db.pending_tasks()}

    for item in work_items:
        item = item.strip()
        if not item or item.lower() == "none":
            continue
        # Create a concise task title.
        title = f"[{workspace_name}] {item[:100]}"
        if title not in existing_titles:
            db.create_task(
                title=title,
                source="discovered",
                related_workspace=workspace_path,
            )
            existing_titles.add(title)
            log.info("profiler.task_created", workspace=workspace_name, task=title[:60])
