"""Workspace detection — identifies active projects on the machine.

A workspace is a directory that looks like a project: has a .git directory,
contains source files, or shows up in file activity. The detector scans
known locations and registers workspaces in the database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)

# Common project locations to scan.
DEFAULT_SCAN_PATHS = [
    "~/Projects",
    "~/Developer",
    "~/Code",
    "~/repos",
    "~/src",
    "~/Documents/Projects",
    "~/Desktop",
]

# File extensions that indicate a project's primary language.
_LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".rs": "rust", ".go": "go", ".swift": "swift",
    ".java": "java", ".kt": "kotlin", ".rb": "ruby",
    ".cpp": "cpp", ".c": "c", ".h": "c",
    ".m": "objc", ".mm": "objc",
    ".sh": "shell", ".lua": "lua", ".ex": "elixir",
    ".exs": "elixir", ".heex": "elixir",
    ".dart": "dart", ".scala": "scala", ".clj": "clojure",
}

# Files that indicate a project root.
_PROJECT_MARKERS = {
    ".git", "package.json", "pyproject.toml", "Cargo.toml",
    "go.mod", "Package.swift", "pom.xml", "build.gradle",
    "Gemfile", "mix.exs", "pubspec.yaml", "WORKSPACE",
    "Makefile", "CMakeLists.txt",
}


def detect_language(path: Path) -> str:
    """Detect the primary language of a project by counting file extensions."""
    counts: dict[str, int] = {}
    try:
        for item in path.iterdir():
            if item.is_file():
                lang = _LANGUAGE_MAP.get(item.suffix)
                if lang:
                    counts[lang] = counts.get(lang, 0) + 1
    except (PermissionError, OSError):
        pass
    if not counts:
        return ""
    return max(counts, key=counts.get)


def is_project(path: Path) -> bool:
    """Check if a directory looks like a project."""
    if not path.is_dir():
        return False
    for marker in _PROJECT_MARKERS:
        if (path / marker).exists():
            return True
    return False


def count_source_files(path: Path, max_depth: int = 2) -> int:
    """Count source files in a project (limited depth for speed)."""
    count = 0
    try:
        for item in path.rglob("*"):
            if item.is_file() and item.suffix in _LANGUAGE_MAP:
                count += 1
                if count > 500:
                    break
    except (PermissionError, OSError):
        pass
    return count


def scan_workspaces(
    db: Database,
    *,
    scan_paths: list[str] | None = None,
    extra_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Scan for project directories and register them as workspaces."""
    paths_to_scan = scan_paths or DEFAULT_SCAN_PATHS
    if extra_paths:
        paths_to_scan = list(paths_to_scan) + extra_paths

    discovered: list[dict[str, Any]] = []

    for scan_path in paths_to_scan:
        p = Path(scan_path).expanduser()
        if not p.exists() or not p.is_dir():
            continue

        # Check if the scan path itself is a project.
        if is_project(p):
            _register_workspace(db, p, discovered)

        # Check immediate children (depth 1).
        try:
            for child in p.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    if is_project(child):
                        _register_workspace(db, child, discovered)
        except (PermissionError, OSError):
            continue

    log.info("workspace.scan.done", discovered=len(discovered))
    return discovered


def _register_workspace(
    db: Database, path: Path, discovered: list[dict[str, Any]]
) -> None:
    """Register a workspace in the database."""
    resolved = str(path.resolve())
    language = detect_language(path)
    git_repo = (path / ".git").exists()
    file_count = count_source_files(path) if git_repo or language else 0
    last_activity = datetime.now(UTC).isoformat()

    ws_id = db.upsert_workspace(
        resolved,
        name=path.name,
        language=language,
        git_repo=git_repo,
        file_count=file_count,
        last_activity=last_activity,
    )
    discovered.append({
        "id": ws_id,
        "path": resolved,
        "name": path.name,
        "language": language,
        "git_repo": git_repo,
        "file_count": file_count,
    })
    log.info("workspace.registered", name=path.name, path=resolved,
             language=language, git=git_repo)


def detect_active_workspace(db: Database) -> dict[str, Any] | None:
    """Detect the most recently active workspace.

    Priority:
    1. Current working directory if it's a registered workspace
    2. Most recently active workspace from FILE_INDEXED events
    3. Most recently registered workspace
    """
    import os

    # 1. Check if CWD is a registered workspace.
    cwd = os.getcwd()
    ws = db.workspace_by_path(cwd)
    if ws:
        return dict(ws)

    # Also check if CWD is a project that hasn't been registered yet.
    cwd_path = Path(cwd)
    if is_project(cwd_path):
        # Register it on the fly.
        _register_workspace(db, cwd_path, [])
        ws = db.workspace_by_path(cwd)
        if ws:
            return dict(ws)

    # 2. Look at recent FILE_INDEXED events to find the most active source.
    import json

    events = db.query(
        """SELECT payload, timestamp FROM events
           WHERE type = 'FILE_INDEXED'
           ORDER BY timestamp DESC LIMIT 50""",
    )

    if events:
        from collections import Counter

        workspace_counts: Counter[str] = Counter()
        for e in events:
            try:
                payload = json.loads(e["payload"])
                source = payload.get("source", "")
                if source:
                    workspace_counts[source] += 1
            except Exception:
                continue

        if workspace_counts:
            most_active = workspace_counts.most_common(1)[0][0]
            ws = db.workspace_by_path(most_active)
            if ws:
                return dict(ws)
            return {"path": most_active, "name": Path(most_active).name, "activity_count": workspace_counts[most_active]}

    # 3. Fall back to the most recently registered workspace.
    recent = db.query("SELECT * FROM workspaces ORDER BY last_activity DESC LIMIT 1")
    if recent:
        return dict(recent[0])

    return None
