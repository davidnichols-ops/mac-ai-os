"""Umbrella control-plane views for MAOS Slack and MCP consumers.

This module is read-only by design. It assembles the current computer,
portfolio, repository, CLI, and MCP surface into one bounded status packet.
Mutations remain behind MAOS actions, approvals, and audit logging.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.db import Database


@dataclass(frozen=True, slots=True)
class UmbrellaSnapshot:
    """Evidence-backed summary of the MAOS-managed computer surface."""

    workspace_count: int
    pending_tasks: int
    repositories: tuple[dict[str, str], ...]
    custom_clis: tuple[str, ...]
    mcp_configs: tuple[str, ...]
    active_focus: str
    docs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def _repositories(root: Path) -> tuple[dict[str, str], ...]:
    if not root.is_dir():
        return ()
    records: list[dict[str, str]] = []
    for path in sorted(root.iterdir()):
        if not (path / ".git").is_dir():
            continue
        records.append(
            {
                "name": path.name,
                "branch": _git(path, "branch", "--show-current"),
                "remote": _git(path, "remote", "get-url", "origin") or "none",
                "dirty": "yes" if _git(path, "status", "--porcelain") else "no",
            }
        )
    return tuple(records)


def _mcp_configs(home: Path) -> tuple[str, ...]:
    roots = (home / ".config", home / ".local" / "share", home / ".devin", home / ".codex")
    paths: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and ("mcp" in path.name.lower() or path.name in {"config.json", "settings.json"}):
                    paths.add(str(path))
        except OSError:
            continue
    return tuple(sorted(paths))[:100]


def build_snapshot(db: Database, *, home: Path | None = None) -> UmbrellaSnapshot:
    """Build a bounded snapshot from local evidence, without model inference."""
    home = home or Path.home()
    workspaces = db.all_workspaces()
    focus = ""
    try:
        row = db.query_one("SELECT value FROM user_model WHERE key = 'portfolio_focus'")
        focus = str(row["value"]) if row else ""
    except Exception:
        focus = ""
    docs = tuple(
        str(path.relative_to(home / "mac-ai-os"))
        for path in sorted((home / "mac-ai-os" / "docs").glob("**/*.md"))
        if path.is_file()
    )[:80]
    cli_roots = (home / "bin", home / ".grok" / "bin")
    clis = tuple(
        sorted({str(path) for root in cli_roots if root.is_dir() for path in root.iterdir() if path.is_file()})
    )
    return UmbrellaSnapshot(
        workspace_count=len(workspaces),
        pending_tasks=db.task_count(),
        repositories=_repositories(home / "Projects"),
        custom_clis=clis,
        mcp_configs=_mcp_configs(home),
        active_focus=focus,
        docs=docs,
    )


def render_snapshot(snapshot: UmbrellaSnapshot) -> str:
    """Render a Slack-sized umbrella report."""
    dirty = sum(item["dirty"] == "yes" for item in snapshot.repositories)
    with_remote = sum(item["remote"] != "none" for item in snapshot.repositories)
    lines = [
        "🧭 *MAOS Umbrella*",
        f"Workspaces: {snapshot.workspace_count} | Pending tasks: {snapshot.pending_tasks}",
        f"Repositories: {len(snapshot.repositories)} ({with_remote} with origin, {dirty} dirty)",
        f"Custom CLIs: {len(snapshot.custom_clis)} | MCP configs: {len(snapshot.mcp_configs)}",
        f"Docs indexed: {len(snapshot.docs)}",
        "",
        "*Operating model:* observe → delegate → verify → brief David",
        "*Safety:* read-only umbrella views; mutations stay behind MAOS approval and audit.",
    ]
    if snapshot.active_focus:
        lines.insert(2, f"Focus: {snapshot.active_focus[:180]}")
    if snapshot.repositories:
        lines.extend(["", "*Portfolio repos*"])
        for item in snapshot.repositories[:12]:
            marker = "⚠️" if item["dirty"] == "yes" else "✅"
            lines.append(f"{marker} `{item['name']}` — {item['branch'] or '?'} — {item['remote']}")
    return "\n".join(lines)


def snapshot_json(snapshot: UmbrellaSnapshot) -> str:
    """Serialize a snapshot for MCP, logs, or downstream agents."""
    return json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
