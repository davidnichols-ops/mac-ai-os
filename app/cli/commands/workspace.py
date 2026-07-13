"""`maos workspace` — workspace detection and management."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

workspace_app = typer.Typer(name="workspace", help="Workspace management commands.", no_args_is_help=True)


@workspace_app.command("scan")
def scan(
    path: str = typer.Argument(None, help="Additional path to scan (defaults to standard locations)"),
) -> None:
    """Scan for project directories and register them as workspaces."""
    rt = build_runtime()
    from app.workspace import scan_workspaces

    extra = [path] if path else None
    discovered = scan_workspaces(rt.db, extra_paths=extra)

    if not discovered:
        print_panel("No workspaces found. Try specifying a path with projects.", title="Workspace Scan")
        return

    lines = [f"Discovered {len(discovered)} workspace(s):"]
    for w in discovered:
        git = "git" if w["git_repo"] else ""
        lines.append(f"  {w['name']:>20s}  {w['language'] or '?':>10s}  {git:>3s}  {w['file_count']} files")
        lines.append(f"    {w['path']}")
    print_panel("\n".join(lines), title="Workspace Scan")


@workspace_app.command("list")
def list_workspaces() -> None:
    """List all registered workspaces."""
    rt = build_runtime()
    workspaces = rt.db.all_workspaces()
    if not workspaces:
        typer.echo("No workspaces registered. Run `maos workspace scan` first.")
        return

    lines = [f"{'Name':>20s}  {'Language':>10s}  {'Git':>3s}  {'Files':>5s}  Last Active"]
    for w in workspaces:
        git = "yes" if w["git_repo"] else "no"
        last = w["last_activity"][:19] if w["last_activity"] else "?"
        lines.append(
            f"{w['name']:>20s}  {w['language'] or '?':>10s}  {git:>3s}  "
            f"{w['file_count']:5d}  {last}"
        )
    print_panel("\n".join(lines), title=f"Workspaces ({len(workspaces)})")


@workspace_app.command("active")
def active() -> None:
    """Show the most recently active workspace."""
    rt = build_runtime()
    from app.workspace import detect_active_workspace

    ws = detect_active_workspace(rt.db)
    if not ws:
        typer.echo("No active workspace detected.")
        return

    lines = [f"Name:     {ws.get('name', '?')}", f"Path:     {ws.get('path', '?')}"]
    if ws.get("language"):
        lines.append(f"Language: {ws['language']}")
    if ws.get("activity_count"):
        lines.append(f"Activity: {ws['activity_count']} recent file events")
    print_panel("\n".join(lines), title="Active Workspace")
