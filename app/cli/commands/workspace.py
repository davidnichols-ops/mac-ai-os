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

    lines = [f"{'Name':>20s}  {'Language':>10s}  {'Git':>3s}  {'Files':>5s}  {'Status':>12s}  Last Active"]
    for w in workspaces:
        git = "yes" if w["git_repo"] else "no"
        last = w["last_activity"][:19] if w["last_activity"] else "?"
        status = w["project_status"] or "-"
        lines.append(
            f"{w['name']:>20s}  {w['language'] or '?':>10s}  {git:>3s}  "
            f"{w['file_count']:5d}  {status:>12s}  {last}"
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


@workspace_app.command("profile")
def profile(
    path: str = typer.Argument(None, help="Specific workspace path to profile (default: all)"),
    force: bool = typer.Option(False, "--force", help="Re-profile even if already profiled"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use heuristic profiling only (no model)"),
) -> None:
    """Generate LLM-driven profiles for workspaces.

    For each workspace, reads the README, recent git commits, TODOs, and
    uncommitted changes, then asks the utility model to produce a structured
    profile: what the project is, its status, and open work items.

    Discovered open work items are automatically created as tasks.
    """
    rt = build_runtime()
    from app.workspace_profiler import profile_all_workspaces, profile_workspace

    model = None if no_llm else rt.utility

    if path:
        from pathlib import Path
        p = Path(path).expanduser().resolve()
        result = profile_workspace(rt.db, str(p), model=model)
        if "error" in result:
            typer.echo(result["error"], err=True)
            raise typer.Exit(1)
        _print_profile(result)
    else:
        results = profile_all_workspaces(rt.db, model=model, force=force)
        if not results:
            total = len(rt.db.all_workspaces())
            typer.echo(
                f"All {total} workspaces already profiled. "
                f"Use --force to re-profile."
            )
            return

        for r in results:
            if "error" in r:
                typer.echo(f"  SKIP: {r['error']}")
                continue
            _print_profile(r)
            typer.echo()

        # Summary.
        status_counts: dict[str, int] = {}
        total_tasks = 0
        for r in results:
            s = r.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
            total_tasks += len(r.get("open_work", []))

        summary_lines = [
            f"Profiled: {len(results)} workspaces",
            f"Status: {', '.join(f'{k}={v}' for k, v in sorted(status_counts.items()))}",
            f"Discovered tasks: {total_tasks}",
            f"Total profiled in DB: {rt.db.profiled_workspace_count()}",
        ]
        print_panel("\n".join(summary_lines), title="Profiling Complete")


def _print_profile(result: dict) -> None:
    """Print a single workspace profile."""
    lines = [
        f"Name:   {result['name']}",
        f"Path:   {result['path']}",
        f"Status: {result['status']}",
    ]
    if result.get("git_info", {}).get("branch"):
        lines.append(f"Branch: {result['git_info']['branch']}")
    if result.get("git_info", {}).get("modified_files"):
        lines.append(f"Modified: {len(result['git_info']['modified_files'])} files")
    if result.get("branches"):
        lines.append(f"Branches: {', '.join(result['branches'])}")
    if result.get("open_work"):
        lines.append(f"Open work: {', '.join(result['open_work'])}")
    if result.get("todos"):
        lines.append(f"TODOs: {len(result['todos'])} found")
    lines.append("")
    lines.append("Profile:")
    lines.append(result["profile"])
    print_panel("\n".join(lines), title=f"Workspace: {result['name']}")
