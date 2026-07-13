"""`maos snapshot` — workspace snapshot commands."""

from __future__ import annotations

from datetime import UTC

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

snapshot_app = typer.Typer(name="snapshot", help="Workspace snapshot commands.", no_args_is_help=True)


@snapshot_app.command("capture")
def capture(
    workspace: str = typer.Argument(None, help="Workspace name (defaults to all)"),
) -> None:
    """Capture a workspace snapshot."""
    rt = build_runtime()
    from app.observers.snapshots import capture_all_workspaces, capture_snapshot

    if workspace:
        ws = rt.db.workspace_by_path(workspace)
        if not ws:
            for w in rt.db.all_workspaces():
                if w["name"] == workspace:
                    ws = w
                    break
        if not ws:
            typer.echo(f"Workspace not found: {workspace}")
            raise typer.Exit(1)
        snap = capture_snapshot(ws["path"], ws["name"], rt.db, rt.bus)
        print_panel(f"Captured snapshot #{snap['id']} for {ws['name']}\n"
                   f"Branch: {snap.get('git_branch', '?')}\n"
                   f"Modified: {snap.get('modified_files', 'none')[:100]}",
                   title="Snapshot Captured")
    else:
        snapshots = capture_all_workspaces(rt.db, rt.bus)
        print_panel(f"Captured {len(snapshots)} snapshot(s)", title="Snapshots Captured")


@snapshot_app.command("list")
def list_snapshots(
    workspace: str = typer.Option("", "--workspace", "-w", help="Filter by workspace"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of snapshots"),
) -> None:
    """List workspace snapshots."""
    rt = build_runtime()
    from datetime import datetime, timedelta

    since = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    snapshots = rt.db.snapshots_since(since, workspace=workspace or None)
    if not snapshots:
        typer.echo("No snapshots found. Run `maos snapshot capture` first.")
        return

    lines = [f"{'Time':>19s}  {'Workspace':>15s}  {'Branch':>15s}  Modified"]
    for s in snapshots[-limit:]:
        modified = s["modified_files"][:40] if s["modified_files"] else "clean"
        lines.append(
            f"{s['timestamp'][:19]:>19s}  {s['workspace']:>15s}  "
            f"{(s['git_branch'] or '?'):>15s}  {modified}"
        )
    print_panel("\n".join(lines), title=f"Snapshots ({len(snapshots)})")


@snapshot_app.command("latest")
def latest(
    workspace: str = typer.Argument(None, help="Workspace name"),
) -> None:
    """Show the latest snapshot."""
    rt = build_runtime()
    snap = rt.db.latest_snapshot(workspace=workspace)
    if not snap:
        typer.echo("No snapshots found.")
        return

    lines = [
        f"Workspace: {snap['workspace']}",
        f"Time:      {snap['timestamp'][:19]}",
        f"Branch:    {snap['git_branch'] or '?'}",
    ]
    if snap["modified_files"]:
        lines.append("Modified files:")
        for f in snap["modified_files"].split("\n")[:10]:
            lines.append(f"  {f}")
    else:
        lines.append("Modified files: none (clean)")
    if snap["active_processes"]:
        lines.append("Active processes:")
        for p in snap["active_processes"].split("\n")[:5]:
            lines.append(f"  {p}")
    print_panel("\n".join(lines), title=f"Latest Snapshot #{snap['id']}")
