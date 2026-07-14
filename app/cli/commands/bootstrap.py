"""maos bootstrap — first-run data acquisition.

Takes a new install from empty database to usable context engine by:
  1. Scanning for workspaces
  2. Importing shell history
  3. Collecting initial telemetry
  4. Indexing git history for configured repos
  5. Generating initial context (attention cycle + blackboard)

After bootstrap, the system has enough data for `maos context evaluate`
to produce meaningful scores and for MCP clients to receive useful
context packets.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

from app.runtime import build_runtime
from app.ui.console import print_panel

console = Console()


def bootstrap(
    scan_path: str = typer.Option(
        "~/Projects", "--scan-path", "-s",
        help="Directory to scan for workspaces.",
    ),
    git_repos: list[str] = typer.Option(
        None, "--git-repo", "-g",
        help="Git repo paths to index (can specify multiple).",
    ),
    telemetry_seconds: int = typer.Option(
        10, "--telemetry-seconds", "-t",
        help="How long to collect telemetry.",
    ),
    skip_git: bool = typer.Option(False, "--skip-git", help="Skip git indexing."),
    skip_telemetry: bool = typer.Option(False, "--skip-telemetry", help="Skip telemetry collection."),
    skip_shell: bool = typer.Option(False, "--skip-shell", help="Skip shell history import."),
) -> None:
    """Bootstrap MAOS from empty database to usable context engine.

    Runs all data acquisition steps in sequence. Safe to re-run —
    each step is idempotent (skips already-imported data).
    """
    rt = build_runtime()

    steps_completed: list[str] = []
    steps_failed: list[str] = []

    console.print("\n[bold cyan]MAOS Bootstrap[/bold cyan]")
    console.print("Goal: populate memory, context, and activity data.\n")

    # --- Step 1: Workspace scan ---
    console.print("[1/5] Scanning for workspaces...")
    try:
        from app.workspace import scan_workspaces

        extra = [str(Path(scan_path).expanduser())] if scan_path else None
        discovered = scan_workspaces(rt.db, extra_paths=extra)
        steps_completed.append(f"workspace scan: {len(discovered)} workspaces found")
        if discovered:
            for w in discovered[:5]:
                console.print(f"  - {w['name']} ({w['language'] or '?'}) {w['path']}")
        else:
            console.print("  [yellow]No workspaces found.[/yellow]")
    except Exception as e:
        steps_failed.append(f"workspace scan: {e}")
        console.print(f"  [red]Failed: {e}[/red]")

    # --- Step 2: Shell history import ---
    if not skip_shell:
        console.print("\n[2/5] Importing shell history...")
        try:
            from app.observers.shell import import_shell_history

            count = import_shell_history(rt.db, shell="auto")
            total = rt.db.shell_command_count()
            steps_completed.append(f"shell history: {count} new, {total} total")
            console.print(f"  Imported {count} commands ({total} total in DB)")
        except Exception as e:
            steps_failed.append(f"shell history: {e}")
            console.print(f"  [red]Failed: {e}[/red]")
    else:
        console.print("\n[2/5] Shell history: [yellow]skipped[/yellow]")

    # --- Step 3: Telemetry collection ---
    if not skip_telemetry:
        console.print(f"\n[3/5] Collecting telemetry ({telemetry_seconds}s)...")
        try:
            from app.telemetry.collector import TelemetryCollector

            collector = TelemetryCollector(rt.db, rt.bus, interval_seconds=5.0)
            collector.start()
            time.sleep(telemetry_seconds)
            collector.stop()
            count = rt.db.telemetry_count()
            steps_completed.append(f"telemetry: {count} snapshots")
            console.print(f"  Collected {count} snapshots")
        except Exception as e:
            steps_failed.append(f"telemetry: {e}")
            console.print(f"  [red]Failed: {e}[/red]")
    else:
        console.print("\n[3/5] Telemetry: [yellow]skipped[/yellow]")

    # --- Step 4: Git indexing ---
    if not skip_git:
        repos_to_index = list(git_repos) if git_repos else []
        # Auto-discover git repos from scanned workspaces.
        if not repos_to_index:
            for w in rt.db.all_workspaces():
                if w["git_repo"] and Path(w["path"]).exists():
                    repos_to_index.append(w["path"])

        if repos_to_index:
            console.print(f"\n[4/5] Indexing {len(repos_to_index)} git repo(s)...")
            from app.retrieval.indexers.git import GitIndexer

            indexer = GitIndexer(rt.db)
            total_commits = 0
            for repo_path in repos_to_index[:5]:  # limit to 5 repos
                try:
                    p = Path(repo_path).expanduser().resolve()
                    if not (p / ".git").exists():
                        console.print(f"  [yellow]Skip {p.name}: not a git repo[/yellow]")
                        continue
                    chunks = indexer.index(str(p))
                    if chunks:
                        rt.retriever.index(str(p), chunks)
                    total_commits += len(chunks)
                    console.print(f"  {p.name}: {len(chunks)} commits")
                except Exception as e:
                    console.print(f"  [red]{Path(repo_path).name}: {e}[/red]")
            steps_completed.append(f"git indexing: {total_commits} commits from {len(repos_to_index)} repos")
        else:
            console.print("\n[4/5] Git indexing: [yellow]no repos found[/yellow]")
            steps_completed.append("git indexing: no repos found")
    else:
        console.print("\n[4/5] Git indexing: [yellow]skipped[/yellow]")

    # --- Step 5: Initial context generation ---
    console.print("\n[5/5] Generating initial context...")
    try:
        from app.attention import AttentionDaemon

        # Run a single attention cycle to populate the blackboard.
        daemon = AttentionDaemon(rt.db, rt.blackboard, interval_seconds=999)
        # Manually run one cycle.
        daemon.cycle()
        bb = rt.blackboard.snapshot()
        steps_completed.append(
            f"context: goal='{bb['current_goal'][:40]}', "
            f"workspace='{bb['current_workspace']}', "
            f"confidence={bb['confidence']:.2f}"
        )
        console.print(f"  Goal: {bb['current_goal'][:50]}")
        console.print(f"  Workspace: {bb['current_workspace']}")
        console.print(f"  Confidence: {bb['confidence']:.2f}")
    except Exception as e:
        steps_failed.append(f"context generation: {e}")
        console.print(f"  [red]Failed: {e}[/red]")

    # --- Summary ---
    console.print("\n[bold]Bootstrap Summary[/bold]\n")

    # Database stats.
    db = rt.db
    stats_lines = [
        f"Workspaces:      {len(db.all_workspaces())}",
        f"Shell commands:  {db.shell_command_count()}",
        f"Telemetry:       {db.telemetry_count()}",
        f"Git commits:     {db.commit_count()}",
        f"Indexed files:   {db.file_hash_count()}",
        f"Sessions:        {db.session_count()}",
        f"Tasks:           {db.task_count()}",
        f"Events:          {db.query_one('SELECT COUNT(*) as n FROM events')['n']}",
        f"Entities:        {db.query_one('SELECT COUNT(*) as n FROM entities')['n']}",
    ]

    print_panel("\n".join(stats_lines), title="Database State")

    if steps_failed:
        console.print(f"\n[yellow]{len(steps_failed)} step(s) had issues:[/yellow]")
        for s in steps_failed:
            console.print(f"  - {s}")

    console.print("\n[green]Bootstrap complete.[/green]")
    console.print("Next steps:")
    console.print("  1. Run [cyan]maos context evaluate[/cyan] to check context quality")
    console.print("  2. Run [cyan]maos mcp serve --transport http[/cyan] to start the MCP server")
    console.print("  3. Connect an AI client and ask 'What should I work on?'")
