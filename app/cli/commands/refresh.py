"""`maos refresh` — unified refresh of the entire operating context.

Replaces the standalone `scripts/refresh_operating_context.py` with a
first-class CLI command. Runs the full observe → index → retrieve →
reconcile pipeline with configurable scope, checkpointing, and
conditional external dependencies.

Usage:
    maos refresh                          # incremental (default)
    maos refresh --full                   # full re-index of all workspaces
    maos refresh --scope context          # just context evaluation
    maos refresh --scope storage          # just storage/hidden scan
    maos refresh --workspace ~/Projects/foo  # refresh single workspace
    maos refresh --skip-trajectory        # skip Trajectory OS integration
    maos refresh --dry-run                # show what would run
    maos refresh --json                   # JSON output
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# Module-level constants — match refresh_operating_context.py
ROOT = Path(__file__).resolve().parents[3]  # mac-ai-os root
TRAJECTORY_ROOT = ROOT.parent / "Projects" / "trajectory-os"
MAOS_DB = ROOT / "data" / "maos.db"


def _run_step(
    name: str,
    command: list[str],
    cwd: Path,
    *,
    timeout: int = 120,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute a refresh step and capture evidence."""
    started = datetime.now(UTC).isoformat()
    if dry_run:
        return {
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "started_at": started,
            "exit_code": 0,
            "dry_run": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 0,
        }
    t0 = time.monotonic()
    try:
        # Strip inherited VIRTUAL_ENV so uv uses the target project's venv
        # (fixes cross-project calls like Trajectory OS from MAOS refresh).
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        result = subprocess.run(
            command, cwd=cwd, env=env, text=True, capture_output=True, check=False, timeout=timeout
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = f"timed out after {timeout}s\n{(exc.stderr or '') if isinstance(exc.stderr, str) else ''}"
    duration_ms = int((time.monotonic() - t0) * 1000)
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "started_at": started,
        "exit_code": exit_code,
        "dry_run": False,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "duration_ms": duration_ms,
    }


def _workspace_paths() -> list[Path]:
    """Get deduplicated workspace paths from the MAOS database."""
    if not MAOS_DB.exists():
        return []
    conn = sqlite3.connect(f"file:{MAOS_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT path FROM workspaces WHERE path != '' ORDER BY path"
        ).fetchall()
        paths = sorted(
            {Path(row[0]).resolve() for row in rows if Path(row[0]).exists()},
            key=str,
        )
        # Exclude parent containers if children are registered
        return [
            path
            for path in paths
            if not any(other != path and other.is_relative_to(path) for other in paths)
        ]
    finally:
        conn.close()


def _build_steps(
    *,
    full_index: bool,
    scope: str,
    workspace: str | None,
    skip_trajectory: bool,
    skip_storage: bool,
    skip_security: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Build the ordered list of refresh steps based on scope and flags."""
    steps: list[dict[str, Any]] = []
    uv = ["uv", "run"]

    # --- Always: external session ingestion ---
    if scope in ("all", "sessions"):
        steps.append(
            _run_step(
                "external-session-ingest",
                [*uv, "python", "-c",
                 "from pathlib import Path; from app.external_sessions import ingest; "
                 "import json; print(json.dumps(ingest(Path('data'))))"],
                ROOT,
                dry_run=dry_run,
            )
        )

    # --- Slack history ingestion ---
    if scope in ("all", "slack"):
        command = [*uv, "app", "slack", "sync", "--json"]
        if full_index:
            command.append("--full")
        steps.append(
            _run_step(
                "slack-history-sync",
                command,
                ROOT,
                timeout=300,
                dry_run=dry_run,
            )
        )

    # --- Always: project activity audit ---
    if scope in ("all", "context") and not workspace:
        audit_script = ROOT / "scripts" / "audit_project_activity.py"
        if audit_script.exists():
            steps.append(
                _run_step(
                    "project-activity-audit",
                    [*uv, "python", "scripts/audit_project_activity.py"],
                    ROOT,
                    dry_run=dry_run,
                )
            )

    # --- Confidence adapters (trustcard + cvconform) ---
    if scope in ("all", "confidence"):
        steps.append(
            _run_step(
                "live-confidence-adapters",
                [*uv, "python", "-c",
                 "from pathlib import Path; from app.confidence_adapters import run; "
                 "import json; result=run(Path('.'), Path('data/confidence_adapters.json')); "
                 "print(json.dumps(result))"],
                ROOT,
                timeout=90,
                dry_run=dry_run,
            )
        )

    # --- Workspace scan + profile ---
    if scope in ("all", "workspaces") and not workspace:
        steps.append(
            _run_step(
                "workspace-scan",
                [*uv, "app", "workspace", "scan", str(Path.home() / "Projects")],
                ROOT,
                dry_run=dry_run,
            )
        )
        steps.append(
            _run_step(
                "workspace-profile",
                [*uv, "app", "workspace", "profile", "--no-llm"],
                ROOT,
                dry_run=dry_run,
            )
        )

    # --- File + git indexing ---
    if scope in ("all", "workspaces"):
        if workspace:
            ws_path = Path(workspace).expanduser().resolve()
            steps.append(
                _run_step(
                    f"file-index:{ws_path.name}",
                    [*uv, "app", "index", str(ws_path)],
                    ROOT,
                    dry_run=dry_run,
                )
            )
            if (ws_path / ".git").exists():
                steps.append(
                    _run_step(
                        f"git-index:{ws_path.name}",
                        [*uv, "app", "git", "index", str(ws_path)],
                        ROOT,
                        dry_run=dry_run,
                    )
                )
        elif full_index:
            for path in _workspace_paths():
                steps.append(
                    _run_step(
                        f"file-index:{path.name}",
                        [*uv, "app", "index", str(path)],
                        ROOT,
                        dry_run=dry_run,
                    )
                )
                if (path / ".git").exists():
                    steps.append(
                        _run_step(
                            f"git-index:{path.name}",
                            [*uv, "app", "git", "index", str(path)],
                            ROOT,
                            dry_run=dry_run,
                        )
                    )

    # --- Shell history import ---
    if scope in ("all", "context"):
        steps.append(
            _run_step(
                "history-import",
                [*uv, "app", "history", "import"],
                ROOT,
                dry_run=dry_run,
            )
        )

    # --- Context evaluation ---
    if scope in ("all", "context"):
        steps.append(
            _run_step(
                "context-evaluate",
                [*uv, "app", "context", "evaluate"],
                ROOT,
                dry_run=dry_run,
            )
        )

    # --- Session search rebuild ---
    if scope in ("all", "context"):
        steps.append(
            _run_step(
                "session-search-rebuild",
                [*uv, "python", "-c",
                 "from app.db import Database; from app.session_search import SessionSearch; "
                 "from pathlib import Path; db=Database(str(Path('data/maos.db'))); "
                 "print('indexed', SessionSearch(db).rebuild())"],
                ROOT,
                dry_run=dry_run,
            )
        )

    # --- Storage scan ---
    if scope in ("all", "storage") and not skip_storage:
        steps.append(
            _run_step(
                "storage-scan",
                [*uv, "app", "storage", "scan", "--mode", "fast", "--json"],
                ROOT,
                timeout=60,
                dry_run=dry_run,
            )
        )
        steps.append(
            _run_step(
                "storage-hidden",
                [*uv, "app", "storage", "hidden", "--json"],
                ROOT,
                timeout=60,
                dry_run=dry_run,
            )
        )

    # --- Trajectory OS integration ---
    if scope in ("all", "trajectory") and not skip_trajectory and TRAJECTORY_ROOT.exists():
        steps.append(
            _run_step(
                "trajectory-ingest-dry-run",
                [*uv, "trajectory", "ingest-maos", "--maos-db", str(MAOS_DB)],
                TRAJECTORY_ROOT,
                dry_run=dry_run,
            )
        )
        steps.append(
            _run_step(
                "trajectory-findings",
                [*uv, "trajectory", "findings", "--format", "json"],
                TRAJECTORY_ROOT,
                dry_run=dry_run,
            )
        )

    # --- Evidence reconciliation ---
    if scope in ("all", "evidence"):
        reconcile_script = ROOT / "scripts" / "reconcile_evidence.py"
        if reconcile_script.exists():
            steps.append(
                _run_step(
                    "evidence-reconciliation",
                    [*uv, "python", "scripts/reconcile_evidence.py"],
                    ROOT,
                    dry_run=dry_run,
                )
            )

    # --- Task handoff ---
    if scope in ("all", "context"):
        steps.append(
            _run_step(
                "task-handoff",
                [*uv, "app", "task", "list"],
                ROOT,
                dry_run=dry_run,
            )
        )

    return steps


def refresh(
    full: bool = typer.Option(
        False, "--full", "-f",
        help="Full re-index of every registered workspace (slower).",
    ),
    scope: str = typer.Option(
        "all", "--scope", "-s",
        help="Scope: all, slack, sessions, context, workspaces, storage, trajectory, evidence, confidence.",
    ),
    workspace: str = typer.Option(
        None, "--workspace", "-w",
        help="Refresh a single workspace path only.",
    ),
    skip_trajectory: bool = typer.Option(
        False, "--skip-trajectory",
        help="Skip Trajectory OS integration.",
    ),
    skip_storage: bool = typer.Option(
        False, "--skip-storage",
        help="Skip storage/hidden data scanning.",
    ),
    skip_security: bool = typer.Option(
        False, "--skip-security",
        help="Skip security campaign integration.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would run without executing.",
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Emit JSON report only.",
    ),
) -> None:
    """Refresh the entire MAOS operating context.

    Runs the full observe → index → retrieve → reconcile pipeline.
    By default: incremental (only indexes changed files). Use --full
    for a complete re-index of every registered workspace.

    Scopes:
        all         — everything (default)
        sessions    — external AI session ingestion only
        slack       — Slack history ingestion and search refresh
        context     — context eval + session search + history + tasks
        workspaces  — workspace scan + profile + file/git indexing
        storage     — disk storage + hidden data scan
        trajectory  — Trajectory OS ingest + findings
        evidence    — evidence reconciliation
        confidence  — trustcard + cvconform adapters
    """
    steps = _build_steps(
        full_index=full,
        scope=scope,
        workspace=workspace,
        skip_trajectory=skip_trajectory,
        skip_storage=skip_storage,
        skip_security=skip_security,
        dry_run=dry_run,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": scope,
        "full_index": full,
        "workspace": workspace,
        "dry_run": dry_run,
        "steps": steps,
        "all_passed": all(step["exit_code"] == 0 for step in steps),
        "total_steps": len(steps),
        "passed": sum(1 for s in steps if s["exit_code"] == 0),
        "failed": sum(1 for s in steps if s["exit_code"] != 0),
        "total_duration_ms": sum(s.get("duration_ms", 0) for s in steps),
        "next_action": "Review MAOS task list after refresh; confirm or reject Trajectory candidates separately.",
    }

    # Save report to data/refresh_report.json
    report_path = ROOT / "data" / "refresh_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["all_passed"] else "ATTENTION"
        console.print(f"\n[bold cyan]MAOS Refresh[/bold cyan] — {status}")
        console.print(f"Scope: {scope} | Full: {full} | Steps: {report['total_steps']}")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Status", width=6)
        table.add_column("Step", style="cyan")
        table.add_column("Duration", justify="right")
        for step in steps:
            status_str = "[green]PASS[/green]" if step["exit_code"] == 0 else "[red]FAIL[/red]"
            if step.get("dry_run"):
                status_str = "[yellow]DRY[/yellow]"
            dur = f"{step.get('duration_ms', 0)}ms"
            table.add_row(status_str, step["name"], dur)
        console.print(table)

        if report["failed"]:
            console.print(f"\n[red]{report['failed']} step(s) failed:[/red]")
            for step in steps:
                if step["exit_code"] != 0 and not step.get("dry_run"):
                    console.print(f"  [red]✗[/red] {step['name']}: {step['stderr_tail'][:200]}")

        console.print(f"\nReport saved: {report_path}")
        console.print(f"[dim]{report['next_action']}[/dim]")

    raise typer.Exit(0 if report["all_passed"] else 1)
