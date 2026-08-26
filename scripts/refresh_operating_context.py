#!/usr/bin/env python3
"""Run the canonical MAOS → Trajectory refresh and emit an auditable report.

Default mode refreshes workspace metadata and validates the existing context.
Use --full-index when a complete incremental pass over every registered
workspace is justified. No Trajectory evidence is confirmed by this script.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = ROOT.parent / "Projects" / "trajectory-os"
MAOS_DB = ROOT / "data" / "maos.db"


def run_step(name: str, command: list[str], cwd: Path, *, timeout: int = 120) -> dict[str, object]:
    started = datetime.now(UTC).isoformat()
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False, timeout=timeout
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = f"timed out after {timeout}s\n{(exc.stderr or '') if isinstance(exc.stderr, str) else ''}"
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "started_at": started,
        "exit_code": exit_code,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def workspace_paths() -> list[Path]:
    if not MAOS_DB.exists():
        return []
    conn = sqlite3.connect(f"file:{MAOS_DB}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT path FROM workspaces WHERE path != '' ORDER BY path").fetchall()
        paths = sorted({Path(row[0]).resolve() for row in rows if Path(row[0]).exists()}, key=str)
        # A broad container such as ~/Projects may be registered alongside
        # its child repositories. Index the children, not the container, or a
        # full pass becomes an unbounded recursive scan.
        return [path for path in paths if not any(
            other != path and other.is_relative_to(path) for other in paths
        )]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-index", action="store_true", help="Incrementally index every registered workspace")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    steps: list[dict[str, object]] = []
    steps.append(run_step("external-session-ingest", ["uv", "run", "python", "-c", "from pathlib import Path; from app.external_sessions import ingest; import json; print(json.dumps(ingest(Path('data'))))"], ROOT))
    steps.append(run_step("project-activity-audit", ["uv", "run", "python", "scripts/audit_project_activity.py"], ROOT))
    steps.append(run_step("live-confidence-adapters", ["uv", "run", "python", "-c", "from pathlib import Path; from app.confidence_adapters import run; import json; result=run(Path('.'), Path('data/confidence_adapters.json')); print(json.dumps(result))"], ROOT, timeout=90))
    steps.append(run_step(
        "workspace-scan",
        ["uv", "run", "app", "workspace", "scan", str(Path.home() / "Projects")],
        ROOT,
    ))
    steps.append(run_step("workspace-profile", ["uv", "run", "app", "workspace", "profile", "--no-llm"], ROOT))

    if args.full_index:
        for path in workspace_paths():
            steps.append(run_step(f"file-index:{path.name}", ["uv", "run", "app", "index", str(path)], ROOT))
            if (path / ".git").exists():
                steps.append(run_step(f"git-index:{path.name}", ["uv", "run", "app", "git", "index", str(path)], ROOT))

    steps.append(run_step("context-evaluate", ["uv", "run", "app", "context", "evaluate"], ROOT))
    steps.append(run_step("session-search-rebuild", [
        "uv", "run", "python", "-c",
        "from app.db import Database; from app.session_search import SessionSearch; "
        "from pathlib import Path; db=Database(str(Path('data/maos.db'))); "
        "print('indexed', SessionSearch(db).rebuild())",
    ], ROOT))
    steps.append(run_step("trajectory-ingest-dry-run", [
        "uv", "run", "trajectory", "ingest-maos", "--maos-db", str(MAOS_DB),
    ], TRAJECTORY_ROOT))
    steps.append(run_step("trajectory-stage-hf-artifact", [
        "uv", "run", "trajectory", "artifacts", "stage-external",
        "--location", "https://huggingface.co/davidnichols-ops/adaptive-operator-v4-gguf",
        "--description", "User-reported shipped adaptive-operator-v4-gguf artifact. Reported as a useful failure: TRASK was complete, but the result should be validated against reproducible inference checks before being treated as capability evidence.",
        "--project-name", "Local Operator Distillation",
    ], TRAJECTORY_ROOT))
    steps.append(run_step("trajectory-findings", ["uv", "run", "trajectory", "findings", "--format", "json"], TRAJECTORY_ROOT))
    steps.append(run_step("evidence-and-task-generation", ["uv", "run", "python", "scripts/reconcile_evidence.py"], ROOT))
    steps.append(run_step("task-handoff", ["uv", "run", "app", "task", "list"], ROOT))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "full_index": args.full_index,
        "steps": steps,
        "all_passed": all(step["exit_code"] == 0 for step in steps),
        "next_action": "Review MAOS task list after refresh; confirm or reject Trajectory candidates separately.",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Operating-context refresh: {'PASS' if report['all_passed'] else 'ATTENTION'}")
        for step in steps:
            print(f"  {'PASS' if step['exit_code'] == 0 else 'FAIL'} {step['name']}")
        print(report["next_action"])
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
