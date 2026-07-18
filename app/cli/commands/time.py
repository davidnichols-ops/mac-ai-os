"""`maos time` — time awareness for AI agents."""

from __future__ import annotations

import json
import sys

import typer

from app.runtime import build_runtime
from app.time import build_time_packet, detect_session_start, format_time_packet, reading_profile

time_app = typer.Typer(
    name="time",
    help="Time awareness — current time, session duration, tool call stats.",
    no_args_is_help=False,
)


@time_app.callback(invoke_without_command=True)
def time_cmd(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current time and session duration information."""
    if ctx.invoked_subcommand is not None:
        return
    rt = build_runtime()
    packet = build_time_packet(rt.db)
    if json_output:
        typer.echo(json.dumps(packet, default=str, indent=2))
    else:
        typer.echo(format_time_packet(packet))


@time_app.command("hook")
def hook_cmd(
    file_path: str = typer.Option("", "--path", help="File path that was read"),
    lines: int = typer.Option(0, "--lines", help="Number of lines read"),
) -> None:
    """PostToolUse hook entry point — logs a read and returns compact time awareness.

    This is designed to be called by the Devin PostToolUse hook on every
    'read' tool call. It logs the read to the file_reads table and outputs
    a compact time awareness summary that gets injected back to the agent.

    Uses direct DB access (no runtime/models) for speed — completes in
    <100ms instead of the 2s+ that build_runtime() takes.

    Usage in hook config:
        command: "uv run --project ~/mac-ai-os app time hook"

    Or read stdin JSON (Devin hook format):
        echo '{"tool_input": {"file_path": "/foo.py"}}' | uv run app time hook
    """
    # If no --path given, try reading from stdin (Devin hook format).
    path = file_path
    if not path:
        try:
            data = json.load(sys.stdin)
            path = data.get("tool_input", {}).get("file_path", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    if not path:
        typer.echo(json.dumps({"context": "Read completed."}))
        return

    # Use direct DB access — no runtime/models needed. This is ~50x faster.
    import logging
    logging.disable(logging.CRITICAL)  # suppress all log output for speed
    from pathlib import Path

    from app.db import Database
    db_path = Path.home() / "mac-ai-os" / "data" / "maos.db"
    db = Database(str(db_path))

    # Log the read.
    db.log_file_read(path=path, lines=lines)

    # Get compact time + reading awareness.
    session = detect_session_start(db)
    reading = reading_profile(db)

    elapsed = session["elapsed_compact"]
    calls = session["tool_call_count"]
    files_read = reading["files_read"]
    unique = reading["unique_files"]
    re_read_pct = int(reading["re_read_ratio"] * 100)

    # Build the context string that gets injected back to the agent.
    parts = [f"⏱ {elapsed}", f"{calls} calls", f"{files_read} files read ({unique} unique, {re_read_pct}% re-reads)"]
    if session["is_active"]:
        parts.append(f"active {session.get('idle_compact', '0s')}")
    else:
        parts.append(f"idle {session.get('idle_compact', '0s')}")

    context = "[MAOS] " + " · ".join(parts)

    # Add warning if going in circles.
    if reading["re_read_ratio"] > 0.5 and files_read > 5:
        context += (
            f"\n[MAOS WARNING] Re-reading {re_read_pct}% of files — "
            f"you may be going in circles. Consider stepping back."
        )

    # Output as JSON for the hook system.
    typer.echo(json.dumps({"context": context}))
