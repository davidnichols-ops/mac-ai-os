"""Shell history observer — imports and monitors shell command history.

Parses ~/.zsh_history (or ~/.bash_history) and stores commands in the
shell_commands table. Supports incremental imports (only new commands since
last import).

zsh history format (extended):
  : <timestamp>:<duration>;<command>

bash history format:
  <command>  (no timestamps in default config)

For bash, we use the file modification time as an approximation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)

# zsh extended history: : 1234567890:0;command
_ZSH_ENTRY_RE = re.compile(r"^: (\d+):(\d+);(.*)$", re.DOTALL)


def parse_zsh_history(path: Path) -> list[dict[str, Any]]:
    """Parse ~/.zsh_history into a list of {timestamp, command} dicts.

    Handles both extended format (: timestamp:duration;command) and plain
    format (just commands, no timestamps). For plain format, uses file mtime.
    """
    if not path.exists():
        return []

    commands: list[dict[str, Any]] = []
    try:
        content = path.read_text(errors="replace")
    except (PermissionError, OSError) as e:
        log.warning("shell.history.read_failed", path=str(path), error=str(e))
        return []

    lines = content.split("\n")

    # Check if the first non-empty line has the extended format.
    has_extended = False
    for line in lines:
        if line.strip():
            has_extended = bool(_ZSH_ENTRY_RE.match(line))
            break

    if not has_extended:
        # Plain format: each line is a command, no timestamps.
        # Use file mtime as the timestamp for all commands.
        mtime = str(int(path.stat().st_mtime))
        for line in lines:
            line = line.strip()
            if line:
                commands.append({"timestamp": mtime, "command": line})
        return commands

    # Extended format: : timestamp:duration;command
    current_ts: str | None = None
    current_cmd: list[str] = []

    for line in lines:
        match = _ZSH_ENTRY_RE.match(line)
        if match:
            # Save previous command if any.
            if current_ts and current_cmd:
                commands.append({
                    "timestamp": current_ts,
                    "command": "\n".join(current_cmd),
                })
            current_ts = match.group(1)
            current_cmd = [match.group(3)]
        elif current_cmd:
            # Continuation of a multi-line command.
            current_cmd.append(line)

    # Don't forget the last command.
    if current_ts and current_cmd:
        commands.append({
            "timestamp": current_ts,
            "command": "\n".join(current_cmd).rstrip(),
        })

    return commands


def parse_bash_history(path: Path) -> list[dict[str, Any]]:
    """Parse ~/.bash_history (no timestamps by default)."""
    if not path.exists():
        return []

    try:
        content = path.read_text(errors="replace")
    except (PermissionError, OSError):
        return []

    # Use file mtime as approximate timestamp for all commands.
    mtime = int(path.stat().st_mtime)
    commands: list[dict[str, Any]] = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            commands.append({"timestamp": str(mtime), "command": line})
    return commands


def import_shell_history(
    db: Database,
    *,
    shell: str = "auto",
    history_path: str | None = None,
) -> int:
    """Import shell history into the database. Returns number of new commands."""
    if history_path:
        path = Path(history_path).expanduser()
    elif shell == "zsh" or shell == "auto":
        path = Path.home() / ".zsh_history"
        if not path.exists() and shell == "auto":
            path = Path.home() / ".bash_history"
            shell = "bash"
    elif shell == "bash":
        path = Path.home() / ".bash_history"
    else:
        log.warning("shell.history.unknown_shell", shell=shell)
        return 0

    if not path.exists():
        log.info("shell.history.no_file", path=str(path))
        return 0

    if shell == "bash" or path.name == ".bash_history":
        commands = parse_bash_history(path)
    else:
        commands = parse_zsh_history(path)

    if not commands:
        return 0

    # Get the latest imported timestamp to skip already-imported commands.
    existing_count = db.shell_command_count()
    latest_existing = None
    if existing_count > 0:
        rows = db.query(
            "SELECT timestamp FROM shell_commands ORDER BY timestamp DESC LIMIT 1"
        )
        if rows:
            latest_existing = rows[0]["timestamp"]

    # Import only new commands.
    new_count = 0
    for cmd in commands:
        if latest_existing and cmd["timestamp"] <= latest_existing:
            continue
        # Try to detect the working directory (not available from history alone).
        db.insert_shell_command(
            cmd["timestamp"],
            cmd["command"],
            directory="",
        )
        new_count += 1

    log.info("shell.history.imported", total=len(commands), new=new_count, path=str(path))
    return new_count


def search_history(db: Database, pattern: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search shell command history by pattern."""
    rows = db.search_shell_commands(pattern, limit=limit)
    return [dict(r) for r in rows]
