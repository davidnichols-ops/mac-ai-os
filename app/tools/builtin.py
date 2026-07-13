"""Built-in permissioned tools.

These wrap risky operations (shell, files, git) so the planner never runs
raw commands. Each declares permissions the orchestrator can gate on.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command and return stdout/stderr. Use only when no specific tool fits."
    permissions = {"shell.exec"}

    def __init__(self, *, allowed: bool = True, timeout: float = 30.0):
        self._allowed = allowed
        self._timeout = timeout

    def execute(self, args: dict[str, Any]) -> ToolResult:
        if not self._allowed:
            return ToolResult(ok=False, error="shell.exec permission denied")
        cmd = args.get("command")
        if not cmd:
            return ToolResult(ok=False, error="missing 'command'")
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=self._timeout
            )
            return ToolResult(
                ok=proc.returncode == 0,
                output=proc.stdout,
                error=proc.stderr or None,
                metadata={"returncode": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"command timed out after {self._timeout}s")


class FileReadTool(Tool):
    name = "file_read"
    description = "Read a UTF-8 text file from disk."
    permissions = {"file.read"}

    def execute(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        if not path:
            return ToolResult(ok=False, error="missing 'path'")
        p = Path(path)
        if not p.is_file():
            return ToolResult(ok=False, error=f"not a file: {path}")
        try:
            return ToolResult(ok=True, output=p.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class FileWriteTool(Tool):
    name = "file_write"
    description = "Write text to a file on disk (overwrites)."
    permissions = {"file.write"}

    def execute(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content", "")
        if not path:
            return ToolResult(ok=False, error="missing 'path'")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(ok=True, output=str(p), metadata={"bytes": len(content)})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class GitTool(Tool):
    name = "git"
    description = "Run a git subcommand inside a repository."
    permissions = {"git.exec"}

    def __init__(self, *, timeout: float = 30.0):
        self._timeout = timeout

    def execute(self, args: dict[str, Any]) -> ToolResult:
        repo = args.get("repo", ".")
        subcmd = args.get("command")
        if not subcmd:
            return ToolResult(ok=False, error="missing git 'command'")
        try:
            proc = subprocess.run(
                f"git {subcmd}", shell=True, cwd=repo, capture_output=True,
                text=True, timeout=self._timeout,
            )
            return ToolResult(
                ok=proc.returncode == 0,
                output=proc.stdout,
                error=proc.stderr or None,
                metadata={"returncode": proc.returncode},
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


def builtin_tools() -> list[Tool]:
    return [ShellTool(), FileReadTool(), FileWriteTool(), GitTool()]
