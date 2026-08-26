"""Built-in permissioned tools.

These wrap risky operations (shell, files, git) so the planner never runs
raw commands. Each declares permissions the orchestrator can gate on.
Every tool has a proper JSON schema so the model knows what parameters to use.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult


def _expand_path(path: str) -> Path:
    """Expand ~ and make relative paths absolute."""
    return Path(path).expanduser()


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command and return stdout, stderr, and exit code. Use for system operations, file inspection, running tests, etc."
    permissions = {"shell.exec"}

    def __init__(self, *, allowed: bool = True, timeout: float = 30.0):
        self._allowed = allowed
        self._timeout = timeout

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                },
            },
            "required": ["command"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        if not self._allowed:
            return ToolResult(ok=False, error="shell.exec permission denied")
        cmd = args.get("command")
        if not cmd:
            return ToolResult(ok=False, error="missing 'command'")
        timeout = args.get("timeout", self._timeout)
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            return ToolResult(
                ok=proc.returncode == 0,
                output=output,
                error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
                metadata={"returncode": proc.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"command timed out after {timeout}s")


class FileReadTool(Tool):
    name = "file_read"
    description = "Read the contents of a text file. Returns the file content as a string."
    permissions = {"file.read"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to workspace)",
                },
            },
            "required": ["path"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        if not path:
            return ToolResult(ok=False, error="missing 'path'")
        p = _expand_path(path)
        if not p.is_file():
            return ToolResult(ok=False, error=f"not a file: {path}")
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            # Truncate very large files
            if len(content) > 50000:
                content = content[:50000] + f"\n... (truncated, {len(content)} total chars)"
            return ToolResult(ok=True, output=content)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class FileWriteTool(Tool):
    name = "file_write"
    description = "Write text content to a file. Creates or overwrites the file."
    permissions = {"file.write"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content", "")
        if not path:
            return ToolResult(ok=False, error="missing 'path'")
        try:
            p = _expand_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(ok=True, output=f"wrote {len(content)} chars to {p}", metadata={"bytes": len(content)})
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class FileEditTool(Tool):
    """Surgical edit: find and replace a unique string in a file."""
    name = "file_edit"
    description = "Replace a specific string in a file with a new string. The old_string must appear exactly once."
    permissions = {"file.write"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The string to find in the file (must be unique)",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string",
                },
            },
            "required": ["path", "old_string", "new_string"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path")
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        if not path:
            return ToolResult(ok=False, error="missing 'path'")
        if not old:
            return ToolResult(ok=False, error="missing 'old_string'")
        try:
            p = _expand_path(path)
            if not p.is_file():
                return ToolResult(ok=False, error=f"not a file: {path}")
            content = p.read_text(encoding="utf-8", errors="ignore")
            count = content.count(old)
            if count == 0:
                return ToolResult(ok=False, error="old_string not found in file")
            if count > 1:
                return ToolResult(ok=False, error=f"old_string found {count} times, need unique match")
            content = content.replace(old, new, 1)
            p.write_text(content, encoding="utf-8")
            return ToolResult(ok=True, output=f"edited {path}")
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class FileListTool(Tool):
    """List files in a directory."""
    name = "file_list"
    description = "List files and directories in a given path. Returns relative paths."
    permissions = {"file.read"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        try:
            p = _expand_path(path)
            if not p.is_dir():
                return ToolResult(ok=False, error=f"not a directory: {path}")
            entries = sorted([str(x.name) + ("/" if x.is_dir() else "") for x in p.iterdir()])
            return ToolResult(ok=True, output="\n".join(entries[:500]))
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class GitTool(Tool):
    name = "git"
    description = "Run a git subcommand (status, diff, log, branch, etc.) inside a repository."
    permissions = {"git.exec"}

    def __init__(self, *, timeout: float = 30.0):
        self._timeout = timeout

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The git subcommand to run (e.g. 'status', 'log --oneline -5', 'diff')",
                },
                "repo": {
                    "type": "string",
                    "description": "Path to the git repository (default: current directory)",
                },
            },
            "required": ["command"],
        }

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
            output = proc.stdout
            if proc.stderr:
                output += f"\n[stderr]\n{proc.stderr}"
            return ToolResult(
                ok=proc.returncode == 0,
                output=output,
                error=None if proc.returncode == 0 else f"exit code {proc.returncode}",
                metadata={"returncode": proc.returncode},
            )
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class GrepTool(Tool):
    """Search file contents using ripgrep (fast, respects gitignore)."""
    name = "grep"
    description = "Search for a pattern in file contents using ripgrep. Returns matching lines with file paths and line numbers. Supports regex patterns."
    permissions = {"file.read"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}')",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 100)",
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context to show around each match (default: 0)",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern")
        if not pattern:
            return ToolResult(ok=False, error="missing 'pattern'")
        path = args.get("path", ".")
        glob_pat = args.get("glob")
        case_insensitive = args.get("case_insensitive", False)
        max_results = args.get("max_results", 100)
        context = args.get("context", 0)

        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if case_insensitive:
            cmd.append("-i")
        if context > 0:
            cmd.extend(["-C", str(context)])
        if glob_pat:
            cmd.extend(["-g", glob_pat])
        cmd.extend(["--max-count", str(max_results)])
        cmd.extend([pattern, path])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = proc.stdout
            if not output and proc.stderr:
                return ToolResult(ok=False, error=proc.stderr.strip())
            return ToolResult(ok=True, output=output or "(no matches)")
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="search timed out after 30s")
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class FindFileTool(Tool):
    """Find files by name pattern using glob matching."""
    name = "find_file"
    description = "Find files by name pattern. Supports glob patterns like '*.py', '**/*.ts', 'test_*'. Returns matching file paths."
    permissions = {"file.read"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match file names (e.g. '*.py', '**/*.ts', 'test_*')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern")
        if not pattern:
            return ToolResult(ok=False, error="missing 'pattern'")
        path = args.get("path", ".")

        try:
            base = _expand_path(path)
            results = sorted(str(p) for p in base.glob(pattern) if p.is_file())
            if not results:
                return ToolResult(ok=True, output="(no files found)")
            return ToolResult(ok=True, output="\n".join(results[:500]))
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key needed)."""
    name = "web_search"
    description = "Search the web for information. Returns relevant web page titles, URLs, and summaries."
    permissions = {"web.search"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 20)",
                },
            },
            "required": ["query"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query")
        if not query:
            return ToolResult(ok=False, error="missing 'query'")
        num_results = min(args.get("num_results", 5), 20)

        try:
            import re as _re
            import urllib.parse
            import urllib.request

            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Parse results from HTML
            results = []
            # Extract result titles and URLs
            for match in _re.finditer(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, _re.DOTALL):
                raw_url = match.group(1)
                title = _re.sub(r"<[^>]+>", "", match.group(2)).strip()
                # DuckDuckGo redirects — extract actual URL
                url_match = _re.search(r"uddg=([^&]+)", raw_url)
                actual_url = urllib.parse.unquote(url_match.group(1)) if url_match else raw_url
                results.append(f"{title}\n  {actual_url}")
                if len(results) >= num_results:
                    break

            if not results:
                return ToolResult(ok=True, output="(no results found)")
            return ToolResult(ok=True, output="\n\n".join(results))
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class WebFetchTool(Tool):
    """Fetch a web page and return its content as text."""
    name = "web_fetch"
    description = "Fetch a web page and return its content as readable text. Useful for reading documentation, articles, or API responses."
    permissions = {"web.fetch"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
            },
            "required": ["url"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        url = args.get("url")
        if not url:
            return ToolResult(ok=False, error="missing 'url'")

        try:
            import re as _re
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Strip HTML tags for readable text
            # Remove script and style blocks
            text = _re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
            # Remove tags
            text = _re.sub(r"<[^>]+>", " ", text)
            # Decode entities
            text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
            # Collapse whitespace
            text = _re.sub(r"\s+", " ", text).strip()

            if len(text) > 20000:
                text = text[:20000] + "\n... (truncated)"
            return ToolResult(ok=True, output=text)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class TodoWriteTool(Tool):
    """Manage a task list for tracking progress on multi-step work."""
    name = "todo_write"
    description = "Create and manage a structured task list. Track progress on multi-step tasks. Each task has a status: pending, in_progress, or completed."
    permissions = {"todo.write"}

    def __init__(self):
        self._todos: list[dict[str, str]] = []

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "List of task objects with 'content' and 'status' fields",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The task description"},
                            "status": {"type": "string", "description": "Task status: pending, in_progress, or completed"},
                        },
                    },
                },
            },
            "required": ["todos"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        todos = args.get("todos")
        if not todos or not isinstance(todos, list):
            return ToolResult(ok=False, error="missing or invalid 'todos' list")

        self._todos = []
        for t in todos:
            content = t.get("content", "")
            status = t.get("status", "pending")
            self._todos.append({"content": content, "status": status})

        # Format output
        lines = []
        for i, t in enumerate(self._todos, 1):
            status_icon = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}.get(t["status"], "[ ]")
            lines.append(f"{i}. {status_icon} {t['content']}")

        return ToolResult(ok=True, output="\n".join(lines) if lines else "(empty todo list)")

    def get_todos(self) -> list[dict[str, str]]:
        return self._todos


class AskUserTool(Tool):
    """Ask the user a question and wait for their response."""
    name = "ask_user"
    description = "Ask the user a question when you need clarification or a decision. Presents options and collects their answer."
    permissions = {"user.interact"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
                "options": {
                    "type": "array",
                    "description": "List of options for the user to choose from",
                    "items": {"type": "string"},
                },
            },
            "required": ["question"],
        }

    def __init__(self, *, input_callback=None):
        self._input_callback = input_callback

    def execute(self, args: dict[str, Any]) -> ToolResult:
        question = args.get("question")
        if not question:
            return ToolResult(ok=False, error="missing 'question'")
        options = args.get("options", [])

        if self._input_callback:
            answer = self._input_callback(question, options)
        else:
            # Default: read from stdin
            print(f"\n[Question] {question}")
            if options:
                for i, opt in enumerate(options, 1):
                    print(f"  {i}. {opt}")
                print(f"  {len(options)+1}. Other (type your own)")
            try:
                answer = input("> ").strip()
                if options and answer.isdigit():
                    idx = int(answer) - 1
                    if 0 <= idx < len(options):
                        answer = options[idx]
            except (EOFError, KeyboardInterrupt):
                answer = "(no response)"

        return ToolResult(ok=True, output=f"User answered: {answer}")


def builtin_tools() -> list[Tool]:
    return [
        ShellTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        FileListTool(),
        GrepTool(),
        FindFileTool(),
        GitTool(),
        WebSearchTool(),
        WebFetchTool(),
        TodoWriteTool(),
        AskUserTool(),
    ]
