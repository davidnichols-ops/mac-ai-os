"""Tool registry + plugin loader.

Tools are registered by name. Plugins are discovered from entry points or
from a configurable directory (future). For now, built-ins are registered
on construction.
"""

from __future__ import annotations

from typing import Any

from app.log import get_logger
from app.tools.base import Tool, ToolResult
from app.tools.builtin import builtin_tools

log = get_logger(__name__)


class ToolRouter:
    """Routes a tool call by name to the registered Tool, after permission checks."""

    def __init__(self, tools: list[Tool] | None = None, *, granted_permissions: set[str] | None = None):
        self._tools: dict[str, Tool] = {}
        self._granted = granted_permissions or set()
        for t in tools or builtin_tools():
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            log.warning("tool.register.duplicate", name=tool.name)
        self._tools[tool.name] = tool
        log.info("tool.registered", name=tool.name, permissions=sorted(tool.permissions))

    def grant(self, permission: str) -> None:
        self._granted.add(permission)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def describe(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}")
        missing = tool.permissions - self._granted
        if missing:
            return ToolResult(ok=False, error=f"missing permissions: {sorted(missing)}")
        try:
            return tool.execute(args)
        except Exception as e:
            log.exception("tool.execute.failed", name=name)
            return ToolResult(ok=False, error=str(e))

    def health(self) -> bool:
        return len(self._tools) > 0
