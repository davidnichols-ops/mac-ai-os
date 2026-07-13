"""Tool framework.

Each tool is a plugin. The planner never executes shell commands directly —
everything goes through permissioned tools that declare a JSON schema and a
permission set.

    tool.execute(args)    -> Any
    tool.schema()         -> dict   (JSON schema for the planner)
    tool.permissions()    -> set[str]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class Tool(ABC):
    name: str = "base"
    description: str = ""
    permissions: set[str] = field(default_factory=set)

    @abstractmethod
    def execute(self, args: dict[str, Any]) -> ToolResult:
        ...

    def schema(self) -> dict[str, Any]:
        """JSON schema describing the args this tool accepts."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {},
            "required": [],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permissions": sorted(self.permissions),
        }
