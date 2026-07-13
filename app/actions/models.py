"""Action models — schema for actions the AI can request.

Actions are the controlled interface between the LLM and the machine.
The LLM never executes anything directly. It produces an ActionRequest,
which the executor validates, checks permissions, and optionally runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """Schema for a registered action."""

    name: str
    description: str
    permissions: list[str] = field(default_factory=list)
    reversible: bool = True
    parameters: dict[str, str] = field(default_factory=dict)  # param name -> type hint


@dataclass
class ActionRequest:
    """A request from the LLM to execute an action."""

    action: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "target": self.target, "parameters": self.parameters}


@dataclass
class ActionResult:
    """The result of executing (or refusing) an action."""

    ok: bool
    action: str
    target: str
    result: str = ""
    error: str = ""
    approved: bool = False
    executed: bool = False
    reversible: bool = True
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "action": self.action, "target": self.target,
            "result": self.result, "error": self.error,
            "approved": self.approved, "executed": self.executed,
            "reversible": self.reversible, "duration_ms": self.duration_ms,
        }
