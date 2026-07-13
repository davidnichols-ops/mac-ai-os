"""Workflow models — schema for multi-step workflows.

A workflow is a sequence of steps that the system executes autonomously.
Each step can:
  - Gather data (query DB, run an agent, capture a snapshot)
  - Execute an action (through the action executor)
  - Verify the result (check that the action had the intended effect)
  - Decide whether to continue or stop

Workflows are NOT agents. They orchestrate agents and actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationResult:
    """The result of verifying that an action had its intended effect."""

    success: bool
    evidence: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "evidence": self.evidence, "error": self.error}


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    name: str
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    verify: Callable[[dict[str, Any]], VerificationResult] | None = None
    required: bool = True  # if False, failure doesn't stop the workflow
    action: str = ""  # optional action name to execute

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description,
                "required": str(self.required), "action": self.action}


@dataclass
class StepResult:
    """The result of executing a single workflow step."""

    step_name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    verification: VerificationResult | None = None
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name, "ok": self.ok, "data": self.data,
            "verification": self.verification.to_dict() if self.verification else None,
            "error": self.error, "duration_ms": self.duration_ms,
        }


@dataclass
class WorkflowResult:
    """The result of running an entire workflow."""

    workflow_name: str
    ok: bool
    steps: list[StepResult] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    duration_ms: int = 0
    run_id: int | None = None

    @property
    def steps_completed(self) -> int:
        return sum(1 for s in self.steps if s.ok)

    @property
    def steps_failed(self) -> int:
        return sum(1 for s in self.steps if not s.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name, "ok": self.ok,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary, "error": self.error,
            "duration_ms": self.duration_ms, "run_id": self.run_id,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
        }


@dataclass
class Workflow:
    """A named sequence of steps with a goal."""

    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: WorkflowStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
        }
