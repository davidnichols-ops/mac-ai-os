"""Workflow models — schema for multi-step workflows.

A workflow is a sequence of steps that the system executes autonomously.
Each step can:
  - Gather data (query DB, run an agent, capture a snapshot)
  - Execute an action (through the action executor)
  - Verify the result (check that the action had its intended effect)
  - Decide whether to continue or stop

Workflows are NOT agents. They orchestrate agents and actions.

Phase 1 additions:
  - Workflow has a stable `workflow_id` (UUID) and `version` (int).
  - WorkflowStep has a stable `step_id` (deterministic hash of workflow_id + name + version).
  - These identities are persisted to the database, enabling resumability
    and audit trails.

Phase 2 additions:
  - WorkflowStep has retry policy (max_retries, retry_delay_ms) with exponential backoff.
  - WorkflowStep has per-step timeout (timeout_seconds).
  - WorkflowStep has per-step permission requirements (permissions: list[str]).
  - WorkflowStep has failure behavior declaration (on_failure: skip|abort|compensate).
  - WorkflowStep has side effect declaration (side_effects: list[str]).
  - StepResult tracks all attempts (not just the final one).
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass
class VerificationResult:
    """The result of verifying that an action had its intended effect."""

    success: bool
    evidence: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "evidence": self.evidence, "error": self.error}


class FailureBehavior(StrEnum):
    """What happens when a step fails.

    SKIP:       Continue to the next step (same as required=False).
    ABORT:      Stop the entire workflow (same as required=True).
    COMPENSATE: Run the step's compensation handler, then continue.
    """

    SKIP = "skip"
    ABORT = "abort"
    COMPENSATE = "compensate"

    def __str__(self) -> str:
        return self.value


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Phase 2 additions:
      - retry: max_retries + retry_delay_ms with exponential backoff
      - timeout: per-step timeout in seconds (0 = no timeout)
      - permissions: per-step permission requirements
      - on_failure: failure behavior (skip/abort/compensate)
      - side_effects: declared external side effects
      - compensation: handler to run on failure (if on_failure=COMPENSATE)
    """

    name: str
    description: str
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    verify: Callable[[dict[str, Any]], VerificationResult] | None = None
    required: bool = True  # if False, failure doesn't stop the workflow
    action: str = ""  # optional action name to execute
    step_id: str = ""  # stable identity (set by parent Workflow)

    # Phase 2: retry policy
    max_retries: int = 0  # 0 = no retry, 1 = one retry after first failure, etc.
    retry_delay_ms: int = 1000  # base delay; doubles each retry (exponential backoff)

    # Phase 2: timeout
    timeout_seconds: float = 0.0  # 0 = no timeout

    # Phase 2: per-step permissions
    permissions: list[str] = field(default_factory=list)

    # Phase 2: failure behavior
    on_failure: FailureBehavior = FailureBehavior.ABORT
    compensation: Callable[[dict[str, Any], StepResult], dict[str, Any]] | None = None

    # Phase 2: side effect declaration
    side_effects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "required": str(self.required), "action": self.action,
            "step_id": self.step_id,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
            "timeout_seconds": self.timeout_seconds,
            "permissions": self.permissions,
            "on_failure": self.on_failure.value,
            "side_effects": self.side_effects,
        }


@dataclass
class StepResult:
    """The result of executing a single workflow step.

    Phase 2: Now tracks all attempts (for retry visibility) and
    whether compensation was run.
    """

    step_name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    verification: VerificationResult | None = None
    error: str = ""
    duration_ms: int = 0
    step_id: str = ""
    attempt: int = 1  # which attempt succeeded (1-based)
    attempts: list[dict[str, Any]] = field(default_factory=list)  # all attempts
    compensated: bool = False  # True if compensation handler was run
    timed_out: bool = False  # True if the step timed out

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name, "ok": self.ok, "data": self.data,
            "verification": self.verification.to_dict() if self.verification else None,
            "error": self.error, "duration_ms": self.duration_ms,
            "step_id": self.step_id, "attempt": self.attempt,
            "attempts": self.attempts,
            "compensated": self.compensated,
            "timed_out": self.timed_out,
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
    workflow_id: str = ""
    status: str = ""  # terminal status string (completed, failed, etc.)

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
            "workflow_id": self.workflow_id, "status": self.status,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
        }


@dataclass
class Workflow:
    """A named sequence of steps with a goal.

    Phase 1: Workflows now have a stable `workflow_id` (UUID) and `version`.
    Steps get deterministic `step_id`s derived from workflow_id + step name + version.
    """

    name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1

    def add_step(self, step: WorkflowStep) -> None:
        self.steps.append(step)
        self._assign_step_id(step)

    def _assign_step_id(self, step: WorkflowStep) -> None:
        """Assign a deterministic step_id if not already set."""
        if not step.step_id:
            raw = f"{self.workflow_id}:{self.version}:{step.name}"
            step.step_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def __post_init__(self) -> None:
        """Assign step_ids to any steps added via the constructor."""
        for step in self.steps:
            self._assign_step_id(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "workflow_id": self.workflow_id, "version": self.version,
        }
