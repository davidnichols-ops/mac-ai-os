"""Workflow lifecycle states and transitions.

Defines the full state machine for durable workflows:

    pending → running → completed
                    ↘    ↗
                     failed
                    ↗
    crashed ──→ running (resumed)
        ↑
    running (stale detected on startup)

    running → paused → running (resumed)
    running → cancelled
    running → timed_out
    failed → compensating → compensated

States:
    pending:       Workflow created but not yet started.
    running:       Currently executing steps.
    paused:        Intentionally paused (e.g., awaiting approval).
    completed:     All steps finished successfully.
    failed:        A required step failed and the workflow stopped.
    cancelled:     Cancelled by user or system.
    crashed:       Process died during execution (detected on restart).
    timed_out:     A step or the workflow exceeded its time limit.
    compensating:  Running compensation handlers after failure.
    compensated:   Compensation complete.

Transitions are validated. Attempting an illegal transition raises
InvalidTransitionError.
"""

from __future__ import annotations

from enum import StrEnum


class WorkflowStatus(StrEnum):
    """All possible workflow states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"
    TIMED_OUT = "timed_out"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

    def __str__(self) -> str:
        return self.value


# Valid state transitions: from_state → {set of allowed to_states}
_VALID_TRANSITIONS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.PENDING: frozenset({
        WorkflowStatus.RUNNING,
        WorkflowStatus.CANCELLED,
    }),
    WorkflowStatus.RUNNING: frozenset({
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.PAUSED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.TIMED_OUT,
        WorkflowStatus.CRASHED,
    }),
    WorkflowStatus.PAUSED: frozenset({
        WorkflowStatus.RUNNING,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.TIMED_OUT,
    }),
    WorkflowStatus.FAILED: frozenset({
        WorkflowStatus.COMPENSATING,
        WorkflowStatus.PENDING,  # re-queue for retry
    }),
    WorkflowStatus.COMPENSATING: frozenset({
        WorkflowStatus.COMPENSATED,
        WorkflowStatus.FAILED,  # compensation itself failed
    }),
    WorkflowStatus.CRASHED: frozenset({
        WorkflowStatus.RUNNING,  # resume from checkpoint
        WorkflowStatus.PENDING,  # re-queue
        WorkflowStatus.CANCELLED,
        WorkflowStatus.FAILED,  # give up
    }),
    WorkflowStatus.TIMED_OUT: frozenset({
        WorkflowStatus.PENDING,  # retry
        WorkflowStatus.CANCELLED,
        WorkflowStatus.FAILED,
    }),
    # Terminal states — no transitions out.
    WorkflowStatus.COMPLETED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
    WorkflowStatus.COMPENSATED: frozenset(),
}


class InvalidTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, from_status: WorkflowStatus, to_status: WorkflowStatus):
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid workflow transition: {from_status.value} → {to_status.value}"
        )


def validate_transition(from_status: WorkflowStatus, to_status: WorkflowStatus) -> None:
    """Raise InvalidTransitionError if the transition is not allowed."""
    allowed = _VALID_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise InvalidTransitionError(from_status, to_status)


def can_transition(from_status: WorkflowStatus, to_status: WorkflowStatus) -> bool:
    """Check if a transition is allowed without raising."""
    return to_status in _VALID_TRANSITIONS.get(from_status, frozenset())


def is_terminal(status: WorkflowStatus) -> bool:
    """Check if a status is terminal (no transitions out)."""
    return len(_VALID_TRANSITIONS.get(status, frozenset())) == 0


def is_active(status: WorkflowStatus) -> bool:
    """Check if a status indicates the workflow is actively consuming resources."""
    return status in (WorkflowStatus.RUNNING, WorkflowStatus.COMPENSATING)
