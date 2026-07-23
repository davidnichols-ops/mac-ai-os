"""Failure, retry, and compensation semantics.

Phase 6: This module formalizes the failure handling model for workflows:

  - CompensationChain: When a step fails, all previously completed steps
    can be compensated in reverse order (saga pattern).
  - FailurePolicy: Declarative failure handling at the workflow level
    (max_retries, timeout, on_cascade_fail, compensation_strategy).
  - DeadLetterQueue: Permanently failed workflows are stored for manual
    inspection and potential re-processing.
  - SagaCoordinator: Orchestrates the saga pattern — forward execution
    with backward compensation on failure.

The saga pattern:
  1. Execute steps forward (T1, T2, T3, ...)
  2. If Ti fails and cannot be retried:
     a. Run compensation for Ti (if it has one)
     b. Run compensation for Ti-1, Ti-2, ..., T1 in reverse order
     c. Mark the workflow as COMPENSATED (not just FAILED)
  3. If all steps succeed, mark as COMPLETED
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.log import get_logger

log = get_logger(__name__)


class CompensationStrategy(StrEnum):
    """How to handle compensation when a step fails.

    NONE:        No compensation — just mark as failed.
    STEP_ONLY:   Compensate only the failed step.
    SAGA:        Compensate the failed step AND all previously completed steps (reverse order).
    """

    NONE = "none"
    STEP_ONLY = "step_only"
    SAGA = "saga"

    def __str__(self) -> str:
        return self.value


@dataclass
class FailurePolicy:
    """Declarative failure handling policy for a workflow.

    This is set at the workflow level and applies to all steps unless
    a step overrides it with its own on_failure behavior.
    """

    strategy: CompensationStrategy = CompensationStrategy.NONE
    max_workflow_retries: int = 0  # retry the entire workflow on failure
    workflow_retry_delay_ms: int = 5000
    timeout_seconds: float = 0.0  # 0 = no workflow-level timeout
    dead_letter_on_failure: bool = True  # send to DLQ on permanent failure

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "max_workflow_retries": self.max_workflow_retries,
            "workflow_retry_delay_ms": self.workflow_retry_delay_ms,
            "timeout_seconds": self.timeout_seconds,
            "dead_letter_on_failure": self.dead_letter_on_failure,
        }


@dataclass
class CompensationRecord:
    """Record of a compensation action that was executed."""

    step_name: str
    step_id: str
    success: bool
    error: str = ""
    duration_ms: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "step_id": self.step_id,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "data": self.data,
        }


class SagaCoordinator:
    """Coordinates the saga pattern for workflow compensation.

    When a step fails, the SagaCoordinator:
    1. Runs the failed step's compensation handler (if any)
    2. Runs compensation for all previously completed steps in reverse order
    3. Records each compensation result
    4. Returns a list of CompensationRecords
    """

    def __init__(self, db: Any | None = None):
        self.db = db

    def compensate(
        self,
        failed_step: Any,
        completed_steps: list[tuple[Any, Any]],
        ctx: dict[str, Any],
    ) -> list[CompensationRecord]:
        """Run compensation for the failed step and all completed steps.

        Args:
            failed_step: The WorkflowStep that failed.
            completed_steps: List of (WorkflowStep, StepResult) tuples for
                steps that completed successfully, in execution order.
            ctx: The workflow context.

        Returns:
            List of CompensationRecords in execution order (failed step first,
            then completed steps in reverse order).
        """
        records: list[CompensationRecord] = []

        # 1. Compensate the failed step (if it has a compensation handler).
        if failed_step.compensation:
            record = self._run_compensation(failed_step, ctx, None)
            records.append(record)

        # 2. Compensate all completed steps in reverse order.
        for step, step_result in reversed(completed_steps):
            if step.compensation:
                record = self._run_compensation(step, ctx, step_result)
                records.append(record)

        log.info(
            "saga.compensation_complete",
            failed_step=failed_step.name,
            compensations_run=len(records),
            successful=sum(1 for r in records if r.success),
        )

        return records

    def _run_compensation(
        self, step: Any, ctx: dict[str, Any], step_result: Any | None,
    ) -> CompensationRecord:
        """Run a single step's compensation handler."""
        import time
        t0 = time.time()

        try:
            if step_result is not None:
                data = step.compensation(ctx, step_result)
            else:
                # Failed step — pass a dummy StepResult.
                from app.workflows.models import StepResult
                dummy = StepResult(step_name=step.name, ok=False, error="failed")
                data = step.compensation(ctx, dummy)

            duration_ms = int((time.time() - t0) * 1000)
            record = CompensationRecord(
                step_name=step.name,
                step_id=step.step_id,
                success=True,
                duration_ms=duration_ms,
                data=data if isinstance(data, dict) else {"result": str(data)},
            )
            log.info("saga.compensated", step=step.name, step_id=step.step_id)
            return record

        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            log.warning("saga.compensation_failed", step=step.name, error=str(e))
            return CompensationRecord(
                step_name=step.name,
                step_id=step.step_id,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )


class DeadLetterQueue:
    """Dead letter queue for permanently failed workflows.

    Workflows that fail and cannot be retried are stored here for
    manual inspection and potential re-processing.
    """

    def __init__(self, db: Any):
        self.db = db

    def enqueue(
        self, run_id: int, workflow_name: str, failure_reason: str,
        compensation_records: list[CompensationRecord] | None = None,
        context_snapshot: dict[str, Any] | None = None,
    ) -> int:
        """Add a failed workflow to the dead letter queue.

        Returns the DLQ entry ID.
        """
        from datetime import UTC, datetime
        comp_json = json.dumps([r.to_dict() for r in (compensation_records or [])])
        ctx_json = json.dumps(context_snapshot or {}, default=str)

        cur = self.db.execute(
            """INSERT INTO workflow_dead_letter_queue
               (run_id, workflow_name, failure_reason, compensation_records,
                context_snapshot, enqueued_at, status)
               VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
            (run_id, workflow_name, failure_reason, comp_json,
             ctx_json, datetime.now(UTC).isoformat()),
        )
        log.info("dlq.enqueued", run_id=run_id, workflow=workflow_name)
        return cur.lastrowid

    def list_entries(self, status: str = "pending") -> list[Any]:
        """List DLQ entries, optionally filtered by status."""
        return self.db.query(
            "SELECT * FROM workflow_dead_letter_queue WHERE status = ? ORDER BY enqueued_at DESC",
            (status,),
        )

    def get_entry(self, dlq_id: int) -> Any | None:
        """Get a specific DLQ entry."""
        return self.db.query_one(
            "SELECT * FROM workflow_dead_letter_queue WHERE id = ?",
            (dlq_id,),
        )

    def mark_inspected(self, dlq_id: int, notes: str = "") -> None:
        """Mark a DLQ entry as inspected."""
        from datetime import UTC, datetime
        self.db.execute(
            "UPDATE workflow_dead_letter_queue SET status = 'inspected', notes = ?, inspected_at = ? WHERE id = ?",
            (notes, datetime.now(UTC).isoformat(), dlq_id),
        )

    def mark_reprocessed(self, dlq_id: int, new_run_id: int) -> None:
        """Mark a DLQ entry as reprocessed with a new run ID."""
        from datetime import UTC, datetime
        self.db.execute(
            "UPDATE workflow_dead_letter_queue SET status = 'reprocessed', new_run_id = ?, reprocessed_at = ? WHERE id = ?",
            (new_run_id, datetime.now(UTC).isoformat(), dlq_id),
        )

    def count(self, status: str = "pending") -> int:
        """Count DLQ entries by status."""
        row = self.db.query_one(
            "SELECT COUNT(*) as n FROM workflow_dead_letter_queue WHERE status = ?",
            (status,),
        )
        return row["n"] if row else 0
