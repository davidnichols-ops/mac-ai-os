"""Workflow engine — executes multi-step workflows with verification.

The engine:
1. Creates a workflow_run record in the database
2. Executes each step in order
3. Runs verification after each step (if provided)
4. Handles failures: required steps stop the workflow, optional steps continue
5. Records the final result and summary

The engine is deliberately simple. It does not do parallel execution,
branching, or loops. Those are future capabilities. The current goal is
reliable sequential execution with verification.
"""

from __future__ import annotations

import time
from typing import Any

from app.db import Database
from app.events.bus import EventBus
from app.events.models import Event
from app.log import get_logger
from app.workflows.models import (
    StepResult,
    Workflow,
    WorkflowResult,
)

log = get_logger(__name__)

WORKFLOW_STARTED = "WORKFLOW_STARTED"
WORKFLOW_STEP_COMPLETED = "WORKFLOW_STEP_COMPLETED"
WORKFLOW_STEP_FAILED = "WORKFLOW_STEP_FAILED"
WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"


class WorkflowEngine:
    """Executes workflows step by step with verification."""

    def __init__(self, db: Database, bus: EventBus | None = None):
        self.db = db
        self.bus = bus

    def run(self, workflow: Workflow, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute a workflow and return the result."""
        ctx = {**workflow.context, **(context or {})}
        ctx["db"] = self.db

        start = time.time()
        run_id = self.db.start_workflow_run(workflow.name, len(workflow.steps))
        log.info("workflow.started", name=workflow.name, steps=len(workflow.steps), run_id=run_id)

        if self.bus:
            self.bus.publish(Event(WORKFLOW_STARTED, {
                "workflow": workflow.name, "run_id": run_id,
                "steps": len(workflow.steps),
            }))

        result = WorkflowResult(workflow_name=workflow.name, ok=True, run_id=run_id)
        step_results: list[StepResult] = []

        for i, step in enumerate(workflow.steps):
            step_start = time.time()
            step_result = StepResult(step_name=step.name, ok=False)

            try:
                # Execute the step handler.
                data = step.handler(ctx)
                step_result.ok = bool(data.get("ok", True))
                step_result.data = data

                # Merge step data into context so subsequent steps can use it.
                if step_result.ok and isinstance(data, dict):
                    ctx.update(data)

                # Run verification if provided.
                if step.verify and step_result.ok:
                    verification = step.verify({**ctx, **data})
                    step_result.verification = verification
                    if not verification.success:
                        step_result.ok = False
                        step_result.error = f"verification failed: {verification.error}"

            except Exception as e:
                step_result.ok = False
                step_result.error = str(e)
                log.warning("workflow.step.exception", step=step.name, error=str(e))

            step_result.duration_ms = int((time.time() - step_start) * 1000)
            step_results.append(step_result)

            # Update progress.
            self.db.update_workflow_run(run_id, i + 1)

            if step_result.ok:
                log.info("workflow.step.done", step=step.name, duration_ms=step_result.duration_ms)
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_COMPLETED, {
                        "workflow": workflow.name, "step": step.name,
                        "run_id": run_id,
                    }))
            else:
                log.warning("workflow.step.failed", step=step.name, error=step_result.error)
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_FAILED, {
                        "workflow": workflow.name, "step": step.name,
                        "error": step_result.error, "run_id": run_id,
                    }))

                if step.required:
                    result.ok = False
                    result.error = f"required step '{step.name}' failed: {step_result.error}"
                    break

        result.steps = step_results
        result.duration_ms = int((time.time() - start) * 1000)

        # Generate summary.
        result.summary = self._generate_summary(workflow, result)

        # Record in database.
        status = "completed" if result.ok else "failed"
        self.db.finish_workflow_run(
            run_id, status, result=result.summary,
            error=result.error, duration_ms=result.duration_ms,
        )

        if self.bus:
            self.bus.publish(Event(WORKFLOW_COMPLETED, {
                "workflow": workflow.name, "run_id": run_id,
                "ok": result.ok, "steps_done": result.steps_completed,
                "steps_failed": result.steps_failed,
                "duration_ms": result.duration_ms,
            }))

        log.info("workflow.completed", name=workflow.name, ok=result.ok,
                 steps_done=result.steps_completed, duration_ms=result.duration_ms)
        return result

    def _generate_summary(self, workflow: Workflow, result: WorkflowResult) -> str:
        """Generate a text summary of the workflow run."""
        lines = [f"Workflow: {workflow.name}"]
        lines.append(f"Status: {'completed' if result.ok else 'failed'}")
        lines.append(f"Steps: {result.steps_completed}/{len(result.steps)} completed")
        for s in result.steps:
            status = "OK" if s.ok else "FAIL"
            lines.append(f"  [{status}] {s.step_name} ({s.duration_ms}ms)")
            if s.verification:
                v = s.verification
                lines.append(f"    verification: {'passed' if v.success else 'failed'}")
                for evidence in v.evidence:
                    lines.append(f"      - {evidence}")
            if s.error:
                lines.append(f"    error: {s.error}")
        return "\n".join(lines)
