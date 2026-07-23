"""Workflow engine — executes multi-step workflows with verification.

The engine:
1. Creates a workflow_run record in the database (with workflow_id + version)
2. Persists the workflow definition for future reference / resumability
3. Executes each step in order, persisting per-step records
4. Runs verification after each step (if provided)
5. Handles failures: required steps stop the workflow, optional steps continue
6. Records the final result and summary
7. Validates state transitions via the WorkflowStatus state machine
8. Sends heartbeats during execution for crash detection

Phase 1 additions:
  - State machine validation (WorkflowStatus)
  - Per-step persistence (workflow_step_runs table)
  - Workflow definition persistence
  - Heartbeat updates during execution
  - Crash recovery (detect stale 'running' on startup)
  - cancel() and pause() methods (pause is a stub for Phase 3)

Phase 2 additions:
  - Retry with exponential backoff (max_retries, retry_delay_ms)
  - Per-step timeout (timeout_seconds)
  - Per-step permission checks (permissions)
  - Failure behavior: skip / abort / compensate
  - Side effect declaration tracking
  - All attempts tracked in StepResult.attempts
"""

from __future__ import annotations

import json
import signal
import time
from typing import Any

from app.db import Database
from app.events.bus import EventBus
from app.events.models import Event
from app.log import get_logger
from app.workflows.models import (
    FailureBehavior,
    StepResult,
    Workflow,
    WorkflowResult,
)
from app.workflows.states import (
    InvalidTransitionError,
    WorkflowStatus,
    validate_transition,
)

log = get_logger(__name__)

WORKFLOW_STARTED = "WORKFLOW_STARTED"
WORKFLOW_STEP_COMPLETED = "WORKFLOW_STEP_COMPLETED"
WORKFLOW_STEP_FAILED = "WORKFLOW_STEP_FAILED"
WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
WORKFLOW_CRASHED = "WORKFLOW_CRASHED"
WORKFLOW_PAUSED = "WORKFLOW_PAUSED"
WORKFLOW_STEP_RETRY = "WORKFLOW_STEP_RETRY"
WORKFLOW_STEP_TIMED_OUT = "WORKFLOW_STEP_TIMED_OUT"
WORKFLOW_STEP_COMPENSATED = "WORKFLOW_STEP_COMPENSATED"
WORKFLOW_RESUMED = "WORKFLOW_RESUMED"


class WorkflowEngine:
    """Executes workflows step by step with verification.

    Phase 1: State machine validation, per-step persistence,
    workflow definition persistence, heartbeats, and crash recovery.

    Phase 2: Retry with backoff, per-step timeout, permission checks,
    and failure behavior (skip/abort/compensate).
    """

    def __init__(
        self, db: Database, bus: EventBus | None = None,
        *,
        permission_checker: Any | None = None,
        action_executor: Any | None = None,
        action_registry: Any | None = None,
        authorization_pipeline: Any | None = None,
    ):
        self.db = db
        self.bus = bus
        self.permission_checker = permission_checker
        self.action_executor = action_executor
        self.action_registry = action_registry
        self._cancelled_runs: set[int] = set()

        # Phase 4: Authorization pipeline for action proposals.
        if authorization_pipeline is not None:
            self.auth_pipeline = authorization_pipeline
        else:
            # Lazy import to avoid circular dependency.
            from app.workflows.proposals import AuthorizationPipeline
            self.auth_pipeline = AuthorizationPipeline(
                action_executor=action_executor,
                permission_checker=permission_checker,
                action_registry=action_registry,
                db=db,
            )

    def run(self, workflow: Workflow, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute a workflow and return the result.

        This is a synchronous, blocking call. The workflow runs to completion
        (or failure/cancellation) before returning.
        """
        ctx = {**workflow.context, **(context or {})}
        ctx["db"] = self.db

        # Persist the workflow definition for future reference.
        self._persist_definition(workflow)

        start = time.time()
        run_id = self.db.start_workflow_run(
            workflow.name, len(workflow.steps),
            workflow_id=workflow.workflow_id, version=workflow.version,
        )
        log.info("workflow.started", name=workflow.name, steps=len(workflow.steps),
                 run_id=run_id, workflow_id=workflow.workflow_id, version=workflow.version)

        if self.bus:
            self.bus.publish(Event(WORKFLOW_STARTED, {
                "workflow": workflow.name, "run_id": run_id,
                "steps": len(workflow.steps),
                "workflow_id": workflow.workflow_id,
                "version": workflow.version,
            }))

        result = WorkflowResult(
            workflow_name=workflow.name, ok=True, run_id=run_id,
            workflow_id=workflow.workflow_id,
        )
        step_results: list[StepResult] = []

        for i, step in enumerate(workflow.steps):
            # Check for cancellation before each step.
            if run_id in self._cancelled_runs:
                log.info("workflow.cancelled", name=workflow.name, run_id=run_id, step_index=i)
                self._transition(run_id, WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED)
                result.ok = False
                result.error = "workflow cancelled by user"
                result.status = WorkflowStatus.CANCELLED.value
                result.steps = step_results
                result.duration_ms = int((time.time() - start) * 1000)
                self._publish_cancelled(workflow.name, run_id)
                return result

            # Heartbeat before each step.
            self.db.heartbeat_workflow_run(run_id)

            step_result = self._execute_step(workflow, step, i, ctx, run_id)
            step_results.append(step_result)

            # Update progress.
            self.db.update_workflow_run(run_id, i + 1)

            if step_result.ok:
                log.info("workflow.step.done", step=step.name,
                         duration_ms=step_result.duration_ms, step_id=step.step_id,
                         attempts=step_result.attempt)
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_COMPLETED, {
                        "workflow": workflow.name, "step": step.name,
                        "run_id": run_id, "step_id": step.step_id,
                        "attempts": step_result.attempt,
                    }))
                # Phase 3: Save checkpoint after each successful step.
                self._save_checkpoint(run_id, i, ctx)
            else:
                log.warning("workflow.step.failed", step=step.name,
                           error=step_result.error, step_id=step.step_id,
                           on_failure=step.on_failure.value)
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_FAILED, {
                        "workflow": workflow.name, "step": step.name,
                        "error": step_result.error, "run_id": run_id,
                        "step_id": step.step_id,
                        "on_failure": step.on_failure.value,
                    }))

                # Determine failure behavior.
                # Backward compat: required=False → SKIP (unless on_failure explicitly overrides)
                should_abort = False
                if step.on_failure == FailureBehavior.ABORT and step.required:
                    should_abort = True
                elif step.on_failure == FailureBehavior.ABORT and not step.required:
                    # required=False with default ABORT → treat as SKIP for backward compat
                    should_abort = False
                elif step.on_failure == FailureBehavior.COMPENSATE:
                    # Run compensation handler if provided.
                    if step.compensation:
                        try:
                            step.compensation(ctx, step_result)
                            step_result.compensated = True
                            log.info("workflow.step.compensated",
                                    step=step.name, step_id=step.step_id)
                            if self.bus:
                                self.bus.publish(Event(WORKFLOW_STEP_COMPENSATED, {
                                    "workflow": workflow.name, "step": step.name,
                                    "run_id": run_id, "step_id": step.step_id,
                                }))
                        except Exception as ce:
                            log.warning("workflow.step.compensation_failed",
                                       step=step.name, error=str(ce))
                            should_abort = True
                    else:
                        # No compensation handler — treat as abort.
                        should_abort = True
                # FailureBehavior.SKIP: just continue

                if should_abort:
                    result.ok = False
                    result.error = f"step '{step.name}' failed: {step_result.error}"
                    break

        result.steps = step_results
        result.duration_ms = int((time.time() - start) * 1000)
        result.summary = self._generate_summary(workflow, result)

        # Determine terminal status.
        if run_id in self._cancelled_runs:
            terminal_status = WorkflowStatus.CANCELLED
        elif result.ok:
            terminal_status = WorkflowStatus.COMPLETED
        else:
            terminal_status = WorkflowStatus.FAILED

        result.status = terminal_status.value

        # Transition to terminal state.
        self._transition(run_id, WorkflowStatus.RUNNING, terminal_status,
                         result=result.summary, error=result.error,
                         duration_ms=result.duration_ms)

        if self.bus:
            self.bus.publish(Event(WORKFLOW_COMPLETED, {
                "workflow": workflow.name, "run_id": run_id,
                "ok": result.ok, "status": terminal_status.value,
                "steps_done": result.steps_completed,
                "steps_failed": result.steps_failed,
                "duration_ms": result.duration_ms,
                "workflow_id": workflow.workflow_id,
            }))

        log.info("workflow.completed", name=workflow.name, ok=result.ok,
                 status=terminal_status.value,
                 steps_done=result.steps_completed, duration_ms=result.duration_ms)
        return result

    def _execute_step(
        self, workflow: Workflow, step: Any, index: int,
        ctx: dict[str, Any], run_id: int,
    ) -> StepResult:
        """Execute a single step with retry, timeout, and permission checks.

        Phase 2: Now supports:
        - Permission checks before execution
        - Retry with exponential backoff
        - Per-step timeout (via SIGALRM on Unix)
        - All attempts tracked in step_result.attempts
        """
        step_start = time.time()
        step_result = StepResult(
            step_name=step.name, ok=False,
            step_id=step.step_id, attempt=1,
        )

        # Phase 2: Permission check.
        if step.permissions and self.permission_checker:
            if not self.permission_checker.check(step.permissions):
                step_result.ok = False
                step_result.error = f"permission denied: requires {step.permissions}"
                step_result.duration_ms = int((time.time() - step_start) * 1000)
                step_result.attempts.append({
                    "attempt": 1, "ok": False,
                    "error": step_result.error, "duration_ms": step_result.duration_ms,
                })
                self._persist_step_result(run_id, step, index, step_result)
                return step_result

        # Retry loop.
        max_attempts = step.max_retries + 1  # max_retries=0 → 1 attempt
        delay_ms = step.retry_delay_ms

        for attempt in range(1, max_attempts + 1):
            # Check for cancellation before each attempt.
            if run_id in self._cancelled_runs:
                step_result.ok = False
                step_result.error = "cancelled"
                step_result.duration_ms = int((time.time() - step_start) * 1000)
                break

            step_result.attempt = attempt
            attempt_start = time.time()

            # Persist step run start for this attempt.
            step_run_id = self.db.start_workflow_step_run(
                run_id, step.step_id, step.name, index, attempt=attempt,
            )

            try:
                # Phase 2: Timeout via SIGALRM (Unix only).
                if step.timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
                    def _timeout_handler(signum: int, frame: Any) -> None:
                        raise TimeoutError(f"step timed out after {step.timeout_seconds}s")
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.setitimer(signal.ITIMER_REAL, step.timeout_seconds)

                try:
                    data = step.handler(ctx)
                finally:
                    if step.timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, old_handler)

                step_result.ok = bool(data.get("ok", True))
                step_result.data = data

                # Extract error message from handler return if failed.
                if not step_result.ok and isinstance(data, dict) and data.get("error"):
                    step_result.error = str(data["error"])

                # Run verification if provided.
                if step.verify and step_result.ok:
                    verification = step.verify({**ctx, **data})
                    step_result.verification = verification
                    if not verification.success:
                        step_result.ok = False
                        step_result.error = f"verification failed: {verification.error}"

                attempt_duration = int((time.time() - attempt_start) * 1000)
                step_result.attempts.append({
                    "attempt": attempt, "ok": step_result.ok,
                    "error": step_result.error if not step_result.ok else "",
                    "duration_ms": attempt_duration,
                })

                # Persist this attempt's result.
                step_status = "completed" if step_result.ok else "failed"
                ver_str = ""
                if step_result.verification:
                    ver_str = json.dumps(step_result.verification.to_dict())
                self.db.finish_workflow_step_run(
                    step_run_id, step_status,
                    data=json.dumps(step_result.data, default=str),
                    verification=ver_str,
                    error=step_result.error,
                    duration_ms=attempt_duration,
                )

                if step_result.ok:
                    # Merge step data into context so subsequent steps can use it.
                    if isinstance(data, dict):
                        ctx.update(data)
                    break  # success, no more retries

                # Failed — check if we should retry.
                if attempt < max_attempts:
                    log.info("workflow.step.retry", step=step.name,
                            attempt=attempt, max_attempts=max_attempts,
                            delay_ms=delay_ms, step_id=step.step_id)
                    if self.bus:
                        self.bus.publish(Event(WORKFLOW_STEP_RETRY, {
                            "workflow": workflow.name, "step": step.name,
                            "run_id": run_id, "step_id": step.step_id,
                            "attempt": attempt, "max_attempts": max_attempts,
                            "delay_ms": delay_ms,
                        }))
                    time.sleep(delay_ms / 1000.0)
                    delay_ms = min(delay_ms * 2, 30000)  # exponential backoff, cap 30s

            except TimeoutError as e:
                step_result.ok = False
                step_result.error = str(e)
                step_result.timed_out = True
                attempt_duration = int((time.time() - attempt_start) * 1000)
                step_result.attempts.append({
                    "attempt": attempt, "ok": False,
                    "error": str(e), "duration_ms": attempt_duration,
                    "timed_out": True,
                })
                self.db.finish_workflow_step_run(
                    step_run_id, "timed_out",
                    data="{}", verification="",
                    error=str(e), duration_ms=attempt_duration,
                )
                log.warning("workflow.step.timed_out", step=step.name,
                           timeout=step.timeout_seconds, step_id=step.step_id)
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_TIMED_OUT, {
                        "workflow": workflow.name, "step": step.name,
                        "run_id": run_id, "step_id": step.step_id,
                        "timeout_seconds": step.timeout_seconds,
                    }))
                # Retry on timeout if retries remaining.
                if attempt < max_attempts:
                    time.sleep(delay_ms / 1000.0)
                    delay_ms = min(delay_ms * 2, 30000)
                    continue
                break

            except Exception as e:
                step_result.ok = False
                step_result.error = str(e)
                attempt_duration = int((time.time() - attempt_start) * 1000)
                step_result.attempts.append({
                    "attempt": attempt, "ok": False,
                    "error": str(e), "duration_ms": attempt_duration,
                })
                self.db.finish_workflow_step_run(
                    step_run_id, "failed",
                    data="{}", verification="",
                    error=str(e), duration_ms=attempt_duration,
                )
                log.warning("workflow.step.exception", step=step.name,
                           error=str(e), step_id=step.step_id, attempt=attempt)
                # Retry on exception if retries remaining.
                if attempt < max_attempts:
                    log.info("workflow.step.retry", step=step.name,
                            attempt=attempt, max_attempts=max_attempts,
                            delay_ms=delay_ms, step_id=step.step_id)
                    if self.bus:
                        self.bus.publish(Event(WORKFLOW_STEP_RETRY, {
                            "workflow": workflow.name, "step": step.name,
                            "run_id": run_id, "step_id": step.step_id,
                            "attempt": attempt, "max_attempts": max_attempts,
                            "delay_ms": delay_ms,
                        }))
                    time.sleep(delay_ms / 1000.0)
                    delay_ms = min(delay_ms * 2, 30000)
                    continue
                break

        step_result.duration_ms = int((time.time() - step_start) * 1000)
        return step_result

    def _persist_step_result(
        self, run_id: int, step: Any, index: int, step_result: StepResult,
    ) -> None:
        """Persist a step result (used for permission-denied cases)."""
        step_run_id = self.db.start_workflow_step_run(
            run_id, step.step_id, step.name, index, attempt=1,
        )
        self.db.finish_workflow_step_run(
            step_run_id, "failed",
            data=json.dumps(step_result.data, default=str),
            verification="",
            error=step_result.error,
            duration_ms=step_result.duration_ms,
        )

    def _save_checkpoint(self, run_id: int, step_index: int, ctx: dict[str, Any]) -> None:
        """Save a checkpoint of the workflow context after a successful step.

        Phase 3: Enables resumability from the last completed step.
        """
        try:
            # Serialize context, excluding non-serializable items like db connection.
            serializable = {
                k: v for k, v in ctx.items()
                if k != "db" and not callable(v)
            }
            self.db.save_checkpoint(
                run_id, step_index, json.dumps(serializable, default=str),
            )
        except Exception as e:
            log.warning("workflow.checkpoint.save_failed",
                       run_id=run_id, step_index=step_index, error=str(e))

    def resume(
        self, workflow: Workflow, run_id: int,
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Resume a crashed or interrupted workflow from the last checkpoint.

        Phase 3: Loads the latest checkpoint, restores context, and
        continues execution from the next step.

        Args:
            workflow: The workflow definition to resume.
            run_id: The ID of the crashed/interrupted run to resume.
            context: Additional context to merge with the checkpoint.

        Returns:
            WorkflowResult for the resumed run. This creates a new run
            with parent_run_id set to the original run_id.
        """
        # Verify the original run exists and is in a resumable state.
        original = self.db.get_workflow_run(run_id)
        if original is None:
            return WorkflowResult(
                workflow_name=workflow.name, ok=False,
                error=f"run {run_id} not found",
            )
        if original["status"] not in ("crashed", "interrupted", "failed", "timed_out"):
            return WorkflowResult(
                workflow_name=workflow.name, ok=False,
                error=f"run {run_id} is in state '{original['status']}', cannot resume",
            )

        # Load the latest checkpoint.
        checkpoint = self.db.latest_checkpoint(run_id)
        if checkpoint is None:
            # No checkpoint — start from the beginning.
            log.info("workflow.resume.no_checkpoint", run_id=run_id)
            return self.run(workflow, context=context)

        # Restore context from checkpoint.
        checkpoint_ctx = json.loads(checkpoint["context"])
        resume_from = checkpoint["step_index"] + 1  # next step after checkpoint

        # Merge with any additional context provided.
        ctx = {**workflow.context, **checkpoint_ctx, **(context or {})}
        ctx["db"] = self.db

        log.info("workflow.resume", run_id=run_id, resume_from=resume_from,
                workflow=workflow.name)

        # Create a new run as a child of the original.
        start = time.time()
        new_run_id = self.db.start_workflow_run(
            workflow.name, len(workflow.steps),
            workflow_id=workflow.workflow_id, version=workflow.version,
            parent_run_id=run_id,
        )

        if self.bus:
            self.bus.publish(Event(WORKFLOW_RESUMED, {
                "workflow": workflow.name, "run_id": new_run_id,
                "parent_run_id": run_id, "resume_from": resume_from,
            }))

        result = WorkflowResult(
            workflow_name=workflow.name, ok=True, run_id=new_run_id,
            workflow_id=workflow.workflow_id,
        )
        step_results: list[StepResult] = []

        # Execute only the remaining steps.
        for i in range(resume_from, len(workflow.steps)):
            step = workflow.steps[i]

            if new_run_id in self._cancelled_runs:
                self._transition(new_run_id, WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED)
                result.ok = False
                result.error = "workflow cancelled by user"
                result.status = WorkflowStatus.CANCELLED.value
                result.steps = step_results
                result.duration_ms = int((time.time() - start) * 1000)
                return result

            self.db.heartbeat_workflow_run(new_run_id)
            step_result = self._execute_step(workflow, step, i, ctx, new_run_id)
            step_results.append(step_result)
            self.db.update_workflow_run(new_run_id, i + 1)

            if step_result.ok:
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_COMPLETED, {
                        "workflow": workflow.name, "step": step.name,
                        "run_id": new_run_id, "step_id": step.step_id,
                        "attempts": step_result.attempt,
                    }))
                self._save_checkpoint(new_run_id, i, ctx)
            else:
                if self.bus:
                    self.bus.publish(Event(WORKFLOW_STEP_FAILED, {
                        "workflow": workflow.name, "step": step.name,
                        "error": step_result.error, "run_id": new_run_id,
                        "step_id": step.step_id,
                    }))

                should_abort = False
                if step.on_failure == FailureBehavior.ABORT and step.required:
                    should_abort = True
                elif step.on_failure == FailureBehavior.COMPENSATE:
                    if step.compensation:
                        try:
                            step.compensation(ctx, step_result)
                            step_result.compensated = True
                        except Exception as ce:
                            log.warning("workflow.step.compensation_failed",
                                       step=step.name, error=str(ce))
                            should_abort = True
                    else:
                        should_abort = True

                if should_abort:
                    result.ok = False
                    result.error = f"step '{step.name}' failed: {step_result.error}"
                    break

        result.steps = step_results
        result.duration_ms = int((time.time() - start) * 1000)
        result.summary = self._generate_summary(workflow, result)

        terminal_status = WorkflowStatus.COMPLETED if result.ok else WorkflowStatus.FAILED
        result.status = terminal_status.value
        self._transition(new_run_id, WorkflowStatus.RUNNING, terminal_status,
                         result=result.summary, error=result.error,
                         duration_ms=result.duration_ms)

        if self.bus:
            self.bus.publish(Event(WORKFLOW_COMPLETED, {
                "workflow": workflow.name, "run_id": new_run_id,
                "ok": result.ok, "status": terminal_status.value,
                "duration_ms": result.duration_ms,
                "workflow_id": workflow.workflow_id,
                "resumed_from": run_id,
            }))

        return result

    def record_side_effect(
        self, run_id: int, step: Any, effect_type: str,
        idempotency_key: str, details: dict[str, Any] | None = None,
    ) -> bool:
        """Record a side effect executed during a workflow step.

        Phase 3: Tracks external side effects for idempotency on resume.
        Returns True if the side effect was recorded (first execution),
        False if it was already recorded (duplicate — should be skipped).
        """
        if self.db.side_effect_exists(idempotency_key):
            log.info("workflow.side_effect.duplicate",
                    idempotency_key=idempotency_key, run_id=run_id)
            return False
        self.db.record_side_effect(
            run_id, step.step_id, step.name, effect_type, idempotency_key,
            json.dumps(details or {}, default=str),
        )
        return True

    def execute_action_proposal(
        self, proposal: Any, *,
        run_id: int | None = None,
        step: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Authorize and execute an action proposal from a workflow step.

        Phase 4: This is the integration point between the workflow engine
        and the action executor (trust boundary). The proposal goes through:
          1. Authorization pipeline (permissions, reversibility, danger score)
          2. Idempotency check (skip if already executed)
          3. ActionExecutor.execute() (the actual trust boundary)
          4. Side effect recording (for resume idempotency)

        Returns a dict with:
          - "ok": bool
          - "result": ActionResult (if executed)
          - "authorization": AuthorizationResult
          - "skipped": bool (True if skipped due to idempotency)
        """

        # Idempotency check — skip if already executed.
        idem_key = proposal.idempotency_key
        if idem_key and self.db.side_effect_exists(idem_key):
            log.info("workflow.action.skipped_duplicate",
                    action=proposal.action_name, idempotency_key=idem_key)
            return {"ok": True, "skipped": True, "authorization": None, "result": None}

        # Authorize.
        auth = self.auth_pipeline.authorize(proposal, run_id=run_id)

        if auth.needs_approval:
            # Workflow should pause and wait for approval.
            log.info("workflow.action.needs_approval",
                    action=proposal.action_name, approval_id=auth.approval_id)
            return {
                "ok": False, "skipped": False, "authorization": auth,
                "result": None, "needs_approval": True,
                "approval_id": auth.approval_id,
            }

        if not auth.approved:
            log.warning("workflow.action.denied",
                       action=proposal.action_name, reason=auth.denied_reason)
            return {
                "ok": False, "skipped": False, "authorization": auth,
                "result": None, "denied_reason": auth.denied_reason,
            }

        # Execute through the action executor (trust boundary).
        result = self.auth_pipeline.execute(proposal, run_id=run_id, context=context)

        # Record side effect for idempotency.
        if idem_key and step is not None and result.ok:
            effect_type = proposal.side_effect_type or f"action.{proposal.action_name}"
            self.record_side_effect(
                run_id or 0, step, effect_type, idem_key,
                details=proposal.to_dict(),
            )

        return {"ok": result.ok, "skipped": False, "authorization": auth, "result": result}

    def cancel(self, run_id: int) -> bool:
        """Request cancellation of a running workflow.

        This sets a flag that is checked between steps. The workflow
        will stop at the next step boundary. Returns True if the cancellation
        request was registered.
        """
        row = self.db.get_workflow_run(run_id)
        if row is None:
            return False
        if row["status"] not in ("running", "paused"):
            return False
        self._cancelled_runs.add(run_id)
        log.info("workflow.cancel.requested", run_id=run_id)
        return True

    def recover_crashed_runs(self, max_heartbeat_age_seconds: float = 300) -> list[int]:
        """Detect and mark stale 'running' workflows as 'crashed'.

        Should be called on startup to clean up workflows that were
        interrupted by a process crash.

        Returns the list of run IDs that were marked as crashed.
        """
        crashed = self.db.mark_stale_runs_as_crashed(max_heartbeat_age_seconds)
        for run_id in crashed:
            log.warning("workflow.crash.recovered", run_id=run_id)
            if self.bus:
                self.bus.publish(Event(WORKFLOW_CRASHED, {
                    "run_id": run_id,
                    "reason": "heartbeat timeout on startup",
                }))
        return crashed

    def get_run_status(self, run_id: int) -> WorkflowStatus | None:
        """Get the current status of a workflow run."""
        row = self.db.get_workflow_run(run_id)
        if row is None:
            return None
        try:
            return WorkflowStatus(row["status"])
        except ValueError:
            return None

    def get_run_details(self, run_id: int) -> dict[str, Any] | None:
        """Get full details of a workflow run including step results."""
        row = self.db.get_workflow_run(run_id)
        if row is None:
            return None
        steps = self.db.step_runs_for_workflow(run_id)
        return {
            "run": dict(row),
            "steps": [dict(s) for s in steps],
        }

    def _persist_definition(self, workflow: Workflow) -> None:
        """Persist the workflow definition to the database."""
        try:
            definition = json.dumps({
                "name": workflow.name,
                "description": workflow.description,
                "steps": [
                    {"name": s.name, "description": s.description,
                     "required": s.required, "action": s.action,
                     "step_id": s.step_id}
                    for s in workflow.steps
                ],
            })
            self.db.save_workflow_definition(
                workflow.workflow_id, workflow.name, workflow.version, definition,
            )
        except Exception as e:
            log.warning("workflow.definition.persist_failed",
                       workflow_id=workflow.workflow_id, error=str(e))

    def _transition(
        self, run_id: int, from_status: WorkflowStatus, to_status: WorkflowStatus,
        **kwargs: Any,
    ) -> None:
        """Validate and execute a state transition."""
        try:
            validate_transition(from_status, to_status)
        except InvalidTransitionError as e:
            log.error("workflow.transition.invalid",
                     from_status=from_status.value, to_status=to_status.value,
                     run_id=run_id, error=str(e))
            # Still write the transition (don't block on validation failure),
            # but log it loudly. This handles edge cases like double-completion.
        self.db.transition_workflow_run(run_id, to_status.value, **kwargs)

    def _publish_cancelled(self, workflow_name: str, run_id: int) -> None:
        if self.bus:
            self.bus.publish(Event(WORKFLOW_CANCELLED, {
                "workflow": workflow_name, "run_id": run_id,
            }))

    def _generate_summary(self, workflow: Workflow, result: WorkflowResult) -> str:
        """Generate a text summary of the workflow run."""
        lines = [f"Workflow: {workflow.name}"]
        lines.append(f"Status: {result.status or ('completed' if result.ok else 'failed')}")
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
