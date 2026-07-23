"""Adversarial architecture review — probing for weaknesses.

Phase 10: This module contains tests and analysis that adversarially
probe the workflow architecture for weaknesses:

  1. Race conditions in async execution
  2. State machine gaps — unreachable or missing transitions
  3. Trust boundary bypasses — can workflows execute actions without auth?
  4. Persistence gaps — what happens if DB writes fail mid-step?
  5. Idempotency holes — can side effects be duplicated on resume?
  6. Resource leaks — thread pool, DB connections

The review identifies known limitations and documents them as
KnownLimitation entries. These are not bugs to fix immediately, but
risks to be aware of and address in future phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.log import get_logger

log = get_logger(__name__)


@dataclass
class KnownLimitation:
    """A known limitation identified during the adversarial review."""

    id: str
    title: str
    severity: str  # low, medium, high, critical
    description: str
    affected_component: str
    mitigation: str = ""
    future_phase: str = ""  # which future phase will address it

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "description": self.description,
            "affected_component": self.affected_component,
            "mitigation": self.mitigation,
            "future_phase": self.future_phase,
        }


def get_known_limitations() -> list[KnownLimitation]:
    """Return all known limitations from the adversarial review."""
    return [
        KnownLimitation(
            id="ADV-001",
            title="Async run_id mismatch",
            severity="medium",
            description=(
                "When start(async_exec=True) is called, a run_id is pre-created "
                "for tracking, but the engine.run() call inside the thread creates "
                "its own run_id. This means the pre-created run becomes orphaned "
                "and the actual run has a different ID than what was returned to "
                "the caller."
            ),
            affected_component="WorkflowRuntime._start_async",
            mitigation=(
                "The get_result() method works correctly because it tracks futures "
                "by the pre-created run_id. However, inspect() and cancel() using "
                "the pre-created run_id will not find the actual run."
            ),
            future_phase="Phase 11: Fix async run_id propagation",
        ),
        KnownLimitation(
            id="ADV-002",
            title="SIGALRM timeout not thread-safe",
            severity="medium",
            description=(
                "The per-step timeout uses SIGALRM, which is process-global and "
                "not thread-safe. If multiple workflows run concurrently in "
                "different threads, a timeout in one thread can affect another."
            ),
            affected_component="WorkflowEngine._execute_step (timeout)",
            mitigation=(
                "Only use timeout in synchronous (single-threaded) execution. "
                "For async execution, consider using threading.Timer or "
                "asyncio.wait_for instead."
            ),
            future_phase="Phase 11: Thread-safe timeout mechanism",
        ),
        KnownLimitation(
            id="ADV-003",
            title="Checkpoint context serialization may lose data",
            severity="low",
            description=(
                "The _save_checkpoint method filters out non-serializable items "
                "(callables, db connection). This means context containing custom "
                "objects will lose those fields on resume."
            ),
            affected_component="WorkflowEngine._save_checkpoint",
            mitigation=(
                "Only primitive types (str, int, float, dict, list) are preserved. "
                "Workflow handlers should store only serializable data in context."
            ),
            future_phase="Phase 11: Pluggable context serializers",
        ),
        KnownLimitation(
            id="ADV-004",
            title="Idempotency keys are global, not per-run",
            severity="low",
            description=(
                "Side effect idempotency keys are checked globally across all "
                "runs. This means if two different workflows use the same key, "
                "the second will be incorrectly skipped. Keys should include "
                "workflow_id or run_id in their construction."
            ),
            affected_component="Database.side_effect_exists",
            mitigation=(
                "Always construct idempotency keys with run_id or workflow_id: "
                "f'action:{action_name}:run:{run_id}'"
            ),
            future_phase="Phase 11: Namespace idempotency keys by run",
        ),
        KnownLimitation(
            id="ADV-005",
            title="No mid-step cancellation",
            severity="medium",
            description=(
                "Cancellation is only checked between steps, not during step "
                "execution. A long-running step cannot be cancelled mid-execution."
            ),
            affected_component="WorkflowEngine.run (cancellation check)",
            mitigation=(
                "Keep steps short. For long-running operations, break them into "
                "sub-steps or use async execution with timeout."
            ),
            future_phase="Phase 11: Cooperative cancellation in steps",
        ),
        KnownLimitation(
            id="ADV-006",
            title="Pause is a stub — no real pausing",
            severity="medium",
            description=(
                "The pause() method transitions the state to PAUSED but doesn't "
                "actually stop execution. The engine doesn't check for pause state "
                "between steps."
            ),
            affected_component="WorkflowRuntime.pause",
            mitigation=(
                "Use cancel() instead of pause() for now. True pausing requires "
                "engine-level support for checking pause state between steps."
            ),
            future_phase="Phase 11: Real pause/resume in engine",
        ),
        KnownLimitation(
            id="ADV-007",
            title="No DB transaction for step execution",
            severity="medium",
            description=(
                "Step execution (handler + persistence) is not wrapped in a DB "
                "transaction. If the DB write fails after the handler succeeds, "
                "the step result is lost but the side effect may have already "
                "occurred."
            ),
            affected_component="WorkflowEngine._execute_step",
            mitigation=(
                "Use idempotency keys for all side effects. On resume, the "
                "checkpoint will be from the previous step, so the step will "
                "re-execute, but the idempotency key will prevent duplicate "
                "side effects."
            ),
            future_phase="Phase 11: DB transactions for step execution",
        ),
        KnownLimitation(
            id="ADV-008",
            title="Saga compensation is not integrated into engine",
            severity="low",
            description=(
                "The SagaCoordinator exists but is not automatically invoked by "
                "the engine when a step fails. Workflows must manually call the "
                "coordinator."
            ),
            affected_component="WorkflowEngine (failure handling)",
            mitigation=(
                "Use FailurePolicy with CompensationStrategy.SAGA and manually "
                "invoke SagaCoordinator.compensate() in the workflow's failure "
                "handler."
            ),
            future_phase="Phase 11: Integrate saga into engine failure path",
        ),
        KnownLimitation(
            id="ADV-009",
            title="Thread pool not isolated per workflow",
            severity="low",
            description=(
                "All async workflows share a single ThreadPoolExecutor. A "
                "long-running workflow can block the pool, preventing other "
                "workflows from starting."
            ),
            affected_component="WorkflowRuntime (ThreadPoolExecutor)",
            mitigation=(
                "Use max_workers to limit concurrency. For isolation, create "
                "separate WorkflowRuntime instances with their own pools."
            ),
            future_phase="Phase 11: Per-workflow thread isolation",
        ),
        KnownLimitation(
            id="ADV-010",
            title="No workflow version migration",
            severity="low",
            description=(
                "If a workflow definition changes between runs, resuming from "
                "a checkpoint may use a different step structure than what was "
                "checkpointed. There's no migration path."
            ),
            affected_component="WorkflowEngine.resume",
            mitigation=(
                "Always increment the workflow version when changing the step "
                "structure. The resume method checks workflow_id and version."
            ),
            future_phase="Phase 11: Workflow version migration",
        ),
    ]


def get_limitations_by_severity(severity: str) -> list[KnownLimitation]:
    """Get limitations filtered by severity."""
    return [
        lim for lim in get_known_limitations()
        if lim.severity == severity
    ]


def format_review_report() -> str:
    """Format the adversarial review as a human-readable report."""
    limitations = get_known_limitations()
    lines = [
        "Adversarial Architecture Review",
        "=" * 40,
        f"Total known limitations: {len(limitations)}",
        "",
    ]

    by_severity: dict[str, list[KnownLimitation]] = {}
    for lim in limitations:
        by_severity.setdefault(lim.severity, []).append(lim)

    for sev in ["critical", "high", "medium", "low"]:
        if sev in by_severity:
            lines.append(f"\n{sev.upper()} ({len(by_severity[sev])}):")
            for lim in by_severity[sev]:
                lines.append(f"  [{lim.id}] {lim.title}")
                lines.append(f"    Component: {lim.affected_component}")
                lines.append(f"    Mitigation: {lim.mitigation[:100]}")
                if lim.future_phase:
                    lines.append(f"    Future: {lim.future_phase}")

    return "\n".join(lines)


def get_trust_boundary_analysis() -> dict[str, Any]:
    """Analyze the trust boundary for potential bypasses.

    The trust boundary in MAOS is the ActionExecutor. Workflows should
    not be able to execute actions without going through the authorization
    pipeline. This analysis checks for bypasses.
    """
    return {
        "primary_trust_boundary": "ActionExecutor",
        "workflow_integration_point": "AuthorizationPipeline → ActionExecutor",
        "bypass_risks": [
            {
                "risk": "Workflow step handler directly calls ActionExecutor",
                "severity": "high",
                "mitigation": "Step handlers should use engine.execute_action_proposal()",
                "status": "mitigated_by_convention",
            },
            {
                "risk": "Workflow step handler directly modifies filesystem",
                "severity": "high",
                "mitigation": "Step handlers should declare side_effects and use actions",
                "status": "mitigated_by_convention",
            },
            {
                "risk": "Permission checker not configured",
                "severity": "medium",
                "mitigation": "Engine accepts permission_checker=None, which skips checks",
                "status": "known_limitation",
            },
        ],
        "trustcard_integration": {
            "level": "interface_only",
            "description": (
                "TrustcardIntegrationPoint provides the hook for danger scoring, "
                "but the actual Trustcard integration runs at the MCP proxy level. "
                "The workflow engine uses the interface; the proxy enforces."
            ),
            "enforcement_point": "MCP proxy (mcp-trustcard)",
            "workflow_enforcement": "AuthorizationPipeline.trustcard.score_action()",
        },
    }
