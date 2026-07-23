"""Action proposal — the bridge between workflows and the action executor.

Phase 4: Workflows can propose actions that go through the same trust
boundary (ActionExecutor) as direct LLM requests. This module provides:

  - ActionProposal: a dataclass linking a workflow step to an action request
  - AuthorizationPipeline: validates proposals, checks permissions, and
    optionally routes through the approval queue for irreversible actions
  - TrustcardIntegrationPoint: interface for manifest-based danger scoring
    (the actual Trustcard integration is at the MCP proxy level; this
    provides the hook for MAOS-internal danger scoring)

The flow:
  1. Workflow step declares it wants to execute an action
  2. Engine creates an ActionProposal
  3. AuthorizationPipeline validates:
     a. Action exists in registry
     b. Permissions are satisfied
     c. Reversibility check (irreversible → approval queue)
     d. Danger score (if trustcard integration is configured)
  4. If approved → ActionExecutor.execute()
  5. If needs approval → workflow pauses, approval queue entry created
  6. If denied → step fails with authorization error
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.actions.models import ActionRequest, ActionResult
from app.log import get_logger

log = get_logger(__name__)


@dataclass
class ActionProposal:
    """A proposal to execute an action from within a workflow step.

    Attributes:
        action_name: Name of the registered action to execute.
        target: Optional target for the action (e.g., file path, workspace).
        parameters: Parameters to pass to the action handler.
        reason: Human-readable explanation of why this action is needed.
        risk_level: Assessed risk level ("low", "medium", "high").
        requires_approval: If True, the action will be routed through the
            approval queue before execution. If False, it executes directly
            (if permissions are satisfied).
        idempotency_key: Unique key to prevent duplicate execution on resume.
            If not provided, one is generated from action + target + run_id.
        side_effect_type: Type of side effect (e.g., "email.send", "task.create")
            for tracking in the workflow_side_effects table.
    """

    action_name: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    risk_level: str = "low"
    requires_approval: bool = False
    idempotency_key: str = ""
    side_effect_type: str = ""

    def to_action_request(self) -> ActionRequest:
        """Convert to an ActionRequest for the ActionExecutor."""
        return ActionRequest(
            action=self.action_name,
            target=self.target,
            parameters=self.parameters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "target": self.target,
            "parameters": self.parameters,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "idempotency_key": self.idempotency_key,
            "side_effect_type": self.side_effect_type,
        }


@dataclass
class AuthorizationResult:
    """The result of authorizing an action proposal.

    Attributes:
        approved: True if the action is authorized to execute.
        needs_approval: True if the action requires human approval (irreversible).
        approval_id: The ID of the approval queue entry (if needs_approval).
        denied_reason: Why the action was denied (if approved is False).
        danger_score: Trustcard danger score (0-100, if computed).
        danger_flags: List of danger flags from trustcard analysis.
    """

    approved: bool
    needs_approval: bool = False
    approval_id: int | None = None
    denied_reason: str = ""
    danger_score: float = 0.0
    danger_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "needs_approval": self.needs_approval,
            "approval_id": self.approval_id,
            "denied_reason": self.denied_reason,
            "danger_score": self.danger_score,
            "danger_flags": self.danger_flags,
        }


class TrustcardIntegrationPoint:
    """Interface for manifest-based danger scoring.

    This is the integration point where Trustcard (or a similar danger
    detection system) can be plugged in. By default, it does nothing
    (no danger scoring). When Trustcard is configured, it scores actions
    based on their danger profile.

    The actual Trustcard integration runs at the MCP proxy level (see
    ~/.config/devin/manifests/maos.json). This class provides the hook
    for MAOS-internal danger scoring that can be used by the workflow
    engine to make authorization decisions.
    """

    def __init__(self, *, danger_threshold: float = 70.0):
        self.danger_threshold = danger_threshold

    def score_action(
        self, action_name: str, target: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> tuple[float, list[str]]:
        """Score an action for danger.

        Returns:
            (score, flags) where score is 0-100 (higher = more dangerous)
            and flags is a list of danger indicators.
        """
        # Default: no danger scoring. Subclasses or configurations
        # can override this to integrate with Trustcard.
        return 0.0, []

    def is_dangerous(self, score: float) -> bool:
        """Check if a danger score exceeds the threshold."""
        return score >= self.danger_threshold


class AuthorizationPipeline:
    """Validates and authorizes action proposals from workflow steps.

    The pipeline:
    1. Checks if the action exists in the registry
    2. Checks permissions via PermissionChecker
    3. Checks reversibility (irreversible → approval queue)
    4. Scores danger via TrustcardIntegrationPoint (if configured)
    5. Returns AuthorizationResult
    """

    def __init__(
        self,
        *,
        action_executor: Any | None = None,
        permission_checker: Any | None = None,
        action_registry: Any | None = None,
        db: Any | None = None,
        trustcard: TrustcardIntegrationPoint | None = None,
        allow_irreversible: bool = False,
    ):
        self.action_executor = action_executor
        self.permission_checker = permission_checker
        self.action_registry = action_registry
        self.db = db
        self.trustcard = trustcard or TrustcardIntegrationPoint()
        self.allow_irreversible = allow_irreversible

    def authorize(
        self, proposal: ActionProposal, *,
        run_id: int | None = None,
    ) -> AuthorizationResult:
        """Authorize an action proposal.

        Args:
            proposal: The action proposal to authorize.
            run_id: The workflow run ID (for approval queue linking).

        Returns:
            AuthorizationResult indicating whether to proceed.
        """
        # 1. Check action exists.
        if self.action_registry:
            action = self.action_registry.get(proposal.action_name)
            if action is None:
                return AuthorizationResult(
                    approved=False,
                    denied_reason=f"unknown action: {proposal.action_name}",
                )

            # 2. Check permissions.
            if self.permission_checker:
                if not self.permission_checker.check(action.permissions):
                    return AuthorizationResult(
                        approved=False,
                        denied_reason=f"permission denied: requires {action.permissions}",
                    )

            # 3. Check reversibility.
            if not action.reversible and not self.allow_irreversible:
                if proposal.requires_approval and self.db:
                    # Route to approval queue.
                    approval_id = self.db.submit_approval(
                        action_name=proposal.action_name,
                        target=proposal.target,
                        parameters=json.dumps(proposal.parameters),
                        reason=proposal.reason,
                        risk=proposal.risk_level,
                        workflow_id=run_id,
                    )
                    log.info(
                        "authorization.needs_approval",
                        action=proposal.action_name,
                        approval_id=approval_id,
                        run_id=run_id,
                    )
                    return AuthorizationResult(
                        approved=False,
                        needs_approval=True,
                        approval_id=approval_id,
                    )
                return AuthorizationResult(
                    approved=False,
                    denied_reason="irreversible action requires explicit confirmation",
                )

        # 4. Danger scoring (Trustcard integration point).
        score, flags = self.trustcard.score_action(
            proposal.action_name, proposal.target, proposal.parameters,
        )
        if self.trustcard.is_dangerous(score):
            return AuthorizationResult(
                approved=False,
                denied_reason=f"action flagged as dangerous (score={score:.0f}): {flags}",
                danger_score=score,
                danger_flags=flags,
            )

        # 5. All checks passed.
        return AuthorizationResult(
            approved=True,
            danger_score=score,
            danger_flags=flags,
        )

    def execute(
        self, proposal: ActionProposal, *,
        run_id: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActionResult:
        """Authorize and execute an action proposal.

        If authorization fails, returns a failed ActionResult.
        If authorization requires approval, returns a failed ActionResult
        with a "needs approval" error (the workflow should pause).
        """
        auth = self.authorize(proposal, run_id=run_id)

        if not auth.approved:
            return ActionResult(
                ok=False,
                action=proposal.action_name,
                target=proposal.target,
                error=auth.denied_reason or "authorization denied",
                approved=False,
                executed=False,
            )

        if not self.action_executor:
            return ActionResult(
                ok=False,
                action=proposal.action_name,
                target=proposal.target,
                error="no action executor configured",
            )

        request = proposal.to_action_request()
        return self.action_executor.execute(request, context=context)
