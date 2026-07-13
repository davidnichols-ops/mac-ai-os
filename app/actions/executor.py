"""Action executor — the trust boundary between the LLM and the machine.

The executor:
1. Looks up the action in the registry
2. Checks permissions
3. Checks reversibility (refuses irreversible actions without confirmation)
4. Executes the handler
5. Logs the result to the action_log table
6. Publishes an event on the event bus

The LLM never calls the executor directly. The orchestrator or agent calls
it with an ActionRequest, and the executor decides whether to proceed.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.actions.models import ActionRequest, ActionResult
from app.actions.permissions import PermissionChecker
from app.actions.registry import ActionRegistry
from app.db import Database
from app.events.bus import EventBus
from app.events.models import Event
from app.log import get_logger

log = get_logger(__name__)

ACTION_EXECUTED = "ACTION_EXECUTED"
ACTION_DENIED = "ACTION_DENIED"


class ActionExecutor:
    """Validates and executes action requests from agents."""

    def __init__(
        self,
        registry: ActionRegistry,
        permissions: PermissionChecker,
        db: Database,
        bus: EventBus | None = None,
        *,
        allow_irreversible: bool = False,
    ):
        self.registry = registry
        self.permissions = permissions
        self.db = db
        self.bus = bus
        self.allow_irreversible = allow_irreversible

    def execute(
        self, request: ActionRequest, context: dict[str, Any] | None = None,
    ) -> ActionResult:
        """Execute an action request with full validation and logging."""
        ctx = context or {}
        start = time.time()

        # 1. Check action exists.
        action = self.registry.get(request.action)
        if not action:
            result = ActionResult(
                ok=False, action=request.action, target=request.target,
                error=f"unknown action: {request.action}",
            )
            self._log(request, result, 0)
            return result

        # 2. Check permissions.
        approved = self.permissions.check(action.permissions)
        if not approved:
            result = ActionResult(
                ok=False, action=request.action, target=request.target,
                error=f"permission denied: requires {action.permissions}",
                approved=False, reversible=action.reversible,
            )
            self._log(request, result, 0)
            if self.bus:
                self.bus.publish(Event(ACTION_DENIED, {
                    "action": request.action, "target": request.target,
                    "reason": "permission denied",
                }))
            return result

        # 3. Check reversibility.
        if not action.reversible and not self.allow_irreversible:
            result = ActionResult(
                ok=False, action=request.action, target=request.target,
                error="irreversible action requires explicit confirmation",
                approved=True, executed=False, reversible=False,
            )
            self._log(request, result, 0)
            return result

        # 4. Execute the handler.
        try:
            handler = self.registry.get_handler(request.action)
            if not handler:
                result = ActionResult(
                    ok=False, action=request.action, target=request.target,
                    error=f"no handler for action: {request.action}",
                    approved=True,
                )
                self._log(request, result, 0)
                return result

            result = handler(request, ctx)
            result.approved = True
            result.executed = True
            result.reversible = action.reversible
            result.duration_ms = int((time.time() - start) * 1000)

            self._log(request, result, result.duration_ms)
            if self.bus:
                self.bus.publish(Event(ACTION_EXECUTED, {
                    "action": request.action, "target": request.target,
                    "ok": result.ok, "result": result.result[:200],
                }))
            log.info("action.executed", action=request.action,
                     target=request.target, ok=result.ok,
                     duration_ms=result.duration_ms)
            return result

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            result = ActionResult(
                ok=False, action=request.action, target=request.target,
                error=str(e), approved=True, executed=False,
                reversible=action.reversible, duration_ms=duration_ms,
            )
            self._log(request, result, duration_ms)
            log.warning("action.failed", action=request.action, error=str(e))
            return result

    def _log(self, request: ActionRequest, result: ActionResult, duration_ms: int) -> None:
        self.db.log_action(
            request.action, request.target, json.dumps(request.parameters),
            approved=result.approved, executed=result.executed,
            result=result.result[:500], error=result.error[:500],
            duration_ms=duration_ms,
        )
