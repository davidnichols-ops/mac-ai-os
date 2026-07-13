"""Actions package — the agency layer for controlled AI actions."""

from app.actions.builtin import builtin_actions
from app.actions.executor import ACTION_DENIED, ACTION_EXECUTED, ActionExecutor
from app.actions.models import Action, ActionRequest, ActionResult
from app.actions.permissions import PermissionChecker, PermissionError
from app.actions.registry import ActionRegistry

__all__ = [
    "Action",
    "ActionRequest",
    "ActionResult",
    "ActionRegistry",
    "ActionExecutor",
    "PermissionChecker",
    "PermissionError",
    "builtin_actions",
    "ACTION_EXECUTED",
    "ACTION_DENIED",
]
