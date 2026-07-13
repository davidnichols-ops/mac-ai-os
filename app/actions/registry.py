"""Action registry — maps action names to their schemas and handlers.

Each action registers a schema (name, permissions, parameters) and a handler
function. The executor looks up the action by name, checks permissions, and
calls the handler.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.actions.models import Action, ActionRequest, ActionResult
from app.log import get_logger

log = get_logger(__name__)

ActionHandler = Callable[[ActionRequest, dict[str, Any]], ActionResult]


class ActionRegistry:
    """Registry of available actions and their handlers."""

    def __init__(self):
        self._actions: dict[str, Action] = {}
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, action: Action, handler: ActionHandler) -> None:
        self._actions[action.name] = action
        self._handlers[action.name] = handler
        log.info("action.registered", name=action.name,
                 permissions=action.permissions, reversible=action.reversible)

    def get(self, name: str) -> Action | None:
        return self._actions.get(name)

    def get_handler(self, name: str) -> ActionHandler | None:
        return self._handlers.get(name)

    def list_actions(self) -> list[Action]:
        return list(self._actions.values())

    def names(self) -> list[str]:
        return list(self._actions.keys())

    def exists(self, name: str) -> bool:
        return name in self._actions
