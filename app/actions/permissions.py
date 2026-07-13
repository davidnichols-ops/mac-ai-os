"""Permission system — controls which actions are allowed.

Permissions are simple strings (e.g. "filesystem.read", "application.launch").
The PermissionChecker validates that the caller has the required permissions
for an action. Permissions can be granted globally, per-session, or
interactively (prompting the user).
"""

from __future__ import annotations

from app.log import get_logger

log = get_logger(__name__)


class PermissionError(Exception):
    """Raised when an action lacks required permissions."""


class PermissionChecker:
    """Checks whether a set of granted permissions satisfies an action's requirements."""

    def __init__(self, granted: set[str] | None = None, auto_approve: bool = False):
        self.granted = granted or set()
        self.auto_approve = auto_approve

    def grant(self, permission: str) -> None:
        self.granted.add(permission)
        log.info("permission.granted", permission=permission)

    def revoke(self, permission: str) -> None:
        self.granted.discard(permission)
        log.info("permission.revoked", permission=permission)

    def has(self, permission: str) -> bool:
        # Wildcard permission grants everything.
        if "*" in self.granted:
            return True
        return permission in self.granted

    def check(self, required: list[str]) -> bool:
        """Check if all required permissions are granted."""
        missing = [p for p in required if not self.has(p)]
        if missing:
            log.warning("permission.denied", missing=missing, granted=list(self.granted))
            return False
        return True

    def check_or_raise(self, required: list[str]) -> None:
        if not self.check(required):
            missing = [p for p in required if not self.has(p)]
            raise PermissionError(f"missing permissions: {missing}")
