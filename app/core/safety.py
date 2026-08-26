"""Safety gate — risk classification and approval enforcement.

Before any tool executes, the safety gate evaluates the risk level. High-risk
and critical operations require user approval. The model must know what is
dangerous and what requires confirmation.

Risk levels:
    LOW      — read-only operations (file_read, git_status, list_files)
    MEDIUM   — side-effect operations (file_write, git_commit, run_command)
    HIGH     — potentially destructive (file_delete, git_reset, install packages)
    CRITICAL — irreversible damage (rm -rf, format disk, modify secrets)

The safety gate is the trust boundary between the model's intent and the
environment's execution. The model decides what to do; the safety gate
decides whether it's allowed to do it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from app.log import get_logger

log = get_logger(__name__)


class RiskLevel(IntEnum):
    """Severity of a tool operation."""
    LOW = 0       # read-only, no side effects
    MEDIUM = 1    # side effects, but reversible
    HIGH = 2      # potentially destructive, may not be reversible
    CRITICAL = 3  # irreversible damage, always requires approval


@dataclass
class SafetyDecision:
    """The result of a safety evaluation."""
    allowed: bool
    risk: RiskLevel
    reason: str = ""
    requires_approval: bool = False
    approval_id: str | None = None


@dataclass
class SafetyViolation:
    """A recorded safety violation."""
    level: RiskLevel
    tool_name: str
    reason: str
    args: dict[str, Any] = field(default_factory=dict)


# Commands that are always blocked — too dangerous for autonomous execution.
# Stored lowercase; the input command is lowercased before matching.
_BLOCKED_COMMANDS = {
    "rm -rf /", "rm -rf ~", "rm -rf *", "rm -rf .",
    "mkfs", "dd if=", "shred ", "diskutil erasedisk",
    "chmod 777 /", "chown -r root",
}

# Commands that require approval — potentially destructive.
# Stored lowercase; the input command is lowercased before matching.
_HIGH_RISK_COMMANDS = {
    "rm ", "rmdir", "git reset --hard", "git push --force", "git push -f",
    "git clean -fd", "git branch -d", "git rebase",
    "sudo ", "su ", "pip install", "npm install", "brew install",
    "kill -9", "killall", "pkill",
    "curl ", "wget ", "scp ", "rsync ",
}

# Tools that are inherently high-risk
_HIGH_RISK_TOOLS = {"file_delete", "process_kill", "git_reset", "git_push"}
_CRITICAL_TOOLS = {"format_disk", "wipe_disk"}


class SafetyGate:
    """Evaluates tool calls for risk and enforces approval requirements.

    The gate is the trust boundary. It does not execute tools — it only
    decides whether execution is permitted. The ToolExecutor calls the
    gate before every tool execution.
    """

    def __init__(
        self,
        *,
        auto_approve_low: bool = True,
        auto_approve_medium: bool = True,
        approval_callback: Callable[[RiskLevel, str, dict[str, Any]], bool] | None = None,
    ):
        self.auto_approve_low = auto_approve_low
        self.auto_approve_medium = auto_approve_medium
        self._approval_callback = approval_callback
        self._violations: list[SafetyViolation] = []

    def evaluate(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        tool_risk: RiskLevel = RiskLevel.LOW,
    ) -> SafetyDecision:
        """Evaluate whether a tool call should be allowed.

        Args:
            tool_name: The name of the tool being called.
            args: The arguments to the tool.
            tool_risk: The tool's declared risk level (from its schema).

        Returns:
            SafetyDecision indicating whether the call is allowed and
            whether it requires user approval.
        """
        # Check for hard-blocked commands
        if tool_name in ("shell", "run_command", "git"):
            cmd = str(args.get("command", args.get("subcmd", "")))
            cmd_lower = cmd.lower().strip()
            for blocked in _BLOCKED_COMMANDS:
                if blocked in cmd_lower:
                    violation = SafetyViolation(
                        level=RiskLevel.CRITICAL, tool_name=tool_name,
                        reason=f"blocked command: {blocked}", args=args,
                    )
                    self._violations.append(violation)
                    log.warning("safety.blocked", tool=tool_name, cmd=cmd, reason=blocked)
                    return SafetyDecision(
                        allowed=False, risk=RiskLevel.CRITICAL,
                        reason=f"blocked command: contains '{blocked}'",
                    )

            # Check for high-risk commands
            for high_risk in _HIGH_RISK_COMMANDS:
                if high_risk in cmd_lower:
                    risk = RiskLevel.HIGH
                    if self._needs_approval(risk):
                        return SafetyDecision(
                            allowed=True, risk=risk,
                            requires_approval=True,
                            reason=f"high-risk command: contains '{high_risk}'",
                        )
                    return SafetyDecision(
                        allowed=False, risk=risk,
                        reason=f"high-risk command requires approval: '{high_risk}'",
                    )

        # Check tool-level risk
        if tool_name in _CRITICAL_TOOLS:
            risk = RiskLevel.CRITICAL
        elif tool_name in _HIGH_RISK_TOOLS:
            risk = RiskLevel.HIGH
        else:
            risk = tool_risk

        # Auto-approve low and medium if configured
        if risk <= RiskLevel.LOW and self.auto_approve_low:
            return SafetyDecision(allowed=True, risk=risk)
        if risk == RiskLevel.MEDIUM and self.auto_approve_medium:
            return SafetyDecision(allowed=True, risk=risk)

        # High and critical always need approval
        if self._needs_approval(risk):
            return SafetyDecision(
                allowed=True, risk=risk,
                requires_approval=True,
                reason=f"{risk.name} risk operation requires approval",
            )

        return SafetyDecision(
            allowed=False, risk=risk,
            reason=f"{risk.name} risk operation not approved",
        )

    def _needs_approval(self, risk: RiskLevel) -> bool:
        """Check if a risk level needs approval and we have a callback."""
        if risk < RiskLevel.HIGH:
            return False
        return self._approval_callback is not None

    def request_approval(
        self,
        risk: RiskLevel,
        tool_name: str,
        args: dict[str, Any],
    ) -> bool:
        """Request user approval for a high-risk operation."""
        if self._approval_callback is None:
            log.warning("safety.no_callback", tool=tool_name, risk=risk.name)
            return False
        try:
            approved = self._approval_callback(risk, tool_name, args)
            log.info("safety.approval", tool=tool_name, risk=risk.name, approved=approved)
            return approved
        except Exception as e:
            log.error("safety.approval.error", tool=tool_name, error=str(e))
            return False

    @property
    def violations(self) -> list[SafetyViolation]:
        """Recorded safety violations from this session."""
        return list(self._violations)

    def reset(self) -> None:
        self._violations.clear()
