"""Tool executor — tool execution with safety gates and risk evaluation.

The ToolExecutor is the execution layer between the model's tool calls and
the actual tool implementations. It:

1. Evaluates safety risk via SafetyGate
2. Requests user approval for high-risk operations
3. Executes the tool via the existing ToolRouter
4. Formats results for the model
5. Records everything in the trajectory

The model decides what to do. The ToolExecutor decides how safely to do it.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.safety import RiskLevel, SafetyGate
from app.core.trajectory import TrajectoryRecorder
from app.log import get_logger
from app.tools.base import ToolResult
from app.tools.registry import ToolRouter

log = get_logger(__name__)

# Default risk levels for known tools
_DEFAULT_RISK: dict[str, RiskLevel] = {
    "shell": RiskLevel.MEDIUM,
    "run_command": RiskLevel.MEDIUM,
    "file_read": RiskLevel.LOW,
    "read_file": RiskLevel.LOW,
    "list_files": RiskLevel.LOW,
    "file_write": RiskLevel.MEDIUM,
    "write_file": RiskLevel.MEDIUM,
    "edit_file": RiskLevel.MEDIUM,
    "file_delete": RiskLevel.HIGH,
    "file_move": RiskLevel.MEDIUM,
    "git": RiskLevel.MEDIUM,
    "git_status": RiskLevel.LOW,
    "git_diff": RiskLevel.LOW,
    "git_log": RiskLevel.LOW,
    "git_commit": RiskLevel.MEDIUM,
    "git_push": RiskLevel.HIGH,
    "git_reset": RiskLevel.HIGH,
    "web_search": RiskLevel.LOW,
    "web_browse": RiskLevel.LOW,
    "rag_search": RiskLevel.LOW,
    "system_status": RiskLevel.LOW,
    "process_list": RiskLevel.LOW,
    "process_kill": RiskLevel.HIGH,
}


class ToolExecutor:
    """Executes tool calls with safety evaluation and trajectory recording.

    Wraps the existing ToolRouter with:
    - Safety gate (risk classification + approval)
    - Trajectory recording (every call is logged)
    - Result formatting (truncation, error handling)
    """

    def __init__(
        self,
        router: ToolRouter,
        *,
        safety: SafetyGate | None = None,
        trajectory: TrajectoryRecorder | None = None,
        max_result_chars: int = 8000,
    ):
        self.router = router
        self.safety = safety or SafetyGate()
        self.trajectory = trajectory
        self.max_result_chars = max_result_chars

    def execute(
        self,
        name: str,
        args: dict[str, Any],
    ) -> tuple[str, bool, str | None]:
        """Execute a tool call and return formatted result.

        Returns:
            (result_text, ok, error) where:
            - result_text: the tool output formatted for the model
            - ok: whether the tool succeeded
            - error: error message if failed, None if succeeded
        """
        # Evaluate safety
        tool_risk = self._get_risk_level(name)
        decision = self.safety.evaluate(name, args, tool_risk=tool_risk)

        if not decision.allowed:
            error = f"Safety block: {decision.reason}"
            log.warning("tool_executor.blocked", tool=name, reason=decision.reason)
            if self.trajectory:
                self.trajectory.record_tool_call(name, args, risk_level=decision.risk.name)
                self.trajectory.record_tool_result(None, ok=False, error=error)
            return (error, False, error)

        # Request approval if needed
        if decision.requires_approval:
            approved = self.safety.request_approval(decision.risk, name, args)
            if not approved:
                error = f"User denied approval for {name} ({decision.risk.name} risk)"
                log.info("tool_executor.denied", tool=name, risk=decision.risk.name)
                if self.trajectory:
                    self.trajectory.record_tool_call(name, args, risk_level=decision.risk.name)
                    self.trajectory.record_tool_result(None, ok=False, error=error, approved=False)
                return (error, False, error)
            if self.trajectory:
                self.trajectory.trajectory.approvals_requested += 1

        # Record the tool call in trajectory
        if self.trajectory:
            self.trajectory.record_tool_call(name, args, risk_level=decision.risk.name)

        # Execute the tool
        try:
            result = self.router.execute(name, args)
            result_text = self._format_result(result)
            ok = result.ok
            error = result.error

            if self.trajectory:
                self.trajectory.record_tool_result(
                    result.output, ok=ok, error=error, approved=decision.requires_approval or None,
                )

            log.info(
                "tool_executor.executed",
                tool=name, ok=ok, risk=decision.risk.name,
                result_chars=len(result_text),
            )
            return (result_text, ok, error)

        except Exception as e:
            error = f"Tool execution error: {e}"
            log.exception("tool_executor.exception", tool=name)
            if self.trajectory:
                self.trajectory.record_tool_result(None, ok=False, error=error)
            return (error, False, error)

    def _get_risk_level(self, tool_name: str) -> RiskLevel:
        """Get the default risk level for a tool."""
        return _DEFAULT_RISK.get(tool_name, RiskLevel.MEDIUM)

    def _format_result(self, result: ToolResult) -> str:
        """Format a ToolResult as text for the model."""
        parts: list[str] = []
        if result.output is not None:
            output = str(result.output)
            if len(output) > self.max_result_chars:
                output = output[: self.max_result_chars] + f"\n... (truncated, {len(output)} total chars)"
            parts.append(output)
        if result.error:
            parts.append(f"[error] {result.error}")
        if result.metadata:
            meta = {k: v for k, v in result.metadata.items() if k in ("returncode", "exit_code", "bytes")}
            if meta:
                parts.append(f"[metadata] {json.dumps(meta)}")
        return "\n".join(parts) if parts else "(no output)"

    def available_tools_schema(self) -> list[dict[str, Any]]:
        """Return tool schemas in the format the model expects."""
        return self.router.schemas()
