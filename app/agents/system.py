"""System agent — reports on macOS hardware telemetry and process state."""

from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.telemetry.system_telemetry import SystemTelemetry


class SystemAgent(Agent):
    name = "system"
    description = "Reports macOS hardware telemetry: CPU, memory, GPU, thermal, battery, processes."

    def plan(self, goal: str) -> list[str]:
        return ["Collect a telemetry snapshot", "Format it for the user"]

    def execute(self, goal: str) -> AgentResult:
        telemetry = SystemTelemetry()
        snapshot = telemetry.snapshot()
        if self.planner and goal.strip().lower() not in {"status", "telemetry", "snapshot", ""}:
            from app.models.types import Message

            system = (
                "You are a system agent. Answer the user's question using the provided "
                "macOS telemetry snapshot. Be concise."
            )
            user = f"Telemetry:\n{snapshot}\n\nQuestion: {goal}"
            resp = self.planner.chat([Message("system", system), Message("user", user)])
            return AgentResult(ok=True, answer=resp.text, data={"snapshot": snapshot})
        return AgentResult(ok=True, answer=str(snapshot), data={"snapshot": snapshot})

    def health(self) -> bool:
        return True
