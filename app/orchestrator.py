"""Agent Orchestrator.

The orchestrator is the brain that sits between the Conversation API and the
agents/tools. It:

  1. asks the planner to classify the goal into an agent
  2. delegates execution to that agent
  3. (optionally) lets the planner call tools via the ToolRouter
  4. returns a unified result

For Phase 1 the routing is keyword + planner-assisted; it is intentionally
simple and replaceable.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import Agent, AgentResult
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message
from app.tools.registry import ToolRouter

log = get_logger(__name__)

_DEFAULT_ROUTE_HINTS: dict[str, list[str]] = {
    "git": ["git", "commit", "commits", "branch", "diff", "blame", "pushed", "what changed", "since yesterday", "last commit", "recent commits", "most modified"],
    "repository": ["repo", "codebase", "source code", "function", "class", "how does", "where is"],
    "filesystem": ["file", "folder", "path", "read", "find", "directory"],
    "research": ["paper", "docs", "documentation", "research", "study", "article", "notes"],
    "system": ["cpu", "memory", "battery", "thermal", "process", "telemetry", "system status", "hardware"],
}


class AgentOrchestrator:
    def __init__(
        self,
        *,
        planner: Model,
        agents: dict[str, Agent],
        tools: ToolRouter | None = None,
    ):
        self.planner = planner
        self.agents = agents
        self.tools = tools

    def route(self, goal: str) -> str:
        """Pick an agent name for the goal. Planner-assisted with keyword fallback."""
        lowered = goal.lower()
        scored: dict[str, int] = {name: 0 for name in self.agents}
        for name, hints in _DEFAULT_ROUTE_HINTS.items():
            if name not in scored:
                continue
            for h in hints:
                if h in lowered:
                    scored[name] += 1
        best = max(scored, key=scored.get)
        if scored[best] > 0:
            return best
        # Fall back to asking the planner (best-effort, single-shot).
        try:
            names = list(self.agents)
            system = (
                "You route a user goal to exactly one agent. "
                f"Respond with only one of: {', '.join(names)}."
            )
            resp = self.planner.chat([
                Message("system", system),
                Message("user", goal),
            ])
            choice = resp.text.strip().lower()
            for name in self.agents:
                if name in choice:
                    return name
        except Exception as e:
            log.warning("orchestrator.route.planner_failed", error=str(e))
        # Ultimate fallback: the research agent (RAG over everything).
        return "research" if "research" in self.agents else next(iter(self.agents))

    def run(self, goal: str) -> AgentResult:
        name = self.route(goal)
        agent = self.agents.get(name)
        if agent is None:
            return AgentResult(ok=False, error=f"no agent named {name}")
        log.info("orchestrator.run", agent=name, goal=goal[:80])
        result = agent.execute(goal)
        result.steps = [f"routed to: {name}", *result.steps]
        return result

    def available_tools(self) -> list[dict[str, Any]]:
        return self.tools.describe() if self.tools else []

    def call_tool(self, name: str, args: dict[str, Any]):
        if not self.tools:
            return None
        return self.tools.execute(name, args)
