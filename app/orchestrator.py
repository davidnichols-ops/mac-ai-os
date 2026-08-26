"""Agent Orchestrator.

The orchestrator sits between the Conversation API and the agents/tools.
It:

  1. routes the goal to an agent via keyword matching (instant, no model call)
  2. delegates execution to that agent
  3. (optionally) lets the planner call tools via the ToolRouter
  4. returns a unified result

For v2 models, the planner self-selects FAST/THINK mode via control tokens.
The orchestrator does NOT do mode selection — it lets the model decide.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import Agent, AgentResult
from app.log import get_logger
from app.models.base import Model
from app.tools.registry import ToolRouter

log = get_logger(__name__)

_DEFAULT_ROUTE_HINTS: dict[str, list[str]] = {
    "general": ["hello", "hi ", "hey", "h!", "greetings", "how are you", "thanks", "thank you", "bye", "goodbye", "help", "what can you do", "who are you", "what is this"],
    "session": ["what was I working", "last session", "before I stopped", "what did I do", "what happened", "working on", "session summary", "recall", "what was going on"],
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
        """Pick an agent name for the goal.

        Keyword-based routing only. The planner is NOT used for routing
        because that would require a full 9B model generation (~17s)
        just to pick an agent name. Keyword matching + fallback to
        "general" is instant and sufficient.
        """
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
        # No keyword match — use the general agent (direct planner call, no RAG).
        return "general" if "general" in self.agents else next(iter(self.agents))

    def run(self, goal: str, *, enable_thinking: bool | None = None) -> AgentResult:
        import time
        t0 = time.monotonic()
        name = self.route(goal)
        t_route = time.monotonic()
        agent = self.agents.get(name)
        if agent is None:
            return AgentResult(ok=False, error=f"no agent named {name}")
        log.info("orchestrator.run", agent=name, goal=goal[:80], thinking=enable_thinking,
                 route_ms=round((t_route - t0) * 1000))
        # Pass thinking mode to the agent if it supports it
        try:
            result = agent.execute(goal, enable_thinking=enable_thinking)
        except TypeError:
            # Agent doesn't accept enable_thinking kwarg
            result = agent.execute(goal)
        t_exec = time.monotonic()
        log.info("orchestrator.done", agent=name,
                 route_ms=round((t_route - t0) * 1000),
                 exec_ms=round((t_exec - t_route) * 1000),
                 total_ms=round((t_exec - t0) * 1000))
        result.steps = [f"routed to: {name}", *result.steps]
        return result

    def available_tools(self) -> list[dict[str, Any]]:
        return self.tools.describe() if self.tools else []

    def call_tool(self, name: str, args: dict[str, Any]):
        if not self.tools:
            return None
        return self.tools.execute(name, args)
