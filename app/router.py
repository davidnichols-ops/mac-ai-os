"""Intent router — classifies user queries to select the right agent.

Uses the small utility model (3B) for fast, cheap classification instead of
the large planner model. Falls back to keyword matching if the utility model
is unavailable.

The router logs every routing decision to the decisions table for auditability.
"""

from __future__ import annotations

from app.db import Database
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message

log = get_logger(__name__)

# Keyword hints for fallback routing.
_ROUTE_HINTS: dict[str, list[str]] = {
    "session": [
        "what was I working", "last session", "before I stopped", "what did I do",
        "what happened", "working on", "session summary", "recall",
        "what was going on", "get me ready", "continue where",
    ],
    "git": [
        "git", "commit", "commits", "branch", "diff", "blame", "pushed",
        "what changed", "since yesterday", "last commit", "recent commits",
        "most modified",
    ],
    "repository": [
        "repo", "codebase", "source code", "function", "class", "how does",
        "where is", "explain this", "what does this code",
    ],
    "filesystem": ["file", "folder", "path", "read", "find", "directory", "list files"],
    "research": ["paper", "docs", "documentation", "research", "study", "article", "notes"],
    "system": [
        "cpu", "memory", "battery", "thermal", "process", "telemetry",
        "system status", "hardware", "why is", "performance",
    ],
}


def _keyword_route(goal: str, agents: list[str]) -> tuple[str, float]:
    """Route using keyword matching. Returns (agent_name, confidence)."""
    lowered = goal.lower()
    scored: dict[str, int] = {name: 0 for name in agents}
    for name, hints in _ROUTE_HINTS.items():
        if name not in scored:
            continue
        for h in hints:
            if h in lowered:
                scored[name] += 1
    best = max(scored, key=scored.get)
    if scored[best] > 0:
        # Confidence: more matches = higher confidence.
        confidence = min(0.5 + scored[best] * 0.15, 0.95)
        return best, confidence
    return "general", 0.3  # low confidence fallback


def _llm_route(
    goal: str, agents: list[str], utility: Model,
) -> tuple[str, float, str]:
    """Route using the utility model. Returns (agent_name, confidence, reasoning)."""
    agent_list = ", ".join(agents)
    system = (
        "You are an intent router for a local AI operating system. "
        f"Classify the user's query into exactly one agent: {agent_list}.\n\n"
        "Respond in this exact format:\n"
        "agent: <name>\n"
        "confidence: <0.0-1.0>\n"
        "reasoning: <one sentence>\n\n"
        "Rules:\n"
        "- 'general' agent: general conversation, math, reasoning, knowledge questions "
        "that don't fit other agents.\n"
        "- 'session' agent: questions about what the user was working on, session history, "
        "getting ready to continue, recent activity.\n"
        "- 'git' agent: git operations, commits, branches, diffs.\n"
        "- 'repository' agent: questions about code, functions, classes, how things work.\n"
        "- 'filesystem' agent: file operations, reading, finding files.\n"
        "- 'research' agent: papers, documentation, research, notes.\n"
        "- 'system' agent: CPU, memory, battery, performance, hardware.\n"
    )
    resp = utility.chat([
        Message("system", system),
        Message("user", goal),
    ])
    text = resp.text.strip()

    # Parse the response.
    agent = ""
    confidence = 0.5
    reasoning = ""

    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("agent:"):
            agent = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("confidence:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    # Validate agent name.
    for a in agents:
        if a in agent:
            return a, confidence, reasoning

    # If parsing failed, fall back.
    log.warning("router.llm_parse_failed", response=text[:200])
    return "general", 0.3, "LLM response parsing failed"


class IntentRouter:
    """Routes user queries to the appropriate agent using the utility model."""

    def __init__(
        self,
        utility: Model | None = None,
        db: Database | None = None,
    ):
        self.utility = utility
        self.db = db

    def route(self, goal: str, agents: list[str]) -> tuple[str, float, str]:
        """Select an agent for the goal. Returns (agent, confidence, reasoning)."""
        # Try LLM routing first if utility model is available.
        if self.utility:
            try:
                agent, confidence, reasoning = _llm_route(goal, agents, self.utility)
                self._log_decision(goal, agent, confidence, reasoning, method="llm")
                return agent, confidence, reasoning
            except Exception as e:
                log.warning("router.llm_failed", error=str(e), fallback="keyword")

        # Fall back to keyword routing.
        agent, confidence = _keyword_route(goal, agents)
        reasoning = f"keyword match (score-based, confidence={confidence:.2f})"
        self._log_decision(goal, agent, confidence, reasoning, method="keyword")
        return agent, confidence, reasoning

    def _log_decision(
        self, query: str, agent: str, confidence: float, reasoning: str, method: str,
    ) -> None:
        log.info("router.decision", agent=agent, confidence=confidence,
                 method=method, query=query[:80])
        if self.db:
            self.db.log_decision(
                agent="IntentRouter",
                decision=f"route to {agent}",
                reasoning=reasoning,
                confidence=confidence,
                query=query[:500],
            )
