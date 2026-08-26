"""General agent — handles greetings, simple questions, and chit-chat.

This is the lightest agent: no RAG, no retrieval, no tools. It just calls
the planner directly with the system prompt. Used for greetings, simple
factual questions, and anything that doesn't fit a domain-specific agent.

For v2 models, the planner self-selects FAST/THINK mode via control tokens.
The general agent passes enable_thinking=None to let the model decide.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.models.types import Message


class GeneralAgent(Agent):
    name = "general"
    description = "General conversation, greetings, and simple questions."

    def plan(self, goal: str) -> list[str]:
        return ["Respond directly"]

    def execute(self, goal: str, *, enable_thinking: bool | None = None) -> AgentResult:
        if not self.planner:
            return AgentResult(ok=False, error="general agent needs a planner")
        system = (
            "You are Mac AI OS, a local-first intelligence layer for macOS. "
            "Answer concisely and accurately."
        )
        kwargs: dict = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        resp = self.planner.chat(
            [Message("system", system), Message("user", goal)],
            **kwargs,
        )
        return AgentResult(ok=True, answer=resp.text)

    def health(self) -> bool:
        return self.planner is not None
