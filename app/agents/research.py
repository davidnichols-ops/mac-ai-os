"""Research agent — synthesizes answers from indexed docs/papers/research."""

from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.models.types import Message


class ResearchAgent(Agent):
    name = "research"
    description = "Retrieval-augmented Q&A over indexed docs, papers, and notes."

    def plan(self, goal: str) -> list[str]:
        return ["Retrieve top-k passages", "Synthesize a cited answer"]

    def execute(self, goal: str) -> AgentResult:
        if not self.retriever or not self.planner:
            return AgentResult(ok=False, error="research agent needs a retriever and planner")
        results = self.retriever.search(goal, top_k=5)
        if not results:
            return AgentResult(ok=True, answer="No indexed research material matched the query.")
        context = "\n\n".join(f"[{r.source}] (score={r.score:.2f})\n{r.text}" for r in results)
        system = (
            "You are a research agent. Answer the question from the retrieved context. "
            "Cite sources as [path]. Be precise and admit uncertainty."
        )
        user = f"Context:\n{context}\n\nQuestion: {goal}"
        resp = self.planner.chat([Message("system", system), Message("user", user)])
        return AgentResult(ok=True, answer=resp.text, data={"sources": [r.source for r in results]})

    def health(self) -> bool:
        return self.retriever is not None and self.planner is not None
