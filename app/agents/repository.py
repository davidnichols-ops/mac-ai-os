"""Repository agent — answers questions about a git repo using retrieval."""

from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.models.types import Message


class RepositoryAgent(Agent):
    name = "repository"
    description = "Answers questions about a git repository using indexed source + git metadata."

    def plan(self, goal: str) -> list[str]:
        return [
            f"Retrieve relevant code/docs for: {goal}",
            "Summarize findings with file references",
            "Optionally run a git command for extra context",
        ]

    def execute(self, goal: str) -> AgentResult:
        if not self.retriever or not self.planner:
            return AgentResult(ok=False, error="repository agent needs a retriever and planner")
        results = self.retriever.search(goal, top_k=5)
        if not results:
            return AgentResult(ok=True, answer="No indexed repository content matched the query.")
        context = "\n\n".join(
            f"[{r.source}] (score={r.score:.2f})\n{r.text}" for r in results
        )
        system = (
            "You are a repository agent. Answer using only the provided context. "
            "Cite sources as [path]. If the context is insufficient, say so."
        )
        user = f"Context:\n{context}\n\nQuestion: {goal}"
        resp = self.planner.chat([Message("system", system), Message("user", user)])
        return AgentResult(
            ok=True,
            answer=resp.text,
            steps=self.plan(goal),
            data={"sources": [r.source for r in results]},
        )

    def health(self) -> bool:
        return self.retriever is not None and self.planner is not None
