"""Filesystem agent — locates and reads files via tools + retrieval."""

from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.models.types import Message


class FilesystemAgent(Agent):
    name = "filesystem"
    description = "Finds and reads local files using indexed content and the file_read tool."

    def plan(self, goal: str) -> list[str]:
        return [
            "Search the index for file content matching the goal",
            "Read the most relevant file via file_read if needed",
            "Answer with the file contents and path",
        ]

    def execute(self, goal: str) -> AgentResult:
        if not self.retriever:
            return AgentResult(ok=False, error="filesystem agent needs a retriever")
        results = self.retriever.search(goal, top_k=3)
        if not results:
            return AgentResult(ok=True, answer="No indexed files matched the query.")
        answer_parts: list[str] = []
        for r in results:
            answer_parts.append(f"[{r.source}] (score={r.score:.2f}):\n{r.text[:400]}")
        if self.planner:
            system = "You are a filesystem agent. Summarize the matching file snippets concisely."
            user = f"Question: {goal}\n\nSnippets:\n" + "\n\n".join(answer_parts)
            resp = self.planner.chat([Message("system", system), Message("user", user)])
            return AgentResult(ok=True, answer=resp.text, data={"sources": [r.source for r in results]})
        return AgentResult(ok=True, answer="\n\n".join(answer_parts), data={"sources": [r.source for r in results]})

    def health(self) -> bool:
        return self.retriever is not None
