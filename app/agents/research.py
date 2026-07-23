"""Research agent — synthesizes answers from web search + indexed docs.

When the perception layer (AAFP CLI) is available, the agent first searches
the web for fresh information, then combines it with local RAG results from
indexed docs/papers/notes. If perception is unavailable, it falls back to
local-only RAG (the original behavior).
"""

from __future__ import annotations

from app.agents.base import Agent, AgentResult
from app.log import get_logger
from app.models.types import Message

log = get_logger(__name__)


class ResearchAgent(Agent):
    name = "research"
    description = "Web search + retrieval-augmented Q&A over indexed docs, papers, and notes."

    def plan(self, goal: str) -> list[str]:
        steps = ["Retrieve relevant passages from local index"]
        if self.perception and self.perception.available:
            steps.insert(0, "Search the web for fresh information")
        steps.append("Synthesize a cited answer combining web + local sources")
        return steps

    def execute(self, goal: str) -> AgentResult:
        if not self.planner:
            return AgentResult(ok=False, error="research agent needs a planner")

        web_context = ""
        web_sources: list[str] = []

        # Phase 1: Web search via AAFP perception (if available).
        if self.perception and self.perception.available:
            try:
                results = self.perception.search(goal, num=5)
                if results:
                    web_context = "\n\n".join(
                        f"[web: {r.title}] ({r.url})\n{r.snippet}"
                        for r in results
                    )
                    web_sources = [r.url for r in results]
                    log.info("research.web_search", query=goal, results=len(results))
            except Exception as e:
                log.warning("research.web_search.failed", error=str(e))

        # Phase 2: Local RAG over indexed content.
        local_context = ""
        local_sources: list[str] = []
        if self.retriever:
            results = self.retriever.search(goal, top_k=5)
            if results:
                local_context = "\n\n".join(
                    f"[local: {r.source}] (score={r.score:.2f})\n{r.text}"
                    for r in results
                )
                local_sources = [r.source for r in results]

        if not web_context and not local_context:
            return AgentResult(
                ok=True,
                answer="No web results and no indexed local material matched the query.",
            )

        # Phase 3: Synthesize with the LLM.
        context_parts = []
        if web_context:
            context_parts.append(f"=== Web Search Results ===\n{web_context}")
        if local_context:
            context_parts.append(f"=== Local Indexed Content ===\n{local_context}")
        full_context = "\n\n".join(context_parts)

        system = (
            "You are a research agent. Answer the question using the provided context. "
            "Cite web sources as [web: title] (url) and local sources as [local: path]. "
            "Be precise, admit uncertainty, and distinguish between web and local sources."
        )
        user = f"Context:\n{full_context}\n\nQuestion: {goal}"
        resp = self.planner.chat([Message("system", system), Message("user", user)])

        return AgentResult(
            ok=True,
            answer=resp.text,
            data={
                "web_sources": web_sources,
                "local_sources": local_sources,
            },
        )

    def health(self) -> bool:
        return self.planner is not None
