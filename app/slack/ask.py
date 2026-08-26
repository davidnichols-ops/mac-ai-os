"""Slack RAG+Q&A — hidden retrieval, model-in-the-middle response.

The user asks a question in Slack. We:
  1. Search the local MAOS index for relevant passages (hidden from the user).
  2. Build a context prompt from those passages.
  3. Send it to the configured model (default: OpenRouter Nemotron free).
  4. Return the model's answer.

The retrieved context is never shown to the user — only the synthesized
response is returned, which is what "man in the middle" means here.
"""

from __future__ import annotations

from typing import Any

from app.context import build_context
from app.db import Database
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message
from app.retrieval.base import Retriever

log = get_logger(__name__)


def answer_question(
    query: str,
    db: Database,
    *,
    retriever: Retriever | None = None,
    model: Model | None = None,
    top_k: int = 5,
    max_context_chars: int = 4000,
) -> dict[str, Any]:
    """Answer a user question with hidden RAG and a model in the middle.

    Returns a dict: {"ok": bool, "text": str, "sources": [...]}.
    """
    if not query:
        return {"ok": False, "text": "Usage: ask <question>"}

    # 1. Hidden RAG retrieval.
    context_text = ""
    sources: list[dict[str, Any]] = []
    try:
        if retriever:
            results = retriever.search(query, top_k=top_k)
            if results:
                chunks: list[str] = []
                for r in results:
                    chunks.append(f"[{r.source}]\n{r.text[:800]}")
                    sources.append({"source": r.source, "score": r.score})
                context_text = "\n\n".join(chunks)[:max_context_chars]
        else:
            # Fallback to MAOS Context builder if no retriever.
            ctx = build_context(db, query, retriever=None, lookback_hours=24, top_k=top_k)
            context_text = ctx.to_prompt()[:max_context_chars]
    except Exception as e:
        log.warning("slack.ask.retrieval_failed", error=str(e))

    # 2. No model available — show raw RAG snippets (backwards-compatible).
    if model is None:
        if sources:
            lines = [f"🔍 *Results for: {query[:60]}*\n"]
            for s in sources[:3]:
                lines.append(f"_{s['source']}_")
            return {"ok": True, "text": "\n".join(lines), "sources": [s["source"] for s in sources]}
        return {
            "ok": False,
            "text": (
                "No Q&A model configured. Set OPENROUTER_API_KEY to enable "
                "Nemotron answers, or configure `runtime.qa_model`."
            ),
        }

    # 3. Model-in-the-middle prompt.
    system = (
        "You are a concise engineering assistant for a Mac AI OS operator. "
        "Use the provided context to answer the question. "
        "Do not mention that you used retrieval or reference sources by name. "
        "Keep answers short and direct."
    )
    if context_text:
        system += f"\n\nContext (do not quote verbatim):\n{context_text}"

    try:
        resp = model.chat([
            Message("system", system),
            Message("user", query),
        ])
        answer = resp.text.strip() if resp.text else "(no response)"
        return {"ok": True, "text": answer, "sources": [s["source"] for s in sources]}
    except Exception as e:
        log.error("slack.ask.model_failed", error=str(e))
        return {"ok": False, "text": f"Model error: {e}"}
