"""`maos rag "..."` — retrieval-augmented question over indexed content."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel


def rag(
    question: str = typer.Argument(..., help="Question to answer from the index"),
    top_k: int = typer.Option(5, help="Number of passages to retrieve"),
) -> None:
    """Answer a question using retrieval-augmented generation."""
    rt = build_runtime()
    results = rt.retriever.search(question, top_k=top_k)
    if not results:
        print_panel("Nothing indexed yet. Run `maos index <path>` first.", title="RAG")
        raise typer.Exit(1)

    from app.models.types import Message

    context = "\n\n".join(f"[{r.source}] (score={r.score:.2f})\n{r.text}" for r in results)
    system = (
        "You are Mac AI OS. Answer the question using only the retrieved context. "
        "Cite sources as [path]. If the context is insufficient, say so explicitly."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = rt.planner.chat([Message("system", system), Message("user", user)])

    sources = "\n".join(f"- {r.source} (score={r.score:.2f})" for r in results)
    print_panel(sources, title="Sources")
    print_markdown(resp.text)
