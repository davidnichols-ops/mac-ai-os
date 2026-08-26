"""`maos rag` and `maos index` — backed by nevernote's RAG pipeline.

This replaces the old ChromaDB-based retrieval with nevernote's production
RAG engine (query expansion, hybrid search, reranking, streaming). The
nevernote Next.js service must be running on localhost:3000.

The bridge module lives at ~/Projects/nevernote/nevernote_rag.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from app.ui.console import print_markdown, print_panel

# Path to the nevernote_rag bridge module
NEVERNOTE_RAG_PATH = Path.home() / "Projects/nevernote/nevernote_rag.py"


def _get_rag_client():
    """Import and return the NevernoteRAG client, or exit with guidance."""
    if not NEVERNOTE_RAG_PATH.exists():
        print_panel(
            "nevernote_rag.py not found at ~/Projects/nevernote/nevernote_rag.py\n"
            "Clone nevernote and ensure the bridge module is installed.",
            title="RAG Bridge Missing",
        )
        raise typer.Exit(1)

    sys.path.insert(0, str(NEVERNOTE_RAG_PATH.parent))
    from nevernote_rag import NevernoteRAG

    return NevernoteRAG()


def rag(
    question: str = typer.Argument(..., help="Question to answer from the index"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of passages to retrieve"),
) -> None:
    """Answer a question using nevernote's RAG pipeline (query expansion + reranking)."""
    client = _get_rag_client()
    try:
        result = client.query(question, k=top_k)
    except Exception as e:
        print_panel(f"Query failed: {e}\nIs nevernote running on localhost:3000?", title="RAG Error")
        raise typer.Exit(1) from e

    answer = result.get("answer", "no answer returned")
    sources = result.get("sources", [])

    if sources:
        source_lines = []
        for i, s in enumerate(sources, 1):
            dist = s.get("distance", 0)
            title = s.get("title", "?")
            source_lines.append(f"[{i}] {title} (dist={dist:.2f})")
        print_panel("\n".join(source_lines), title="Sources")

    print_markdown(answer)


def index(
    path: str = typer.Argument(..., help="Folder or file to index"),
    max_files: int = typer.Option(200, "--max-files", help="Maximum files to index this run"),
) -> None:
    """Index a workspace into nevernote's RAG pipeline."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        typer.echo(f"not found: {p}", err=True)
        raise typer.Exit(1)

    client = _get_rag_client()

    if p.is_file():
        ok = client.ingest_text(p.name, p.read_text(encoding="utf-8", errors="replace"))
        print_panel(
            f"File: {p}\nIndexed: {'yes' if ok else 'failed'}",
            title="Index",
        )
        return

    stats = client.index_workspace(p, max_files=max_files)
    print_panel(
        f"Path: {p}\n"
        f"Scanned: {stats.scanned}\n"
        f"Indexed: {stats.indexed}\n"
        f"Skipped: {stats.skipped}\n"
        f"Failed: {stats.failed}\n"
        f"Size: {stats.bytes_indexed / 1024:.1f}KB",
        title="Index",
    )
    if stats.errors:
        typer.echo("\nErrors:", err=True)
        for e in stats.errors[:5]:
            typer.echo(f"  {e}", err=True)


def rag_status() -> None:
    """Show nevernote RAG index status."""
    client = _get_rag_client()
    try:
        stats = client.status()
    except Exception as e:
        print_panel(f"Status failed: {e}", title="RAG Error")
        raise typer.Exit(1) from e

    kb = stats.get("knowledgeBase", {})
    embeddings = stats.get("embeddings", {})
    llm = stats.get("llm", {})
    chroma = stats.get("chroma", {})

    print_panel(
        f"Total docs: {kb.get('totalDocs', '?')}\n"
        f"Chroma: {'ok' if chroma.get('ok') else 'down'} ({chroma.get('count', '?')} chunks)\n"
        f"Embeddings: {embeddings.get('provider', '?')} / {embeddings.get('model', '?')} ({embeddings.get('dimensions', '?')}d)\n"
        f"LLM: {llm.get('provider', '?')} / {llm.get('model', '?')}",
        title="Nevernote RAG Status",
    )
