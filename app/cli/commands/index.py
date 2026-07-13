"""`maos index <path>` — index a folder or file into the vector store."""

from __future__ import annotations

from pathlib import Path

import typer

from app.db import get_db
from app.runtime import build_runtime
from app.ui.console import print_panel


def index(
    path: str = typer.Argument(..., help="Folder or file to index"),
    source_name: str = typer.Option(None, "--name", help="Override the source label"),
    force: bool = typer.Option(False, "--force", help="Re-index everything, ignoring content hashes"),
) -> None:
    """Index a folder or file into the retrieval backend (hash-aware, incremental)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        typer.echo(f"not found: {p}", err=True)
        raise typer.Exit(1)

    rt = build_runtime()
    from app.retrieval.pipeline import IndexingPipeline

    db = get_db(rt.settings.paths.data_dir + "/maos.db")
    pipeline = IndexingPipeline(
        rt.retriever,
        db,
        chunk_size=rt.settings.retrieval.chunk_size,
        chunk_overlap=rt.settings.retrieval.chunk_overlap,
    )
    label = source_name or str(p)
    stats = pipeline.index_source(label, force=force)

    total = rt.retriever.count() if hasattr(rt.retriever, "count") else "?"
    print_panel(
        f"Path: {p}\n"
        f"Files indexed: {stats.indexed}\n"
        f"Files skipped (unchanged): {stats.skipped}\n"
        f"Chunks added: {stats.chunks_added}\n"
        f"Errors: {stats.errors}\n"
        f"Collection total: {total} chunks",
        title="Index",
    )
