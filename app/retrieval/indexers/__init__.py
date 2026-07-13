"""Indexer package + a dispatcher that picks the right indexer per source."""

from __future__ import annotations

from pathlib import Path

from app.retrieval.indexers.base import Indexer
from app.retrieval.indexers.code import CodeIndexer
from app.retrieval.indexers.folder import FolderIndexer
from app.retrieval.indexers.git import GitIndexer
from app.retrieval.indexers.markdown import MarkdownIndexer
from app.retrieval.indexers.pdf import PdfIndexer

__all__ = [
    "Indexer",
    "FolderIndexer",
    "MarkdownIndexer",
    "CodeIndexer",
    "PdfIndexer",
    "GitIndexer",
    "build_indexer",
    "dispatch",
]


def build_indexers(
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    db=None,
) -> list[Indexer]:
    """Ordered list of indexers; the first that supports a source wins."""
    indexers: list[Indexer] = [
        FolderIndexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        MarkdownIndexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        CodeIndexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        PdfIndexer(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    ]
    if db is not None:
        indexers.append(GitIndexer(db))
    return indexers


def dispatch(source: str, indexers: list[Indexer]) -> list[dict]:
    """Pick the first indexer that supports `source` and run it."""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(source)
    for idx in indexers:
        if idx.supports(source):
            return idx.index(source)
    # Fallback: treat as plain text.
    text = p.read_text(encoding="utf-8", errors="ignore")
    from app.retrieval.chunking import chunk_text

    chunks = chunk_text(text, source=str(p), metadata={"kind": "text"})
    return [{"text": c.text, "metadata": c.to_metadata()} for c in chunks]
