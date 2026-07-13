"""Text chunking for indexing.

Simple, dependency-free, recursive-character-style splitting with overlap.
Good enough for markdown / code / plain text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str  # file path or uri the chunk came from
    index: int   # position within the source
    metadata: dict

    def to_metadata(self) -> dict:
        return {**self.metadata, "source": self.source, "chunk_index": self.index}


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping chunks bounded by `chunk_size` characters."""
    if not text:
        return []
    metadata = metadata or {}
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, index=idx, metadata=dict(metadata)))
            idx += 1
        if end == n:
            break
        start = end - chunk_overlap if end - chunk_overlap > start else end
    return chunks
