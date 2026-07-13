"""Markdown indexer — splits on headers, preserving section context."""

from __future__ import annotations

import re
from pathlib import Path

from app.log import get_logger
from app.retrieval.chunking import chunk_text
from app.retrieval.indexers.base import Indexer

log = get_logger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class MarkdownIndexer(Indexer):
    name = "markdown"

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def supports(self, source: str) -> bool:
        return Path(source).is_file() and Path(source).suffix.lower() == ".md"

    def index(self, source: str) -> list[dict]:
        path = Path(source)
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Track the current header chain as section context.
        sections = self._split_by_headers(text)
        chunks: list[dict] = []
        for section_title, section_text in sections:
            file_chunks = chunk_text(
                section_text,
                source=str(path),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                metadata={"section": section_title, "ext": ".md", "kind": "markdown"},
            )
            chunks.extend({"text": c.text, "metadata": c.to_metadata()} for c in file_chunks)
        log.info("indexer.markdown.done", path=str(path), chunks=len(chunks))
        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        if not text.strip():
            return []
        positions: list[tuple[int, str]] = [(0, "")]
        for m in _HEADER_RE.finditer(text):
            positions.append((m.start(), m.group(2).strip()))
        sections: list[tuple[str, str]] = []
        for i, (pos, title) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[pos:end].strip()
            if body:
                sections.append((title, body))
        return sections
