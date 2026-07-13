"""PDF indexer using pypdf."""

from __future__ import annotations

from pathlib import Path

from app.log import get_logger
from app.retrieval.chunking import chunk_text
from app.retrieval.indexers.base import Indexer

log = get_logger(__name__)


class PdfIndexer(Indexer):
    name = "pdf"

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def supports(self, source: str) -> bool:
        return Path(source).is_file() and Path(source).suffix.lower() == ".pdf"

    def index(self, source: str) -> list[dict]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pypdf not installed") from e

        path = Path(source)
        reader = PdfReader(str(path))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                log.warning("indexer.pdf.page_failed", path=str(path), page=i, error=str(e))
                pages.append("")
        text = "\n\n".join(pages)
        file_chunks = chunk_text(
            text,
            source=str(path),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={"ext": ".pdf", "kind": "pdf", "pages": len(reader.pages)},
        )
        return [{"text": c.text, "metadata": c.to_metadata()} for c in file_chunks]
