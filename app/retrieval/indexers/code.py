"""Code indexer — chunks source files, tagging language from extension."""

from __future__ import annotations

from pathlib import Path

from app.log import get_logger
from app.retrieval.chunking import chunk_text
from app.retrieval.indexers.base import Indexer

log = get_logger(__name__)

_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".java": "java", ".kt": "kotlin",
    ".swift": "swift", ".rb": "ruby", ".sh": "shell", ".sql": "sql",
    ".html": "html", ".css": "css",
}


class CodeIndexer(Indexer):
    name = "code"

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def supports(self, source: str) -> bool:
        return Path(source).is_file() and Path(source).suffix.lower() in _EXT_TO_LANG

    def index(self, source: str) -> list[dict]:
        path = Path(source)
        text = path.read_text(encoding="utf-8", errors="ignore")
        lang = _EXT_TO_LANG.get(path.suffix.lower(), "unknown")
        file_chunks = chunk_text(
            text,
            source=str(path),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata={"language": lang, "ext": path.suffix, "kind": "code"},
        )
        chunks = [{"text": c.text, "metadata": c.to_metadata()} for c in file_chunks]
        log.info("indexer.code.done", path=str(path), language=lang, chunks=len(chunks))
        return chunks
