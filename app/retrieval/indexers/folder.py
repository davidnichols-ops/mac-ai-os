"""Folder indexer — walks a directory and dispatches each file to a sub-indexer."""

from __future__ import annotations

import os
from pathlib import Path

from app.log import get_logger
from app.retrieval.chunking import chunk_text
from app.retrieval.indexers.base import Indexer

log = get_logger(__name__)

_TEXT_EXTS = {".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".toml", ".yaml", ".yml"}
_CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".h", ".cpp", ".hpp",
    ".java", ".kt", ".swift", ".rb", ".sh", ".bash", ".zsh", ".sql", ".html", ".css",
}
_IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", "dist", "build"}
_IGNORE_NAMES = {".DS_Store"}
_MAX_FILE_BYTES = 256 * 1024


class FolderIndexer(Indexer):
    name = "folder"

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def supports(self, source: str) -> bool:
        return Path(source).is_dir()

    def index(self, source: str) -> list[dict]:
        root = Path(source).resolve()
        chunks: list[dict] = []
        for path in self._walk(root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                log.warning("indexer.folder.read_failed", path=str(path), error=str(e))
                continue
            if not text.strip():
                continue
            rel = str(path.relative_to(root))
            file_chunks = chunk_text(
                text,
                source=str(path),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                metadata={"rel_path": rel, "ext": path.suffix, "kind": "file"},
            )
            chunks.extend({"text": c.text, "metadata": c.to_metadata()} for c in file_chunks)
        log.info("indexer.folder.done", root=str(root), chunks=len(chunks))
        return chunks

    def _walk(self, root: Path):
        # Prune ignored directories before descending. Path.rglob() walks
        # them first and filters afterward, which makes large repositories
        # appear hung even though most of their contents are never indexed.
        root = root.resolve()
        for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in _IGNORE_DIRS and not name.startswith(".")
            )
            for name in sorted(filenames):
                if name in _IGNORE_NAMES or name.startswith("."):
                    continue
                path = Path(current) / name
                if path.suffix.lower() not in _TEXT_EXTS and path.suffix.lower() not in _CODE_EXTS:
                    continue
                try:
                    if path.stat().st_size > _MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                yield path
