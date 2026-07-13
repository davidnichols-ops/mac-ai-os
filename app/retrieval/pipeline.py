"""Hash-aware indexing pipeline.

Wraps the indexer + retriever + database so that:
  - unchanged files are skipped (content hash comparison)
  - changed files are re-indexed (delete old chunks, insert new)
  - deleted files are removed from the vector store

This is the core of both the manual `maos index` command and the continuous
filesystem watcher.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.db import Database
from app.events.bus import EventBus
from app.events.models import FILE_DELETED, FILE_INDEXED, FILE_SKIPPED, Event
from app.log import get_logger
from app.retrieval.base import Retriever
from app.retrieval.indexers import build_indexers, dispatch

log = get_logger(__name__)


@dataclass
class IndexStats:
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    errors: int = 0
    chunks_added: int = 0
    files_seen: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"indexed={self.indexed} skipped={self.skipped} "
            f"deleted={self.deleted} errors={self.errors} "
            f"chunks={self.chunks_added}"
        )


def file_content_hash(path: Path) -> str:
    """SHA-256 of file contents (first 256 KB to stay fast on large files)."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            h.update(f.read(256 * 1024))
            h.update(str(path.stat().st_size).encode())
    except OSError:
        return ""
    return h.hexdigest()


class IndexingPipeline:
    """Coordinates indexers, retriever, and hash tracking for incremental indexing."""

    def __init__(
        self,
        retriever: Retriever,
        db: Database,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        bus: EventBus | None = None,
    ):
        self.retriever = retriever
        self.db = db
        self.bus = bus
        self.indexers = build_indexers(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, db=db
        )

    def index_source(self, source: str, *, force: bool = False) -> IndexStats:
        """Index a file or folder. If `force`, re-index everything regardless of hashes."""
        p = Path(source).resolve()
        if not p.exists():
            raise FileNotFoundError(source)

        stats = IndexStats()

        if p.is_file():
            self._index_file(p, source_label=str(p), stats=stats, force=force)
        elif p.is_dir():
            self._index_dir(p, source_label=str(p), stats=stats, force=force)
        else:
            raise ValueError(f"not a file or directory: {p}")

        log.info("pipeline.index_source.done", source=str(p), stats=str(stats))
        return stats

    def _index_dir(self, root: Path, *, source_label: str, stats: IndexStats, force: bool) -> None:
        # Get all files the indexer would process.
        from app.retrieval.indexers.folder import FolderIndexer

        folder_idx = next((i for i in self.indexers if isinstance(i, FolderIndexer)), None)
        walker = folder_idx._walk if folder_idx else _default_walk
        for path in walker(root):
            self._index_file(path, source_label=source_label, stats=stats, force=force)

    def _index_file(
        self,
        path: Path,
        *,
        source_label: str,
        stats: IndexStats,
        force: bool,
    ) -> None:
        stats.files_seen.append(str(path))
        content_hash = file_content_hash(path)
        if not content_hash:
            stats.errors += 1
            return

        stored_hash = self.db.get_file_hash(str(path), source_label)
        if not force and stored_hash == content_hash:
            stats.skipped += 1
            if self.bus:
                self.bus.publish(Event(FILE_SKIPPED, {"path": str(path), "source": source_label}))
            return

        # Index the file (dispatch picks the right indexer per extension).
        try:
            chunks = dispatch(str(path), self.indexers)
        except Exception as e:
            log.warning("pipeline.index_file.failed", path=str(path), error=str(e))
            stats.errors += 1
            return

        if not chunks:
            stats.skipped += 1
            if self.bus:
                self.bus.publish(Event(FILE_SKIPPED, {"path": str(path), "source": source_label}))
            return

        # Delete old chunks for this file, then insert new ones.
        # We use the file path as a sub-source so we can delete per-file.
        file_source = str(path)
        try:
            self.retriever.delete(file_source)
        except Exception:
            pass  # may not exist yet

        n = self.retriever.index(file_source, chunks)
        self.db.set_file_hash(str(path), source_label, content_hash, n)
        stats.indexed += 1
        stats.chunks_added += n
        if self.bus:
            self.bus.publish(Event(FILE_INDEXED, {
                "path": str(path), "chunks": n, "source": source_label,
            }))

    def remove_file(self, path: str, source_label: str) -> bool:
        """Remove a file from the vector store and hash table."""
        try:
            self.retriever.delete(path)
            self.db.remove_file_hash(path, source_label)
            if self.bus:
                self.bus.publish(Event(FILE_DELETED, {"path": path, "source": source_label}))
            log.info("pipeline.remove_file", path=path)
            return True
        except Exception as e:
            log.warning("pipeline.remove_file.failed", path=path, error=str(e))
            return False

    def cleanup_deleted(self, source_label: str) -> int:
        """Remove hash entries for files that no longer exist on disk."""
        rows = self.db.list_file_hashes(source_label)
        removed = 0
        for row in rows:
            if not Path(row["path"]).exists():
                self.remove_file(row["path"], source_label)
                removed += 1
        return removed


def _default_walk(root: Path):
    """Fallback walker if FolderIndexer isn't available."""
    yield from root.rglob("*")
