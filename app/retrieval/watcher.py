"""Continuous filesystem indexer.

Watches directories with `watchdog` and triggers incremental indexing on
file changes. Uses per-file debounce to avoid re-indexing a file that's
being rapidly modified (e.g. a build writing multiple times in a second).

Usage:
    watcher = ContinuousIndexer(pipeline, paths=["~/projects", "knowledge/"])
    watcher.start()   # non-blocking, runs in a background thread
    watcher.stop()    # graceful shutdown
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from app.log import get_logger
from app.retrieval.pipeline import IndexingPipeline

log = get_logger(__name__)

# Default ignore patterns — same as the folder indexer plus watcher-specific noise.
_IGNORE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "dist", "build", ".ruff_cache", ".mypy_cache", "target", ".next",
    ".DS_Store", "data", "logs", "models",
}
_IGNORE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".o", ".a", ".class"}
_DEBOUNCE_SECONDS = 2.0


def _should_ignore(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    if any(part in _IGNORE_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in _IGNORE_SUFFIXES:
        return True
    return False


def _is_indexable(path: Path) -> bool:
    """Match the folder indexer's extension filter."""
    from app.retrieval.indexers.folder import _CODE_EXTS, _TEXT_EXTS

    return path.suffix.lower() in _TEXT_EXTS or path.suffix.lower() in _CODE_EXTS


class ContinuousIndexer:
    """Watches directories and incrementally indexes changed files."""

    def __init__(
        self,
        pipeline: IndexingPipeline,
        paths: list[str],
        *,
        debounce_seconds: float = _DEBOUNCE_SECONDS,
    ):
        self.pipeline = pipeline
        self.paths = [str(Path(p).expanduser().resolve()) for p in paths]
        self.debounce_seconds = debounce_seconds
        self._observer = None
        self._pending: dict[str, float] = {}  # path -> last event time
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        from watchdog.observers import Observer

        handler = _Handler(self)
        self._observer = Observer()
        for p in self.paths:
            if not Path(p).exists():
                log.warning("watcher.path_missing", path=p)
                continue
            self._observer.schedule(handler, p, recursive=True)
            log.info("watcher.watching", path=p)

        self._observer.start()
        self._worker_thread = threading.Thread(target=self._debounce_loop, daemon=True)
        self._worker_thread.start()
        log.info("watcher.started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        log.info("watcher.stopped")

    def on_event(self, event_path: str, deleted: bool = False) -> None:
        """Called by the handler on a filesystem event."""
        p = Path(event_path)
        if _should_ignore(p):
            return
        if not deleted and not _is_indexable(p):
            return

        with self._pending_lock:
            self._pending[event_path] = (time.time(), deleted)

    def _debounce_loop(self) -> None:
        """Process pending events after debounce delay."""
        while not self._stop_event.is_set():
            time.sleep(0.5)
            now = time.time()
            ready: list[tuple[str, bool]] = []
            with self._pending_lock:
                for path, (event_time, deleted) in list(self._pending.items()):
                    if now - event_time >= self.debounce_seconds:
                        ready.append((path, deleted))
                        del self._pending[path]

            for path, deleted in ready:
                if deleted:
                    self._handle_delete(path)
                else:
                    self._handle_change(path)

    def _handle_change(self, path: str) -> None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return
        log.info("watcher.indexing", path=path)
        try:
            stats = self.pipeline.index_source(str(p), force=False)
            if stats.indexed > 0:
                log.info("watcher.indexed", path=path, chunks=stats.chunks_added)
            elif stats.skipped > 0:
                log.debug("watcher.skipped", path=path)
        except Exception as e:
            log.warning("watcher.index_failed", path=path, error=str(e))

    def _handle_delete(self, path: str) -> None:
        source_label = self._find_source_label(Path(path))
        log.info("watcher.deleting", path=path)
        self.pipeline.remove_file(path, source_label)

    def _find_source_label(self, path: Path) -> str:
        """Find which watched path this file belongs to."""
        for watched in self.paths:
            try:
                path.relative_to(watched)
                return watched
            except ValueError:
                continue
        return str(path.parent)


class _Handler:  # type: ignore
    """watchdog event handler that forwards to the ContinuousIndexer."""

    def __init__(self, watcher: ContinuousIndexer):
        self._watcher = watcher

    def on_created(self, event):
        if not event.is_directory:
            self._watcher.on_event(event.src_path, deleted=False)

    def on_modified(self, event):
        if not event.is_directory:
            self._watcher.on_event(event.src_path, deleted=False)

    def on_deleted(self, event):
        if not event.is_directory:
            self._watcher.on_event(event.src_path, deleted=True)

    def on_moved(self, event):
        if not event.is_directory:
            self._watcher.on_event(event.src_path, deleted=True)
            if Path(event.dest_path).exists():
                self._watcher.on_event(event.dest_path, deleted=False)
