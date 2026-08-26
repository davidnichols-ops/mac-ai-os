"""Privacy-aware, device-scale retrieval over user-controlled files.

This module stores local MiniLM vectors and an SQLite FTS5 catalog behind one
hybrid search API. It deliberately indexes the user's knowledge surface rather than
blindly ingesting every byte on disk: system data, credentials, caches,
dependencies, model weights, databases, and binaries are excluded by default.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from array import array
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.base import EmbeddingModel, Model
from app.models.types import Message
from app.retrieval.base import Result, Retriever
from app.retrieval.chunking import chunk_text
from app.retrieval.indexers import build_indexers, dispatch

TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".rtf", ".org", ".tex",
    ".ini", ".cfg", ".toml", ".yaml", ".yml", ".json", ".jsonl",
    ".csv", ".tsv", ".xml", ".html", ".css", ".sql", ".log",
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".java", ".kt",
    ".swift", ".rb", ".php", ".sh", ".bash", ".zsh", ".fish",
    ".ipynb", ".pdf", ".docx",
})

IGNORED_DIRECTORIES = frozenset({
    "Library", "Applications", ".Trash", ".git", ".hg", ".svn",
    ".venv", "venv", "node_modules", "target", "dist", "build",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".next", ".nuxt", ".turbo", "coverage", ".coverage",
    "DerivedData", "Pods", "vendor", "site-packages", "embeddings",
})

SENSITIVE_SUFFIXES = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".mobileprovision", ".sqlite",
    ".sqlite3", ".db", ".kdbx",
})

_SENSITIVE_NAME = re.compile(
    r"(^|[._-])(secret|credentials?|tokens?|passwords?|cookies?|keychain)([._-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeviceIndexPolicy:
    """The explicit boundary for device indexing."""

    roots: tuple[Path, ...] = field(default_factory=lambda: (Path.home(),))
    allowed_extensions: frozenset[str] = TEXT_EXTENSIONS
    ignored_directories: frozenset[str] = IGNORED_DIRECTORIES
    max_file_bytes: int = 2 * 1024 * 1024
    max_pdf_bytes: int = 20 * 1024 * 1024
    include_hidden: bool = False

    def normalized_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        for raw in self.roots:
            root = raw.expanduser().resolve()
            if root not in roots:
                roots.append(root)
        return tuple(roots)

    def accepts(self, path: Path) -> tuple[bool, str]:
        name = path.name
        lower = name.lower()
        suffix = path.suffix.lower()
        if not self.include_hidden and name.startswith("."):
            return False, "hidden"
        if lower == ".env" or lower.startswith(".env."):
            return False, "sensitive_name"
        if _SENSITIVE_NAME.search(lower) or suffix in SENSITIVE_SUFFIXES:
            return False, "sensitive_name"
        if suffix not in self.allowed_extensions:
            return False, "unsupported_type"
        try:
            size = path.stat().st_size
        except OSError:
            return False, "unreadable"
        limit = self.max_pdf_bytes if suffix == ".pdf" else self.max_file_bytes
        if size > limit:
            return False, "too_large"
        return True, "accepted"


@dataclass
class ScanStats:
    roots: list[str] = field(default_factory=list)
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    chunks: int = 0
    errors: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False
    limited: bool = False

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeviceCatalog:
    """SQLite file manifest and lexical index kept beside the vector store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._vector_cache: list[tuple[str, int, list[float]]] | None = None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS device_files (
                path TEXT PRIMARY KEY,
                source_root TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_device_files_root
                ON device_files(source_root);
            CREATE VIRTUAL TABLE IF NOT EXISTS device_chunks USING fts5(
                text,
                path UNINDEXED,
                source_root UNINDEXED,
                metadata UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS device_embeddings (
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                vector BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                source_root TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(path, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS idx_device_embeddings_path
                ON device_embeddings(path);
            CREATE TABLE IF NOT EXISTS device_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                stats_json TEXT NOT NULL
            );
            """
        )
        embedding_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(device_embeddings)").fetchall()
        }
        for column, declaration in (
            ("text", "TEXT NOT NULL DEFAULT ''"),
            ("source_root", "TEXT NOT NULL DEFAULT ''"),
            ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
        ):
            if column not in embedding_columns:
                self._conn.execute(
                    f"ALTER TABLE device_embeddings ADD COLUMN {column} {declaration}"
                )
        self._conn.commit()

    def file_hash(self, path: Path) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM device_files WHERE path = ?", (str(path),)
            ).fetchone()
        return str(row["content_hash"]) if row else None

    def update_source_root(self, path: Path, root: Path) -> None:
        """Move an unchanged file to the most recently scanned source boundary."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE device_files SET source_root = ? WHERE path = ?",
                (str(root), str(path)),
            )
            self._conn.execute(
                "UPDATE device_chunks SET source_root = ? WHERE path = ?",
                (str(root), str(path)),
            )
            self._conn.execute(
                "UPDATE device_embeddings SET source_root = ? WHERE path = ?",
                (str(root), str(path)),
            )

    def paths_for_root(self, root: Path) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path FROM device_files WHERE source_root = ?", (str(root),)
            ).fetchall()
        return {str(row["path"]) for row in rows}

    def replace_file(
        self,
        path: Path,
        root: Path,
        content_hash: str,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]] | None = None,
    ) -> None:
        stat = path.stat()
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            self._vector_cache = None
            self._conn.execute("DELETE FROM device_chunks WHERE path = ?", (str(path),))
            self._conn.execute("DELETE FROM device_embeddings WHERE path = ?", (str(path),))
            self._conn.executemany(
                "INSERT INTO device_chunks(text, path, source_root, metadata) VALUES (?, ?, ?, ?)",
                [
                    (
                        chunk["text"],
                        str(path),
                        str(root),
                        json.dumps(chunk.get("metadata", {}), default=str),
                    )
                    for chunk in chunks
                ],
            )
            if vectors is not None:
                self._conn.executemany(
                    """INSERT INTO device_embeddings(
                           path, chunk_index, vector, dimensions, text, source_root, metadata
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            str(path), index, array("f", vector).tobytes(), len(vector),
                            chunks[index]["text"], str(root),
                            json.dumps(chunks[index].get("metadata", {}), default=str),
                        )
                        for index, vector in enumerate(vectors)
                    ],
                )
            self._conn.execute(
                """INSERT INTO device_files(
                       path, source_root, content_hash, size_bytes, mtime_ns,
                       chunk_count, indexed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(path) DO UPDATE SET
                       source_root=excluded.source_root,
                       content_hash=excluded.content_hash,
                       size_bytes=excluded.size_bytes,
                       mtime_ns=excluded.mtime_ns,
                       chunk_count=excluded.chunk_count,
                       indexed_at=excluded.indexed_at""",
                (
                    str(path), str(root), content_hash, stat.st_size,
                    stat.st_mtime_ns, len(chunks), now,
                ),
            )

    def remove_file(self, path: str) -> None:
        with self._lock, self._conn:
            self._vector_cache = None
            self._conn.execute("DELETE FROM device_chunks WHERE path = ?", (path,))
            self._conn.execute("DELETE FROM device_embeddings WHERE path = ?", (path,))
            self._conn.execute("DELETE FROM device_files WHERE path = ?", (path,))

    def semantic_search(self, query_vector: list[float], limit: int) -> list[Result]:
        """Exact cosine search over local float32 vectors.

        This avoids Chroma's native HNSW boundary, which can be repaired or
        replaced independently. The implementation is deterministic and is a
        good fit for a personal-device corpus; a future ANN adapter can keep
        the same catalog contract.
        """
        if not query_vector:
            return []
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        with self._lock:
            if self._vector_cache is None:
                rows = self._conn.execute(
                    "SELECT path, chunk_index, vector, dimensions FROM device_embeddings"
                ).fetchall()
                cache: list[tuple[str, int, list[float]]] = []
                for row in rows:
                    vector = array("f")
                    vector.frombytes(row["vector"])
                    cache.append((str(row["path"]), int(row["chunk_index"]), list(vector)))
                self._vector_cache = cache
            vectors = self._vector_cache
        scored: list[tuple[float, str, int]] = []
        for path, chunk_index, vector in vectors:
            if len(vector) != len(query_vector):
                continue
            vector_norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            score /= query_norm * vector_norm
            scored.append((score, path, chunk_index))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[Result] = []
        for rank, (score, path, chunk_index) in enumerate(scored[:limit], start=1):
            with self._lock:
                row = self._conn.execute(
                    """SELECT text, source_root, metadata FROM device_embeddings
                       WHERE path = ? AND chunk_index = ?""",
                    (path, chunk_index),
                ).fetchone()
            if row is None:
                continue
            metadata = json.loads(row["metadata"] or "{}")
            metadata["source_root"] = row["source_root"]
            metadata["semantic_rank"] = rank
            results.append(Result(
                text=row["text"],
                score=score,
                source=path,
                metadata=metadata,
            ))
        return results

    def lexical_search(self, query: str, limit: int) -> list[Result]:
        terms = re.findall(r"[\w./:-]+", query, flags=re.UNICODE)
        if not terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:16])
        try:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT text, path, source_root, metadata, bm25(device_chunks) AS rank
                       FROM device_chunks WHERE device_chunks MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (expression, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[Result] = []
        for index, row in enumerate(rows):
            metadata = json.loads(row["metadata"] or "{}")
            metadata["source_root"] = row["source_root"]
            metadata["lexical_rank"] = index + 1
            results.append(Result(
                text=row["text"],
                score=1.0 / (index + 1),
                source=row["path"],
                metadata=metadata,
            ))
        return results

    def record_run(self, started_at: str, stats: ScanStats) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO device_runs(started_at, finished_at, stats_json) VALUES (?, ?, ?)",
                (started_at, datetime.now(UTC).isoformat(), json.dumps(stats.to_dict())),
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            totals = self._conn.execute(
                "SELECT COUNT(*) AS files, COALESCE(SUM(chunk_count), 0) AS chunks, "
                "COALESCE(SUM(size_bytes), 0) AS bytes FROM device_files"
            ).fetchone()
            roots = self._conn.execute(
                """SELECT source_root, COUNT(*) AS files, SUM(chunk_count) AS chunks,
                          MAX(indexed_at) AS last_indexed
                   FROM device_files GROUP BY source_root ORDER BY source_root"""
            ).fetchall()
            last = self._conn.execute(
                "SELECT finished_at, stats_json FROM device_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return {
            "catalog": str(self.path),
            "files": int(totals["files"]),
            "chunks": int(totals["chunks"]),
            "bytes": int(totals["bytes"]),
            "roots": [dict(row) for row in roots],
            "last_sync": (
                {"finished_at": last["finished_at"], "stats": json.loads(last["stats_json"])}
                if last else None
            ),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class DeviceRAGService:
    """Incremental device index, hybrid retrieval, and grounded answering."""

    def __init__(
        self,
        retriever: Retriever | None,
        catalog: DeviceCatalog,
        *,
        embedding_model: EmbeddingModel | None = None,
        policy: DeviceIndexPolicy | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.retriever = retriever
        self.catalog = catalog
        self.embedding_model = embedding_model
        self.policy = policy or DeviceIndexPolicy()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.indexers = build_indexers(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _walk(self, root: Path, stats: ScanStats) -> Iterable[Path]:
        for current, dirnames, filenames in os.walk(
            root, topdown=True, followlinks=False, onerror=lambda _e: stats.skip("unreadable_dir")
        ):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in self.policy.ignored_directories
                and (self.policy.include_hidden or not name.startswith("."))
            )
            for name in sorted(filenames):
                path = Path(current) / name
                accepted, reason = self.policy.accepts(path)
                if accepted:
                    yield path
                else:
                    stats.skip(reason)

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _chunks(self, path: Path, root: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".docx":
            text = _read_docx(path)
            raw = chunk_text(
                text,
                source=str(path),
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                metadata={"kind": "document", "ext": ".docx"},
            )
            chunks = [{"text": chunk.text, "metadata": chunk.to_metadata()} for chunk in raw]
        else:
            chunks = dispatch(str(path), self.indexers)
        relative = _relative_to(path, root)
        for chunk in chunks:
            metadata = chunk.setdefault("metadata", {})
            metadata.update({
                "source": str(path),
                "source_root": str(root),
                "rel_path": relative,
                "device_rag": True,
            })
            chunk["text"] = f"Source: {relative}\n\n{chunk['text']}"
        return chunks

    def sync(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
        max_files: int | None = None,
    ) -> ScanStats:
        started_at = datetime.now(UTC).isoformat()
        stats = ScanStats(dry_run=dry_run)
        for root in self.policy.normalized_roots():
            stats.roots.append(str(root))
            if not root.exists() or not root.is_dir():
                stats.skip("missing_root")
                continue
            previously_indexed = self.catalog.paths_for_root(root)
            seen: set[str] = set()
            for path in self._walk(root, stats):
                if max_files is not None and stats.discovered >= max_files:
                    stats.limited = True
                    break
                stats.discovered += 1
                seen.add(str(path))
                try:
                    content_hash = self._hash(path)
                    if not force and self.catalog.file_hash(path) == content_hash:
                        self.catalog.update_source_root(path, root)
                        stats.unchanged += 1
                        continue
                    if dry_run:
                        stats.indexed += 1
                        continue
                    chunks = self._chunks(path, root)
                    if not chunks:
                        stats.skip("empty")
                        continue
                    vectors: list[list[float]] | None = None
                    if self.embedding_model is not None:
                        vectors = [
                            [float(value) for value in response.vector]
                            for response in self.embedding_model.embed_batch(
                                [chunk["text"] for chunk in chunks]
                            )
                        ]
                        added = len(chunks)
                    elif self.retriever is not None:
                        self.retriever.delete(str(path))
                        added = self.retriever.index(str(path), chunks)
                    else:
                        raise RuntimeError("device RAG needs an embedding model or retriever")
                    self.catalog.replace_file(
                        path, root, content_hash, chunks, vectors=vectors
                    )
                    stats.indexed += 1
                    stats.chunks += added
                except Exception:
                    stats.errors += 1
            # A limited or dry scan cannot prove that unseen files were deleted.
            if not dry_run and not stats.limited:
                for deleted in sorted(previously_indexed - seen):
                    if self.retriever is not None:
                        self.retriever.delete(deleted)
                    self.catalog.remove_file(deleted)
                    stats.deleted += 1
        if not dry_run:
            self.catalog.record_run(started_at, stats)
        return stats

    def search(self, query: str, *, top_k: int = 8) -> list[Result]:
        if not query.strip():
            return []
        candidate_count = max(top_k * 4, 20)
        if self.embedding_model is not None:
            query_vector = [float(value) for value in self.embedding_model.embed(query).vector]
            dense = self.catalog.semantic_search(query_vector, candidate_count)
        elif self.retriever is not None:
            dense = self.retriever.search(query, top_k=candidate_count)
        else:
            dense = []
        lexical = self.catalog.lexical_search(query, candidate_count)
        fused: dict[str, dict[str, Any]] = {}

        def add(results: list[Result], signal: str, weight: float) -> None:
            for rank, result in enumerate(results, start=1):
                chunk_index = result.metadata.get("chunk_index", "")
                key = f"{result.source}:{chunk_index}:{hashlib.sha1(result.text.encode()).hexdigest()}"
                entry = fused.setdefault(key, {
                    "result": result,
                    "score": 0.0,
                    "signals": {},
                })
                entry["score"] += weight / (60 + rank)
                entry["signals"][signal] = rank

        add(dense, "semantic", 0.65)
        add(lexical, "lexical", 0.35)
        ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        output: list[Result] = []
        for item in ordered:
            result = item["result"]
            metadata = dict(result.metadata)
            metadata["retrieval_signals"] = item["signals"]
            output.append(Result(
                text=result.text,
                score=float(item["score"]),
                source=result.source,
                metadata=metadata,
            ))
        return output

    def answer(self, question: str, model: Model, *, top_k: int = 8) -> dict[str, Any]:
        results = self.search(question, top_k=top_k)
        if not results:
            return {
                "question": question,
                "answer": "The index does not contain enough evidence to answer this question.",
                "sources": [],
            }
        context = "\n\n".join(
            f"[{index}] {result.source}\n{result.text}"
            for index, result in enumerate(results, start=1)
        )
        response = model.chat([
            Message(
                "system",
                "Answer only from the supplied device evidence. Cite claims with [number]. "
                "Separate contradictions and unknowns; never invent missing facts.",
            ),
            Message("user", f"Evidence:\n{context}\n\nQuestion: {question}"),
        ])
        return {
            "question": question,
            "answer": response.text,
            "sources": [
                {
                    "id": index,
                    "path": result.source,
                    "score": result.score,
                    "text": result.text,
                    "metadata": result.metadata,
                }
                for index, result in enumerate(results, start=1)
            ],
        }

    def status(self) -> dict[str, Any]:
        status = self.catalog.status()
        status.update({
            "semantic_backend": (
                "sqlite-cosine"
                if self.embedding_model is not None
                else getattr(self.retriever, "backend", "lexical-only")
            ),
            "semantic_chunks": status["chunks"],
            "policy": {
                "roots": [str(root) for root in self.policy.normalized_roots()],
                "max_file_bytes": self.policy.max_file_bytes,
                "max_pdf_bytes": self.policy.max_pdf_bytes,
                "include_hidden": self.policy.include_hidden,
                "sensitive_files_indexed": False,
            },
        })
        return status


def build_device_rag(runtime: Any, *, roots: list[str] | None = None) -> DeviceRAGService:
    """Construct the device service from an existing MAOS runtime."""
    data_dir = runtime.settings.resolved_data_dir()
    selected = tuple(Path(root) for root in roots) if roots else (Path.home(),)
    return DeviceRAGService(
        None,
        DeviceCatalog(data_dir / "device_rag.db"),
        embedding_model=runtime.embedding_model,
        policy=DeviceIndexPolicy(roots=selected),
        chunk_size=runtime.settings.retrieval.chunk_size,
        chunk_overlap=runtime.settings.retrieval.chunk_overlap,
    )


def _read_docx(path: Path) -> str:
    """Extract paragraph text from DOCX without adding a heavyweight dependency."""
    import zipfile
    from xml.etree import ElementTree

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = "".join(
            node.text or ""
            for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        )
        if text.strip():
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
