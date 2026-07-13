"""SQLite database manager for Mac AI OS metadata.

Stores:
  - file_hashes:   content hashes for incremental indexing (skip unchanged files)
  - git_commits:   structured commit metadata for git intelligence queries
  - (future):      telemetry time-series, workspace snapshots

Uses a single database file at data/maos.db (configurable via settings).
Schema is created on first use and is idempotent.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.log import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_hashes (
    path         TEXT NOT NULL,
    source       TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    indexed_at   TEXT NOT NULL,
    PRIMARY KEY (path, source)
);

CREATE TABLE IF NOT EXISTS git_commits (
    repo        TEXT NOT NULL,
    hash        TEXT NOT NULL,
    author      TEXT NOT NULL,
    author_date TEXT NOT NULL,
    message     TEXT NOT NULL,
    files_changed TEXT NOT NULL DEFAULT '',
    insertions  INTEGER NOT NULL DEFAULT 0,
    deletions   INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT NOT NULL,
    PRIMARY KEY (repo, hash)
);

CREATE INDEX IF NOT EXISTS idx_file_hashes_source ON file_hashes(source);
CREATE INDEX IF NOT EXISTS idx_git_commits_repo_date ON git_commits(repo, author_date);
"""


class Database:
    """Thin wrapper over a sqlite3 connection with typed helpers."""

    def __init__(self, path: str = "data/maos.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        log.info("db.init", path=str(self.path))

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        cur = self._conn.execute(sql, params)
        return cur.fetchone()

    def close(self) -> None:
        self._conn.close()

    # --- file_hashes helpers -------------------------------------------------

    def get_file_hash(self, path: str, source: str) -> str | None:
        row = self.query_one(
            "SELECT content_hash FROM file_hashes WHERE path = ? AND source = ?",
            (path, source),
        )
        return row["content_hash"] if row else None

    def set_file_hash(self, path: str, source: str, content_hash: str, chunk_count: int) -> None:
        self.execute(
            """INSERT INTO file_hashes (path, source, content_hash, chunk_count, indexed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(path, source) DO UPDATE SET
                   content_hash = excluded.content_hash,
                   chunk_count = excluded.chunk_count,
                   indexed_at = excluded.indexed_at""",
            (path, source, content_hash, chunk_count, datetime.now(UTC).isoformat()),
        )

    def remove_file_hash(self, path: str, source: str) -> None:
        self.execute("DELETE FROM file_hashes WHERE path = ? AND source = ?", (path, source))

    def list_file_hashes(self, source: str) -> list[sqlite3.Row]:
        return self.query("SELECT path, content_hash FROM file_hashes WHERE source = ?", (source,))

    def file_hash_count(self, source: str | None = None) -> int:
        if source:
            row = self.query_one("SELECT COUNT(*) as n FROM file_hashes WHERE source = ?", (source,))
        else:
            row = self.query_one("SELECT COUNT(*) as n FROM file_hashes")
        return row["n"] if row else 0

    # --- git_commits helpers -------------------------------------------------

    def upsert_commit(
        self,
        *,
        repo: str,
        hash: str,
        author: str,
        author_date: str,
        message: str,
        files_changed: str,
        insertions: int,
        deletions: int,
    ) -> None:
        self.execute(
            """INSERT INTO git_commits
                   (repo, hash, author, author_date, message, files_changed,
                    insertions, deletions, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(repo, hash) DO UPDATE SET
                   author = excluded.author,
                   author_date = excluded.author_date,
                   message = excluded.message,
                   files_changed = excluded.files_changed,
                   insertions = excluded.insertions,
                   deletions = excluded.deletions,
                   indexed_at = excluded.indexed_at""",
            (repo, hash, author, author_date, message, files_changed,
             insertions, deletions, datetime.now(UTC).isoformat()),
        )

    def commits_since(self, repo: str, since: str) -> list[sqlite3.Row]:
        return self.query(
            """SELECT * FROM git_commits
               WHERE repo = ? AND author_date >= ?
               ORDER BY author_date DESC""",
            (repo, since),
        )

    def recent_commits(self, repo: str, limit: int = 10) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM git_commits WHERE repo = ? ORDER BY author_date DESC LIMIT ?",
            (repo, limit),
        )

    def most_modified_files(self, repo: str, limit: int = 10) -> list[tuple[str, int]]:
        rows = self.query("SELECT files_changed FROM git_commits WHERE repo = ?", (repo,))
        counts: dict[str, int] = {}
        for row in rows:
            for f in row["files_changed"].split("\n"):
                f = f.strip()
                if f:
                    counts[f] = counts.get(f, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def commit_count(self, repo: str | None = None) -> int:
        if repo:
            row = self.query_one("SELECT COUNT(*) as n FROM git_commits WHERE repo = ?", (repo,))
        else:
            row = self.query_one("SELECT COUNT(*) as n FROM git_commits")
        return row["n"] if row else 0


# Module-level singleton accessor (lazy, not a global singleton pattern —
# callers can construct their own Database for testing).
_default_db: Database | None = None


def get_db(path: str = "data/maos.db") -> Database:
    global _default_db
    if _default_db is None or _default_db.path != Path(path):
        if _default_db is not None:
            _default_db.close()
        _default_db = Database(path)
    return _default_db


def reset_db() -> None:
    global _default_db
    if _default_db is not None:
        _default_db.close()
        _default_db = None
