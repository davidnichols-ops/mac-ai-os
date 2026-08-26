"""SQLite database manager for Mac AI OS metadata.

Stores:
  - file_hashes:   content hashes for incremental indexing (skip unchanged files)
  - git_commits:   structured commit metadata for git intelligence queries
  - (future):      telemetry time-series, workspace snapshots

Uses a single database file at data/maos.db (configurable via settings).
Schema is created on first use and is idempotent.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from datetime import timedelta as _timedelta
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

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    type      TEXT NOT NULL,
    payload   TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    cpu_percent     REAL NOT NULL DEFAULT 0,
    cpu_cores       INTEGER NOT NULL DEFAULT 0,
    memory_total_gb REAL NOT NULL DEFAULT 0,
    memory_used_gb  REAL NOT NULL DEFAULT 0,
    memory_percent  REAL NOT NULL DEFAULT 0,
    swap_used_gb    REAL NOT NULL DEFAULT 0,
    swap_total_gb   REAL NOT NULL DEFAULT 0,
    disk_read_mb    REAL NOT NULL DEFAULT 0,
    disk_write_mb   REAL NOT NULL DEFAULT 0,
    battery_percent INTEGER,
    battery_plugged INTEGER,
    thermal_state   TEXT,
    process_count   INTEGER NOT NULL DEFAULT 0,
    gpu_usage       REAL,
    neural_engine   REAL
);

CREATE TABLE IF NOT EXISTS process_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    telemetry_id    INTEGER NOT NULL,
    pid             INTEGER NOT NULL,
    name            TEXT NOT NULL,
    cpu_percent     REAL NOT NULL DEFAULT 0,
    memory_mb       REAL NOT NULL DEFAULT 0,
    FOREIGN KEY (telemetry_id) REFERENCES telemetry_snapshots(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    type        TEXT NOT NULL,
    description TEXT,
    source      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL,
    relation    TEXT NOT NULL,
    target_id   INTEGER NOT NULL,
    source_ref  TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE,
    UNIQUE (source_id, relation, target_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time  TEXT NOT NULL,
    end_time    TEXT NOT NULL,
    project     TEXT,
    workspace   TEXT,
    summary     TEXT,
    event_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspaces (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    language      TEXT,
    git_repo      INTEGER NOT NULL DEFAULT 0,
    last_activity TEXT,
    file_count    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shell_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    command     TEXT NOT NULL,
    directory   TEXT,
    exit_code   INTEGER,
    session_id  INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,
    agent             TEXT NOT NULL,
    decision          TEXT NOT NULL,
    reasoning_summary TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 0.0,
    outcome           TEXT,
    query             TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    title             TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    created_at        TEXT NOT NULL,
    completed_at      TEXT,
    source            TEXT NOT NULL DEFAULT 'manual',
    related_workspace TEXT,
    due_date          TEXT,
    priority          INTEGER NOT NULL DEFAULT 0,
    parent_id         INTEGER,
    dependencies      TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (parent_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS approval_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    action_name TEXT NOT NULL,
    target      TEXT,
    parameters  TEXT NOT NULL DEFAULT '{}',
    reason      TEXT NOT NULL DEFAULT '',
    risk        TEXT NOT NULL DEFAULT 'low',
    status      TEXT NOT NULL DEFAULT 'pending',
    decided_at  TEXT,
    decided_by  TEXT,
    workflow_id INTEGER
);

CREATE TABLE IF NOT EXISTS user_model (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL UNIQUE,
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    source      TEXT NOT NULL DEFAULT 'observed',
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_memories (
    id              TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',
    source          TEXT NOT NULL DEFAULT 'manual',
    source_ref      TEXT NOT NULL DEFAULT '',
    author_id       TEXT NOT NULL DEFAULT '',
    archive_status  TEXT NOT NULL DEFAULT 'pending',
    archive_path    TEXT NOT NULL DEFAULT '',
    archive_hash    TEXT NOT NULL DEFAULT '',
    archive_error   TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    indexed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_personal_memories_hash ON personal_memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_personal_memories_archive ON personal_memories(archive_status);

CREATE TABLE IF NOT EXISTS workflow_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    workflow    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    steps_total INTEGER NOT NULL DEFAULT 0,
    steps_done  INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    duration_ms INTEGER,
    workflow_id TEXT NOT NULL DEFAULT '',
    version     INTEGER NOT NULL DEFAULT 1,
    heartbeat_at TEXT,
    parent_run_id INTEGER,
    paused_at   TEXT,
    cancelled_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS workflow_definitions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    definition  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS workflow_step_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    step_id     TEXT NOT NULL DEFAULT '',
    step_name   TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    attempt     INTEGER NOT NULL DEFAULT 1,
    data        TEXT NOT NULL DEFAULT '{}',
    verification TEXT,
    error       TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER,
    started_at  TEXT,
    finished_at TEXT,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    step_index  INTEGER NOT NULL,
    context     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_side_effects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    step_id     TEXT NOT NULL DEFAULT '',
    step_name   TEXT NOT NULL,
    effect_type TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    details     TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_dead_letter_queue (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                INTEGER NOT NULL,
    workflow_name         TEXT NOT NULL,
    failure_reason        TEXT NOT NULL DEFAULT '',
    compensation_records  TEXT NOT NULL DEFAULT '[]',
    context_snapshot      TEXT NOT NULL DEFAULT '{}',
    enqueued_at           TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'pending',
    notes                 TEXT NOT NULL DEFAULT '',
    inspected_at          TEXT,
    new_run_id            INTEGER,
    reprocessed_at        TEXT,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workspace_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    workspace       TEXT NOT NULL,
    git_branch      TEXT,
    git_status      TEXT,
    modified_files  TEXT NOT NULL DEFAULT '',
    recent_commands TEXT NOT NULL DEFAULT '',
    active_processes TEXT NOT NULL DEFAULT '',
    telemetry_id    INTEGER,
    FOREIGN KEY (telemetry_id) REFERENCES telemetry_snapshots(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS action_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    action_name     TEXT NOT NULL,
    target          TEXT,
    parameters      TEXT NOT NULL DEFAULT '{}',
    approved        INTEGER NOT NULL DEFAULT 0,
    executed        INTEGER NOT NULL DEFAULT 0,
    result          TEXT,
    error           TEXT,
    duration_ms     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_file_hashes_source ON file_hashes(source);
CREATE INDEX IF NOT EXISTS idx_git_commits_repo_date ON git_commits(repo, author_date);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(type, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_processes_telemetry ON process_snapshots(telemetry_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_workspaces_path ON workspaces(path);
CREATE INDEX IF NOT EXISTS idx_shell_commands_time ON shell_commands(timestamp);
CREATE INDEX IF NOT EXISTS idx_shell_commands_dir ON shell_commands(directory);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON workspace_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_workspace ON workspace_snapshots(workspace);
CREATE INDEX IF NOT EXISTS idx_action_log_time ON action_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_queue(status);
CREATE INDEX IF NOT EXISTS idx_user_model_key ON user_model(key);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_time ON workflow_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status);
-- idx_workflow_runs_workflow_id is created in _run_migrations() after the
-- workflow_id column is added to existing tables. Creating it here fails on
-- databases with an older workflow_runs schema (no workflow_id column yet).
CREATE INDEX IF NOT EXISTS idx_workflow_step_runs_run_id ON workflow_step_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_step_runs_step_id ON workflow_step_runs(step_id);
CREATE INDEX IF NOT EXISTS idx_workflow_definitions_name ON workflow_definitions(name);
CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_run_id ON workflow_checkpoints(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_side_effects_run_id ON workflow_side_effects(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_side_effects_idem_key ON workflow_side_effects(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_workflow_dlq_status ON workflow_dead_letter_queue(status);

CREATE TABLE IF NOT EXISTS mcp_requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    client_name TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    success     INTEGER NOT NULL DEFAULT 1,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_mcp_requests_time ON mcp_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_mcp_requests_kind ON mcp_requests(kind);

CREATE TABLE IF NOT EXISTS context_evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    workspace       REAL NOT NULL DEFAULT 0,
    session         REAL NOT NULL DEFAULT 0,
    tasks           REAL NOT NULL DEFAULT 0,
    knowledge       REAL NOT NULL DEFAULT 0,
    preferences     REAL NOT NULL DEFAULT 0,
    activity        REAL NOT NULL DEFAULT 0,
    overall         REAL NOT NULL DEFAULT 0,
    details         TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_context_evals_time ON context_evaluations(timestamp);

CREATE TABLE IF NOT EXISTS file_reads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    path        TEXT NOT NULL,
    lines       INTEGER NOT NULL DEFAULT 0,
    session_id  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_file_reads_time ON file_reads(timestamp);
CREATE INDEX IF NOT EXISTS idx_file_reads_path ON file_reads(path);

-- Network intelligence layer (Phase 5.7).
-- Raw per-interface samples (retention: 24h, enforced by NetworkCollector).
CREATE TABLE IF NOT EXISTS network_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    interface       TEXT NOT NULL,
    network_type    TEXT NOT NULL DEFAULT 'unknown',
    connection_state TEXT NOT NULL DEFAULT 'unknown',
    rx_bytes        INTEGER NOT NULL DEFAULT 0,
    tx_bytes        INTEGER NOT NULL DEFAULT 0,
    rx_rate_bps     REAL NOT NULL DEFAULT 0,
    tx_rate_bps     REAL NOT NULL DEFAULT 0,
    latency_ms      REAL,
    packet_loss     REAL,
    rtt_ok          INTEGER
);

CREATE INDEX IF NOT EXISTS idx_network_samples_time ON network_samples(timestamp);
CREATE INDEX IF NOT EXISTS idx_network_samples_iface_time ON network_samples(interface, timestamp);

-- Minute aggregates (retention: 30 days).
CREATE TABLE IF NOT EXISTS network_minute (
    bucket          TEXT NOT NULL,
    interface       TEXT NOT NULL,
    network_type    TEXT NOT NULL DEFAULT 'unknown',
    rx_bytes        INTEGER NOT NULL DEFAULT 0,
    tx_bytes        INTEGER NOT NULL DEFAULT 0,
    rx_rate_avg     REAL NOT NULL DEFAULT 0,
    tx_rate_avg     REAL NOT NULL DEFAULT 0,
    rx_rate_peak    REAL NOT NULL DEFAULT 0,
    tx_rate_peak    REAL NOT NULL DEFAULT 0,
    latency_avg     REAL,
    latency_p95     REAL,
    packet_loss_avg REAL,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket, interface)
);

CREATE INDEX IF NOT EXISTS idx_network_minute_bucket ON network_minute(bucket);
CREATE INDEX IF NOT EXISTS idx_network_minute_iface ON network_minute(interface);

-- Hourly aggregates (retention: indefinite).
CREATE TABLE IF NOT EXISTS network_hourly (
    bucket          TEXT NOT NULL,
    interface       TEXT NOT NULL,
    network_type    TEXT NOT NULL DEFAULT 'unknown',
    rx_bytes        INTEGER NOT NULL DEFAULT 0,
    tx_bytes        INTEGER NOT NULL DEFAULT 0,
    rx_rate_avg     REAL NOT NULL DEFAULT 0,
    tx_rate_avg     REAL NOT NULL DEFAULT 0,
    latency_avg     REAL,
    packet_loss_avg REAL,
    minute_count    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket, interface)
);

CREATE INDEX IF NOT EXISTS idx_network_hourly_bucket ON network_hourly(bucket);

-- Network events: connection interruptions, degradation, interface transitions.
-- Append-only event log (the spec's events.jsonl equivalent, but in SQLite
-- so it joins with the rest of the activity timeline).
CREATE TABLE IF NOT EXISTS network_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    interface       TEXT NOT NULL DEFAULT '',
    network_type    TEXT NOT NULL DEFAULT '',
    detail          TEXT NOT NULL DEFAULT '{}',
    severity        TEXT NOT NULL DEFAULT 'info'
);

CREATE INDEX IF NOT EXISTS idx_network_events_time ON network_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_network_events_type ON network_events(event_type);
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
        self._txn_depth: int = 0
        self._init_schema()
        log.info("db.init", path=str(self.path))

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Add columns to existing tables that were created before schema updates."""
        # Check tasks table for parent_id column (added in Phase 4).
        cols = {row["name"] for row in self.query("PRAGMA table_info(tasks)")}
        if "parent_id" not in cols:
            self.execute("ALTER TABLE tasks ADD COLUMN parent_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE")
        if "dependencies" not in cols:
            self.execute("ALTER TABLE tasks ADD COLUMN dependencies TEXT NOT NULL DEFAULT ''")
        # Create index after column exists.
        self.execute("CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id)")

        # Phase 5.5: workspace profiling columns.
        ws_cols = {row["name"] for row in self.query("PRAGMA table_info(workspaces)")}
        if "profile" not in ws_cols:
            self.execute("ALTER TABLE workspaces ADD COLUMN profile TEXT NOT NULL DEFAULT ''")
        if "project_status" not in ws_cols:
            self.execute("ALTER TABLE workspaces ADD COLUMN project_status TEXT NOT NULL DEFAULT ''")
        if "profiled_at" not in ws_cols:
            self.execute("ALTER TABLE workspaces ADD COLUMN profiled_at TEXT")

        # Lesson metadata: category, keywords, reminder for contextual injection.
        um_cols = {row["name"] for row in self.query("PRAGMA table_info(user_model)")}
        if "metadata" not in um_cols:
            self.execute("ALTER TABLE user_model ADD COLUMN metadata TEXT")

        # Phase 1 (workflow evolution): workflow_runs new columns.
        wr_cols = {row["name"] for row in self.query("PRAGMA table_info(workflow_runs)")}
        if "workflow_id" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN workflow_id TEXT NOT NULL DEFAULT ''")
        if "version" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
        if "heartbeat_at" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN heartbeat_at TEXT")
        if "parent_run_id" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN parent_run_id INTEGER")
        if "paused_at" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN paused_at TEXT")
        if "cancelled_at" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN cancelled_at TEXT")
        if "finished_at" not in wr_cols:
            self.execute("ALTER TABLE workflow_runs ADD COLUMN finished_at TEXT")
        self.execute("CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_workflow_runs_workflow_id ON workflow_runs(workflow_id)")

        # Hash-chained decisions: add digest + prev_digest columns for tamper-evidence.
        dec_cols = {row["name"] for row in self.query("PRAGMA table_info(decisions)")}
        if "digest" not in dec_cols:
            self.execute("ALTER TABLE decisions ADD COLUMN digest TEXT")
        if "prev_digest" not in dec_cols:
            self.execute("ALTER TABLE decisions ADD COLUMN prev_digest TEXT")
        if "kind" not in dec_cols:
            self.execute("ALTER TABLE decisions ADD COLUMN kind TEXT NOT NULL DEFAULT 'decision'")
        if "tags" not in dec_cols:
            self.execute("ALTER TABLE decisions ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'")

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, params)
        if self._txn_depth == 0:
            self._conn.commit()
        return cur

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        cur = self._conn.executemany(sql, params)
        if self._txn_depth == 0:
            self._conn.commit()
        return cur

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Batch writes and commit at the end; rollback on exception."""
        self._txn_depth += 1
        try:
            yield
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._txn_depth -= 1
            if self._txn_depth == 0:
                self._conn.commit()

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

    # --- events helpers -------------------------------------------------------

    def query_events(
        self,
        *,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM events WHERE 1=1"
        params: list = []
        if event_type:
            sql += " AND type = ?"
            params.append(event_type)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return self.query(sql, tuple(params))

    def event_count(self, event_type: str | None = None) -> int:
        if event_type:
            row = self.query_one("SELECT COUNT(*) as n FROM events WHERE type = ?", (event_type,))
        else:
            row = self.query_one("SELECT COUNT(*) as n FROM events")
        return row["n"] if row else 0

    # --- telemetry helpers ----------------------------------------------------

    def insert_telemetry(self, snapshot: dict, top_processes: list[dict]) -> int:
        cur = self.execute(
            """INSERT INTO telemetry_snapshots
                   (timestamp, cpu_percent, cpu_cores, memory_total_gb, memory_used_gb,
                    memory_percent, swap_used_gb, swap_total_gb, disk_read_mb, disk_write_mb,
                    battery_percent, battery_plugged, thermal_state, process_count,
                    gpu_usage, neural_engine)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot["timestamp"],
                snapshot["cpu_percent"],
                snapshot["cpu_cores"],
                snapshot["memory_total_gb"],
                snapshot["memory_used_gb"],
                snapshot["memory_percent"],
                snapshot.get("swap_used_gb", 0),
                snapshot.get("swap_total_gb", 0),
                snapshot.get("disk_read_mb", 0),
                snapshot.get("disk_write_mb", 0),
                snapshot.get("battery_percent"),
                snapshot.get("battery_plugged"),
                snapshot.get("thermal_state"),
                snapshot.get("process_count", 0),
                snapshot.get("gpu_usage"),
                snapshot.get("neural_engine"),
            ),
        )
        telemetry_id = cur.lastrowid
        for p in top_processes:
            self.execute(
                """INSERT INTO process_snapshots
                       (telemetry_id, pid, name, cpu_percent, memory_mb)
                   VALUES (?, ?, ?, ?, ?)""",
                (telemetry_id, p["pid"], p["name"], p["cpu"], p["memory_mb"]),
            )
        return telemetry_id

    def telemetry_since(self, since: str, limit: int = 100) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM telemetry_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
            (since, limit),
        )

    def latest_telemetry(self) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT 1")

    def processes_for_telemetry(self, telemetry_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM process_snapshots WHERE telemetry_id = ? ORDER BY memory_mb DESC",
            (telemetry_id,),
        )

    def telemetry_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM telemetry_snapshots")
        return row["n"] if row else 0

    # --- network intelligence helpers ----------------------------------------

    def insert_network_sample(self, sample: dict) -> int:
        """Insert one raw network sample. Returns the row id."""
        cur = self.execute(
            """INSERT INTO network_samples
                   (timestamp, interface, network_type, connection_state,
                    rx_bytes, tx_bytes, rx_rate_bps, tx_rate_bps,
                    latency_ms, packet_loss, rtt_ok)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sample["timestamp"],
                sample["interface"],
                sample.get("network_type", "unknown"),
                sample.get("connection_state", "unknown"),
                sample.get("rx_bytes", 0),
                sample.get("tx_bytes", 0),
                sample.get("rx_rate_bps", 0.0),
                sample.get("tx_rate_bps", 0.0),
                sample.get("latency_ms"),
                sample.get("packet_loss"),
                sample.get("rtt_ok"),
            ),
        )
        return cur.lastrowid

    def latest_network_sample(self, interface: str = "") -> sqlite3.Row | None:
        if interface:
            return self.query_one(
                "SELECT * FROM network_samples WHERE interface = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (interface,),
            )
        return self.query_one(
            "SELECT * FROM network_samples ORDER BY timestamp DESC LIMIT 1"
        )

    def network_samples_since(self, since: str, limit: int = 500) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM network_samples WHERE timestamp >= ? "
            "ORDER BY timestamp ASC LIMIT ?",
            (since, limit),
        )

    def network_sample_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM network_samples")
        return row["n"] if row else 0

    def upsert_network_minute(self, agg: dict) -> None:
        """Upsert a minute aggregate (keyed by bucket + interface)."""
        self.execute(
            """INSERT INTO network_minute
                   (bucket, interface, network_type, rx_bytes, tx_bytes,
                    rx_rate_avg, tx_rate_avg, rx_rate_peak, tx_rate_peak,
                    latency_avg, latency_p95, packet_loss_avg, sample_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(bucket, interface) DO UPDATE SET
                   network_type   = excluded.network_type,
                   rx_bytes       = excluded.rx_bytes,
                   tx_bytes       = excluded.tx_bytes,
                   rx_rate_avg    = excluded.rx_rate_avg,
                   tx_rate_avg    = excluded.tx_rate_avg,
                   rx_rate_peak   = excluded.rx_rate_peak,
                   tx_rate_peak   = excluded.tx_rate_peak,
                   latency_avg    = excluded.latency_avg,
                   latency_p95    = excluded.latency_p95,
                   packet_loss_avg = excluded.packet_loss_avg,
                   sample_count   = excluded.sample_count""",
            (
                agg["bucket"], agg["interface"], agg.get("network_type", "unknown"),
                agg["rx_bytes"], agg["tx_bytes"],
                agg["rx_rate_avg"], agg["tx_rate_avg"],
                agg["rx_rate_peak"], agg["tx_rate_peak"],
                agg.get("latency_avg"), agg.get("latency_p95"),
                agg.get("packet_loss_avg"), agg["sample_count"],
            ),
        )

    def upsert_network_hourly(self, agg: dict) -> None:
        self.execute(
            """INSERT INTO network_hourly
                   (bucket, interface, network_type, rx_bytes, tx_bytes,
                    rx_rate_avg, tx_rate_avg, latency_avg, packet_loss_avg,
                    minute_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(bucket, interface) DO UPDATE SET
                   network_type   = excluded.network_type,
                   rx_bytes       = excluded.rx_bytes,
                   tx_bytes       = excluded.tx_bytes,
                   rx_rate_avg    = excluded.rx_rate_avg,
                   tx_rate_avg    = excluded.tx_rate_avg,
                   latency_avg    = excluded.latency_avg,
                   packet_loss_avg = excluded.packet_loss_avg,
                   minute_count   = excluded.minute_count""",
            (
                agg["bucket"], agg["interface"], agg.get("network_type", "unknown"),
                agg["rx_bytes"], agg["tx_bytes"],
                agg["rx_rate_avg"], agg["tx_rate_avg"],
                agg.get("latency_avg"), agg.get("packet_loss_avg"),
                agg["minute_count"],
            ),
        )

    def network_minute_since(self, since: str, limit: int = 1000) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM network_minute WHERE bucket >= ? "
            "ORDER BY bucket ASC LIMIT ?",
            (since, limit),
        )

    def network_hourly_since(self, since: str, limit: int = 500) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM network_hourly WHERE bucket >= ? "
            "ORDER BY bucket ASC LIMIT ?",
            (since, limit),
        )

    def insert_network_event(self, event: dict) -> int:
        cur = self.execute(
            """INSERT INTO network_events
                   (timestamp, event_type, interface, network_type, detail, severity)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event["timestamp"],
                event["event_type"],
                event.get("interface", ""),
                event.get("network_type", ""),
                json.dumps(event.get("detail", {})),
                event.get("severity", "info"),
            ),
        )
        return cur.lastrowid

    def network_events_since(self, since: str, limit: int = 100) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM network_events WHERE timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (since, limit),
        )

    def prune_network_samples_before(self, before: str) -> int:
        """Delete raw samples older than `before`. Returns rows deleted."""
        cur = self._conn.execute(
            "DELETE FROM network_samples WHERE timestamp < ?", (before,)
        )
        self._conn.commit()
        return cur.rowcount

    def prune_network_minute_before(self, before: str) -> int:
        cur = self._conn.execute(
            "DELETE FROM network_minute WHERE bucket < ?", (before,)
        )
        self._conn.commit()
        return cur.rowcount

    # --- entity / relationship helpers (knowledge graph) ---------------------

    def get_or_create_entity(
        self, name: str, entity_type: str, *, description: str = "", source: str = ""
    ) -> int:
        row = self.query_one("SELECT id FROM entities WHERE name = ?", (name,))
        if row:
            return row["id"]
        cur = self.execute(
            """INSERT INTO entities (name, type, description, source, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (name, entity_type, description, source, datetime.now(UTC).isoformat()),
        )
        return cur.lastrowid

    def add_relationship(
        self, source_name: str, relation: str, target_name: str,
        *, source_type: str = "concept", target_type: str = "concept",
        source_ref: str = "",
    ) -> None:
        source_id = self.get_or_create_entity(source_name, source_type)
        target_id = self.get_or_create_entity(target_name, target_type)
        self.execute(
            """INSERT OR IGNORE INTO relationships
                   (source_id, relation, target_id, source_ref, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_id, relation, target_id, source_ref,
             datetime.now(UTC).isoformat()),
        )

    def search_entities(self, name: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM entities WHERE name LIKE ? LIMIT ?",
            (f"%{name}%", limit),
        )

    def relationships_for_entity(self, entity_name: str) -> list[sqlite3.Row]:
        return self.query(
            """SELECT r.relation, e2.name as target, e2.type as target_type,
                      r.source_ref
               FROM relationships r
               JOIN entities e1 ON r.source_id = e1.id
               JOIN entities e2 ON r.target_id = e2.id
               WHERE e1.name = ?
               UNION
               SELECT r.relation || "_of", e2.name as target, e2.type as target_type,
                      r.source_ref
               FROM relationships r
               JOIN entities e1 ON r.target_id = e1.id
               JOIN entities e2 ON r.source_id = e2.id
               WHERE e1.name = ?""",
            (entity_name, entity_name),
        )

    def entity_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM entities")
        return row["n"] if row else 0

    def relationship_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM relationships")
        return row["n"] if row else 0

    # --- session helpers ------------------------------------------------------

    def create_session(
        self, start_time: str, end_time: str, *, project: str = "", workspace: str = "",
        summary: str = "", event_count: int = 0,
    ) -> int:
        cur = self.execute(
            """INSERT INTO sessions
                   (start_time, end_time, project, workspace, summary, event_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (start_time, end_time, project, workspace, summary, event_count,
             datetime.now(UTC).isoformat()),
        )
        return cur.lastrowid

    def update_session_summary(self, session_id: int, summary: str, project: str = "") -> None:
        if project:
            self.execute(
                "UPDATE sessions SET summary = ?, project = ? WHERE id = ?",
                (summary, project, session_id),
            )
        else:
            self.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))

    def get_session(self, session_id: int) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))

    def recent_sessions(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,))

    def latest_session(self) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM sessions ORDER BY end_time DESC LIMIT 1")

    def session_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM sessions")
        return row["n"] if row else 0

    # --- workspace helpers ----------------------------------------------------

    def upsert_workspace(
        self, path: str, name: str, *, language: str = "", git_repo: bool = False,
        file_count: int = 0, last_activity: str = "",
    ) -> int:
        row = self.query_one("SELECT id FROM workspaces WHERE path = ?", (path,))
        if row:
            self.execute(
                """UPDATE workspaces SET language = ?, git_repo = ?, file_count = ?,
                   last_activity = ? WHERE path = ?""",
                (language, 1 if git_repo else 0, file_count, last_activity, path),
            )
            return row["id"]
        cur = self.execute(
            """INSERT INTO workspaces
                   (path, name, language, git_repo, file_count, last_activity, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (path, name, language, 1 if git_repo else 0, file_count, last_activity,
             datetime.now(UTC).isoformat()),
        )
        return cur.lastrowid

    def all_workspaces(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM workspaces ORDER BY last_activity DESC")

    def workspace_by_path(self, path: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM workspaces WHERE path = ?", (path,))

    def update_workspace_profile(
        self, path: str, *, profile: str, project_status: str,
    ) -> None:
        """Update the LLM-generated profile for a workspace."""
        self.execute(
            "UPDATE workspaces SET profile = ?, project_status = ?, profiled_at = ? WHERE path = ?",
            (profile, project_status, datetime.now(UTC).isoformat(), path),
        )

    def profiled_workspace_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM workspaces WHERE profile != ''")
        return row["n"] if row else 0

    # --- shell command helpers ------------------------------------------------

    def insert_shell_command(
        self, timestamp: str, command: str, *, directory: str = "",
        exit_code: int | None = None, session_id: int | None = None,
    ) -> int:
        cur = self.execute(
            """INSERT INTO shell_commands (timestamp, command, directory, exit_code, session_id)
               VALUES (?, ?, ?, ?, ?)""",
            (timestamp, command, directory, exit_code, session_id),
        )
        return cur.lastrowid

    def shell_commands_since(self, since: str, limit: int = 100) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM shell_commands WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
            (since, limit),
        )

    def search_shell_commands(self, pattern: str, limit: int = 20) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM shell_commands WHERE command LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{pattern}%", limit),
        )

    def shell_command_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM shell_commands")
        return row["n"] if row else 0

    # --- decision log helpers (hash-chained) ----------------------------------

    @staticmethod
    def _compute_decision_digest(
        prev_digest: str | None, timestamp: str, agent: str, decision: str,
        reasoning: str, kind: str, tags: list[str],
    ) -> str:
        """Compute a SHA-256 digest over the canonical JSON of a decision record.

        The digest chains to the previous decision's digest, making the log
        tamper-evident: modifying any entry breaks all subsequent digests.
        """
        payload = json.dumps(
            {
                "prev": prev_digest,
                "ts": timestamp,
                "agent": agent,
                "decision": decision,
                "rationale": reasoning,
                "kind": kind,
                "tags": tags,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log_decision(
        self, agent: str, decision: str, *, reasoning: str = "",
        confidence: float = 0.0, outcome: str = "", query: str = "",
        kind: str = "decision", tags: list[str] | None = None,
    ) -> int:
        tag_list = tags if tags is not None else []
        timestamp = datetime.now(UTC).isoformat()

        # Get the last decision's digest for chain linking.
        last = self.query_one("SELECT digest FROM decisions ORDER BY id DESC LIMIT 1")
        prev_digest = last["digest"] if last else None

        digest = self._compute_decision_digest(
            prev_digest, timestamp, agent, decision, reasoning, kind, tag_list,
        )

        cur = self.execute(
            """INSERT INTO decisions
                   (timestamp, agent, decision, reasoning_summary, confidence,
                    outcome, query, kind, tags, digest, prev_digest)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, agent, decision, reasoning, confidence, outcome, query,
             kind, json.dumps(tag_list), digest, prev_digest),
        )
        return cur.lastrowid

    def update_decision_outcome(self, decision_id: int, outcome: str) -> None:
        self.execute(
            "UPDATE decisions SET outcome = ? WHERE id = ?",
            (outcome, decision_id),
        )

    def recent_decisions(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,))

    def decision_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM decisions")
        return row["n"] if row else 0

    def verify_decision_chain(self) -> dict[str, Any]:
        """Verify the integrity of the decision hash chain.

        Returns a dict with:
          - valid: bool — True if the entire chain is intact
          - verified: int — number of records verified
          - errors: list[str] — descriptions of any broken links
        """
        rows = self.query("SELECT * FROM decisions ORDER BY id ASC")
        expected_prev: str | None = None
        errors: list[str] = []
        verified = 0
        for row in rows:
            if row["prev_digest"] != expected_prev:
                errors.append(
                    f"decision {row['id']}: chain broken — "
                    f"expected prev={expected_prev}, got prev={row['prev_digest']}"
                )
            recomputed = self._compute_decision_digest(
                row["prev_digest"], row["timestamp"], row["agent"],
                row["decision"], row["reasoning_summary"],
                row["kind"] or "decision",
                json.loads(row["tags"] or "[]"),
            )
            if recomputed != row["digest"]:
                errors.append(
                    f"decision {row['id']}: digest mismatch — "
                    f"expected {recomputed}, got {row['digest']}"
                )
            else:
                verified += 1
            expected_prev = row["digest"]
        return {"valid": len(errors) == 0, "verified": verified, "errors": errors}

    # --- task helpers ---------------------------------------------------------

    def create_task(
        self, title: str, *, source: str = "manual", related_workspace: str = "",
        due_date: str = "", priority: int = 0,
    ) -> int:
        cur = self.execute(
            """INSERT INTO tasks
                   (title, status, created_at, source, related_workspace, due_date, priority)
               VALUES (?, 'pending', ?, ?, ?, ?, ?)""",
            (title, datetime.now(UTC).isoformat(),
             source, related_workspace, due_date, priority),
        )
        return cur.lastrowid

    def complete_task(self, task_id: int) -> None:
        self.execute(
            "UPDATE tasks SET status = 'completed', completed_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), task_id),
        )

    def delete_task(self, task_id: int) -> None:
        self.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    def pending_tasks(
        self, workspace: str | None = None, limit: int = 0,
    ) -> list[sqlite3.Row]:
        if workspace:
            sql = (
                "SELECT * FROM tasks WHERE status = 'pending' AND related_workspace = ? "
                "ORDER BY priority DESC, created_at ASC"
            )
            params: tuple[Any, ...] = (workspace,)
            if limit > 0:
                sql += " LIMIT ?"
                params += (limit,)
            return self.query(sql, params)
        sql = "SELECT * FROM tasks WHERE status = 'pending' ORDER BY priority DESC, created_at ASC"
        if limit > 0:
            return self.query(sql + " LIMIT ?", (limit,))
        return self.query(sql)

    def all_tasks(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))

    def task_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM tasks WHERE status = 'pending'")
        return row["n"] if row else 0

    # --- workspace snapshot helpers -------------------------------------------

    def insert_snapshot(self, snapshot: dict) -> int:
        cur = self.execute(
            """INSERT INTO workspace_snapshots
                   (timestamp, workspace, git_branch, git_status, modified_files,
                    recent_commands, active_processes, telemetry_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot["timestamp"], snapshot["workspace"],
                snapshot.get("git_branch"), snapshot.get("git_status"),
                snapshot.get("modified_files", ""),
                snapshot.get("recent_commands", ""),
                snapshot.get("active_processes", ""),
                snapshot.get("telemetry_id"),
            ),
        )
        return cur.lastrowid

    def snapshots_since(self, since: str, workspace: str | None = None) -> list[sqlite3.Row]:
        if workspace:
            return self.query(
                "SELECT * FROM workspace_snapshots WHERE timestamp >= ? AND workspace = ? "
                "ORDER BY timestamp ASC",
                (since, workspace),
            )
        return self.query(
            "SELECT * FROM workspace_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since,),
        )

    def latest_snapshot(self, workspace: str | None = None) -> sqlite3.Row | None:
        if workspace:
            return self.query_one(
                "SELECT * FROM workspace_snapshots WHERE workspace = ? ORDER BY timestamp DESC LIMIT 1",
                (workspace,),
            )
        return self.query_one(
            "SELECT * FROM workspace_snapshots ORDER BY timestamp DESC LIMIT 1"
        )

    def snapshot_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM workspace_snapshots")
        return row["n"] if row else 0

    # --- action log helpers ---------------------------------------------------

    def log_action(
        self, action_name: str, target: str = "", parameters: str = "{}",
        *, approved: bool = False, executed: bool = False,
        result: str = "", error: str = "", duration_ms: int = 0,
    ) -> int:
        cur = self.execute(
            """INSERT INTO action_log
                   (timestamp, action_name, target, parameters, approved, executed,
                    result, error, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(UTC).isoformat(), action_name, target, parameters,
             1 if approved else 0, 1 if executed else 0, result, error, duration_ms),
        )
        return cur.lastrowid

    def recent_actions(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM action_log ORDER BY timestamp DESC LIMIT ?", (limit,))

    def action_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM action_log")
        return row["n"] if row else 0

    # --- task graph helpers ---------------------------------------------------

    def create_task_with_parent(
        self, title: str, *, parent_id: int | None = None,
        dependencies: list[int] | None = None,
        source: str = "manual", related_workspace: str = "",
        due_date: str = "", priority: int = 0,
    ) -> int:
        deps = ",".join(str(d) for d in (dependencies or []))
        cur = self.execute(
            """INSERT INTO tasks
                   (title, status, created_at, source, related_workspace,
                    due_date, priority, parent_id, dependencies)
               VALUES (?, 'pending', ?, ?, ?, ?, ?, ?, ?)""",
            (title, datetime.now(UTC).isoformat(),
             source, related_workspace, due_date, priority, parent_id, deps),
        )
        return cur.lastrowid

    def task_children(self, task_id: int) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM tasks WHERE parent_id = ? ORDER BY priority DESC", (task_id,))

    def task_dependencies(self, task_id: int) -> list[int]:
        row = self.query_one("SELECT dependencies FROM tasks WHERE id = ?", (task_id,))
        if not row or not row["dependencies"]:
            return []
        return [int(d) for d in row["dependencies"].split(",") if d.strip()]

    def task_is_blocked(self, task_id: int) -> bool:
        """A task is blocked if any dependency is not completed."""
        deps = self.task_dependencies(task_id)
        if not deps:
            return False
        for dep_id in deps:
            row = self.query_one("SELECT status FROM tasks WHERE id = ?", (dep_id,))
            if not row or row["status"] != "completed":
                return True
        return False

    def task_tree(self, root_id: int | None = None) -> list[dict]:
        """Return task tree as nested dicts."""
        if root_id is None:
            roots = self.query("SELECT * FROM tasks WHERE parent_id IS NULL ORDER BY priority DESC, created_at ASC")
        else:
            roots = self.query("SELECT * FROM tasks WHERE id = ?", (root_id,))
        tree = []
        for r in roots:
            node = dict(r)
            node["children"] = self._task_children_recursive(r["id"])
            tree.append(node)
        return tree

    def _task_children_recursive(self, parent_id: int) -> list[dict]:
        """Recursively get children of a task."""
        children = self.query("SELECT * FROM tasks WHERE parent_id = ? ORDER BY priority DESC", (parent_id,))
        result = []
        for c in children:
            node = dict(c)
            node["children"] = self._task_children_recursive(c["id"])
            result.append(node)
        return result

    # --- approval queue helpers -----------------------------------------------

    def submit_approval(
        self, action_name: str, target: str = "", parameters: str = "{}",
        reason: str = "", risk: str = "low", workflow_id: int | None = None,
    ) -> int:
        cur = self.execute(
            """INSERT INTO approval_queue
                   (timestamp, action_name, target, parameters, reason, risk, status, workflow_id)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (datetime.now(UTC).isoformat(), action_name, target,
             parameters, reason, risk, workflow_id),
        )
        return cur.lastrowid

    def decide_approval(self, approval_id: int, approved: bool, decided_by: str = "user") -> None:
        status = "approved" if approved else "denied"
        self.execute(
            "UPDATE approval_queue SET status = ?, decided_at = ?, decided_by = ? WHERE id = ?",
            (status, datetime.now(UTC).isoformat(), decided_by, approval_id),
        )

    def pending_approvals(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM approval_queue WHERE status = 'pending' ORDER BY timestamp ASC")

    def approval_by_id(self, approval_id: int) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM approval_queue WHERE id = ?", (approval_id,))

    def approval_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM approval_queue WHERE status = 'pending'")
        return row["n"] if row else 0

    # --- user model helpers ---------------------------------------------------

    def set_user_model_entry(
        self, key: str, value: str, *, confidence: float = 0.5, source: str = "observed",
        metadata: str | None = None,
    ) -> None:
        existing = self.query_one("SELECT id FROM user_model WHERE key = ?", (key,))
        if existing:
            self.execute(
                "UPDATE user_model SET value = ?, confidence = ?, source = ?, metadata = ?, updated_at = ? WHERE key = ?",
                (value, confidence, source, metadata, datetime.now(UTC).isoformat(), key),
            )
        else:
            self.execute(
                """INSERT INTO user_model (key, value, confidence, source, metadata, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key, value, confidence, source, metadata, datetime.now(UTC).isoformat()),
            )

    def get_user_model_entry(self, key: str) -> dict | None:
        row = self.query_one("SELECT * FROM user_model WHERE key = ?", (key,))
        return dict(row) if row else None

    def all_user_model_entries(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM user_model ORDER BY key ASC")

    def delete_user_model_entry(self, key: str) -> None:
        self.execute("DELETE FROM user_model WHERE key = ?", (key,))

    # --- personal memory helpers ----------------------------------------------

    def upsert_personal_memory(
        self,
        memory_id: str,
        content_hash: str,
        content: str,
        *,
        title: str = "",
        tags: str = "[]",
        source: str = "manual",
        source_ref: str = "",
        author_id: str = "",
        archive_status: str = "pending",
        archive_path: str = "",
        archive_hash: str = "",
        archive_error: str = "",
        indexed_at: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        self.execute(
            """INSERT INTO personal_memories (
                   id, content_hash, title, content, tags, source, source_ref,
                   author_id, archive_status, archive_path, archive_hash,
                   archive_error, created_at, updated_at, indexed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   content_hash = excluded.content_hash,
                   title = excluded.title,
                   content = excluded.content,
                   tags = excluded.tags,
                   source = excluded.source,
                   source_ref = excluded.source_ref,
                   author_id = excluded.author_id,
                   archive_status = excluded.archive_status,
                   archive_path = excluded.archive_path,
                   archive_hash = excluded.archive_hash,
                   archive_error = excluded.archive_error,
                   updated_at = excluded.updated_at,
                   indexed_at = excluded.indexed_at""",
            (
                memory_id, content_hash, title, content, tags, source, source_ref,
                author_id, archive_status, archive_path, archive_hash,
                archive_error, now, now, indexed_at,
            ),
        )

    def personal_memory_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        row = self.query_one(
            "SELECT * FROM personal_memories WHERE content_hash = ?",
            (content_hash,),
        )
        return dict(row) if row else None

    def personal_memory(self, memory_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM personal_memories WHERE id = ?", (memory_id,))
        return dict(row) if row else None

    def update_personal_memory_archive(
        self,
        memory_id: str,
        *,
        status: str,
        archive_path: str = "",
        archive_hash: str = "",
        error: str = "",
    ) -> None:
        self.execute(
            """UPDATE personal_memories
               SET archive_status = ?, archive_path = ?, archive_hash = ?,
                   archive_error = ?, updated_at = ?
               WHERE id = ?""",
            (
                status, archive_path, archive_hash, error,
                datetime.now(UTC).isoformat(), memory_id,
            ),
        )

    def pending_personal_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.query(
            """SELECT * FROM personal_memories
               WHERE archive_status IN ('pending', 'failed')
               ORDER BY created_at ASC LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in rows]

    def list_personal_memories(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM personal_memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    # --- workflow run helpers -------------------------------------------------

    def start_workflow_run(
        self, workflow: str, steps_total: int,
        *, workflow_id: str = "", version: int = 1, parent_run_id: int | None = None,
    ) -> int:
        cur = self.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done,
                workflow_id, version, parent_run_id, heartbeat_at)
               VALUES (?, ?, 'running', ?, 0, ?, ?, ?, ?)""",
            (datetime.now(UTC).isoformat(), workflow, steps_total,
             workflow_id, version, parent_run_id, datetime.now(UTC).isoformat()),
        )
        return cur.lastrowid

    def update_workflow_run(self, run_id: int, steps_done: int, status: str = "running") -> None:
        self.execute(
            "UPDATE workflow_runs SET steps_done = ?, status = ?, heartbeat_at = ? WHERE id = ?",
            (steps_done, status, datetime.now(UTC).isoformat(), run_id),
        )

    def heartbeat_workflow_run(self, run_id: int) -> None:
        """Update the heartbeat timestamp for a running workflow."""
        self.execute(
            "UPDATE workflow_runs SET heartbeat_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), run_id),
        )

    def transition_workflow_run(self, run_id: int, status: str, **kwargs: Any) -> None:
        """Transition a workflow run to a new status, recording timestamps.

        kwargs can include: result, error, duration_ms, paused_at, cancelled_at
        """
        now = datetime.now(UTC).isoformat()
        sets = ["status = ?"]
        vals: list[Any] = [status]
        if "result" in kwargs:
            sets.append("result = ?")
            vals.append(str(kwargs["result"])[:500])
        if "error" in kwargs:
            sets.append("error = ?")
            vals.append(str(kwargs["error"])[:500])
        if "duration_ms" in kwargs:
            sets.append("duration_ms = ?")
            vals.append(kwargs["duration_ms"])
        if status in ("completed", "failed", "cancelled", "timed_out", "crashed", "compensated"):
            sets.append("finished_at = ?")
            vals.append(now)
        if status == "paused":
            sets.append("paused_at = ?")
            vals.append(now)
        if status == "cancelled":
            sets.append("cancelled_at = ?")
            vals.append(now)
        vals.append(run_id)
        self.execute(
            f"UPDATE workflow_runs SET {', '.join(sets)} WHERE id = ?",
            vals,
        )

    def finish_workflow_run(
        self, run_id: int, status: str, result: str = "", error: str = "", duration_ms: int = 0,
    ) -> None:
        self.transition_workflow_run(
            run_id, status, result=result, error=error, duration_ms=duration_ms,
        )

    def get_workflow_run(self, run_id: int) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM workflow_runs WHERE id = ?", (run_id,))

    def recent_workflow_runs(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM workflow_runs ORDER BY timestamp DESC LIMIT ?", (limit,))

    def workflow_runs_by_status(self, status: str) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM workflow_runs WHERE status = ? ORDER BY timestamp", (status,))

    def stale_workflow_runs(self, max_heartbeat_age_seconds: float = 300) -> list[sqlite3.Row]:
        """Find runs with status='running' whose heartbeat is older than the threshold."""
        cutoff = (datetime.now(UTC).timestamp()) - max_heartbeat_age_seconds
        rows = self.workflow_runs_by_status("running")
        stale = []
        for row in rows:
            hb = row["heartbeat_at"]
            if not hb:
                # No heartbeat at all — definitely stale.
                stale.append(row)
                continue
            try:
                hb_dt = datetime.fromisoformat(hb.replace("Z", "+00:00"))
                if hb_dt.timestamp() < cutoff:
                    stale.append(row)
            except (ValueError, TypeError):
                stale.append(row)
        return stale

    def mark_stale_runs_as_crashed(self, max_heartbeat_age_seconds: float = 300) -> list[int]:
        """Detect and mark stale 'running' workflows as 'crashed'. Returns run IDs."""
        stale = self.stale_workflow_runs(max_heartbeat_age_seconds)
        run_ids = []
        for row in stale:
            self.transition_workflow_run(row["id"], "crashed", error="heartbeat timeout — process likely crashed")
            run_ids.append(row["id"])
        return run_ids

    # --- workflow definition helpers ------------------------------------------

    def save_workflow_definition(
        self, workflow_id: str, name: str, version: int, definition: str,
    ) -> None:
        self.execute(
            """INSERT OR REPLACE INTO workflow_definitions
               (id, name, version, definition, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (workflow_id, name, version, definition, datetime.now(UTC).isoformat()),
        )

    def get_workflow_definition(self, name: str, version: int | None = None) -> sqlite3.Row | None:
        if version is not None:
            return self.query_one(
                "SELECT * FROM workflow_definitions WHERE name = ? AND version = ?",
                (name, version),
            )
        return self.query_one(
            "SELECT * FROM workflow_definitions WHERE name = ? ORDER BY version DESC LIMIT 1",
            (name,),
        )

    def list_workflow_definitions(self) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM workflow_definitions ORDER BY name, version",
        )

    # --- workflow step run helpers --------------------------------------------

    def start_workflow_step_run(
        self, run_id: int, step_id: str, step_name: str, step_index: int, attempt: int = 1,
    ) -> int:
        cur = self.execute(
            """INSERT INTO workflow_step_runs
               (run_id, step_id, step_name, step_index, status, attempt, started_at)
               VALUES (?, ?, ?, ?, 'running', ?, ?)""",
            (run_id, step_id, step_name, step_index, attempt, datetime.now(UTC).isoformat()),
        )
        return cur.lastrowid

    def finish_workflow_step_run(
        self, step_run_id: int, status: str, data: str = "", verification: str = "",
        error: str = "", duration_ms: int = 0,
    ) -> None:
        self.execute(
            """UPDATE workflow_step_runs
               SET status = ?, data = ?, verification = ?, error = ?, duration_ms = ?, finished_at = ?
               WHERE id = ?""",
            (status, data[:2000], verification, error[:500], duration_ms,
             datetime.now(UTC).isoformat(), step_run_id),
        )

    def step_runs_for_workflow(self, run_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM workflow_step_runs WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        )

    def completed_step_runs_for_workflow(self, run_id: int) -> list[sqlite3.Row]:
        """Get only successfully completed step runs (for resume)."""
        return self.query(
            "SELECT * FROM workflow_step_runs WHERE run_id = ? AND status = 'completed' ORDER BY step_index",
            (run_id,),
        )

    # --- workflow checkpoint helpers -------------------------------------------

    def save_checkpoint(self, run_id: int, step_index: int, context: str) -> int:
        """Save a workflow context checkpoint after completing a step."""
        cur = self.execute(
            """INSERT INTO workflow_checkpoints (run_id, step_index, context, created_at)
               VALUES (?, ?, ?, ?)""",
            (run_id, step_index, context, datetime.now(UTC).isoformat()),
        )
        return cur.lastrowid

    def latest_checkpoint(self, run_id: int) -> sqlite3.Row | None:
        """Get the most recent checkpoint for a workflow run."""
        return self.query_one(
            "SELECT * FROM workflow_checkpoints WHERE run_id = ? ORDER BY step_index DESC LIMIT 1",
            (run_id,),
        )

    def checkpoints_for_workflow(self, run_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM workflow_checkpoints WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        )

    # --- workflow side effect helpers ------------------------------------------

    def record_side_effect(
        self, run_id: int, step_id: str, step_name: str,
        effect_type: str, idempotency_key: str, details: str = "{}",
    ) -> int:
        """Record that a side effect was executed during a workflow run."""
        cur = self.execute(
            """INSERT INTO workflow_side_effects
               (run_id, step_id, step_name, effect_type, idempotency_key, executed_at, details)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, step_id, step_name, effect_type, idempotency_key,
             datetime.now(UTC).isoformat(), details),
        )
        return cur.lastrowid

    def side_effects_for_workflow(self, run_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM workflow_side_effects WHERE run_id = ? ORDER BY id",
            (run_id,),
        )

    def side_effect_exists(self, idempotency_key: str) -> bool:
        """Check if a side effect with this idempotency key has already been executed."""
        row = self.query_one(
            "SELECT COUNT(*) as cnt FROM workflow_side_effects WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        return row is not None and row["cnt"] > 0

    # --- MCP request helpers --------------------------------------------------

    def log_mcp_request(
        self,
        *,
        client_name: str = "",
        kind: str,
        name: str,
        success: bool = True,
        latency_ms: int = 0,
        error: str = "",
        timestamp_override: datetime | None = None,
    ) -> int:
        ts = timestamp_override if timestamp_override else datetime.now(UTC)
        cur = self.execute(
            """INSERT INTO mcp_requests
                   (timestamp, client_name, kind, name, success, latency_ms, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ts.isoformat(), client_name, kind, name,
             1 if success else 0, latency_ms, error),
        )
        return cur.lastrowid

    def recent_mcp_requests(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM mcp_requests ORDER BY timestamp DESC LIMIT ?", (limit,)
        )

    def mcp_request_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM mcp_requests")
        return row["n"] if row else 0

    def mcp_request_stats(self) -> dict[str, Any]:
        """Aggregate stats for MCP requests."""
        total = self.query_one("SELECT COUNT(*) as n FROM mcp_requests")
        success = self.query_one("SELECT COUNT(*) as n FROM mcp_requests WHERE success = 1")
        avg_lat = self.query_one("SELECT AVG(latency_ms) as v FROM mcp_requests WHERE success = 1")
        by_kind = self.query(
            "SELECT kind, COUNT(*) as count, AVG(latency_ms) as avg_lat "
            "FROM mcp_requests GROUP BY kind"
        )
        return {
            "total": total["n"] if total else 0,
            "success": success["n"] if success else 0,
            "avg_latency_ms": round(avg_lat["v"], 1) if avg_lat and avg_lat["v"] else 0,
            "by_kind": {r["kind"]: {"count": r["count"], "avg_latency_ms": round(r["avg_lat"], 1)} for r in by_kind},
        }

    # --- file read helpers ----------------------------------------------------

    def log_file_read(self, *, path: str, lines: int = 0, timestamp_override: datetime | None = None) -> int:
        """Log a file read for behavior tracking."""
        ts = timestamp_override if timestamp_override else datetime.now(UTC)
        cur = self.execute(
            "INSERT INTO file_reads (timestamp, path, lines) VALUES (?, ?, ?)",
            (ts.isoformat(), path, lines),
        )
        return cur.lastrowid

    def file_reads_since(self, since_iso: str) -> list[sqlite3.Row]:
        """Get file reads since a timestamp."""
        return self.query(
            "SELECT * FROM file_reads WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since_iso,),
        )

    def file_read_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) as n FROM file_reads")
        return row["n"] if row else 0

    def unique_files_read(self) -> int:
        row = self.query_one("SELECT COUNT(DISTINCT path) as n FROM file_reads")
        return row["n"] if row else 0

    # --- context evaluation helpers -------------------------------------------

    def save_context_evaluation(
        self,
        *,
        workspace: float,
        session: float,
        tasks: float,
        knowledge: float,
        preferences: float,
        activity: float,
        overall: float,
        details: str = "{}",
    ) -> int:
        cur = self.execute(
            """INSERT INTO context_evaluations
                   (timestamp, workspace, session, tasks, knowledge,
                    preferences, activity, overall, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(UTC).isoformat(), workspace, session, tasks,
             knowledge, preferences, activity, overall, details),
        )
        return cur.lastrowid

    def recent_context_evaluations(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM context_evaluations ORDER BY timestamp DESC LIMIT ?", (limit,)
        )

    def latest_context_evaluation(self) -> sqlite3.Row | None:
        return self.query_one(
            "SELECT * FROM context_evaluations ORDER BY timestamp DESC LIMIT 1"
        )

    # --- retention / rotation -------------------------------------------------

    # Tables that grow unboundedly with interactions, mapped to their timestamp
    # column.  These are the tables that would push DB size past 10 GB at 50 M
    # interactions.  Order matters: child tables (process_snapshots) are pruned
    # before their parent (telemetry_snapshots) to respect FK constraints.
    _RETENTION_TABLES: list[tuple[str, str]] = [
        ("process_snapshots", "telemetry_id"),      # child of telemetry_snapshots
        ("telemetry_snapshots", "timestamp"),
        ("events", "timestamp"),
        ("shell_commands", "timestamp"),
        ("action_log", "timestamp"),
        ("mcp_requests", "timestamp"),
        ("context_evaluations", "timestamp"),
        ("workspace_snapshots", "timestamp"),
        ("file_reads", "timestamp"),
        ("decisions", "timestamp"),
        ("workflow_runs", "timestamp"),
        ("sessions", "end_time"),
    ]

    def apply_retention(self, days: int = 30) -> dict[str, int]:
        """Delete rows older than *days* days from high-growth tables.

        Returns a mapping of table name to number of rows deleted.
        Addresses scaling blocker #655: projected >10 GB SQLite at 50 M
        interactions.  By pruning transient telemetry/event/log data older
        than a configurable window we keep the DB bounded.

        Tables that are canonical (file_hashes, git_commits, entities,
        relationships, tasks, user_model, workspaces) are NOT pruned —
        they are small or semantically permanent.
        """
        cutoff = (datetime.now(UTC) - _timedelta(days=days)).isoformat()
        deleted: dict[str, int] = {}
        with self.transaction():
            for table, ts_col in self._RETENTION_TABLES:
                # process_snapshots has no timestamp column; prune by
                # joining to telemetry_snapshots.
                if table == "process_snapshots":
                    cur = self.execute(
                        """DELETE FROM process_snapshots
                           WHERE telemetry_id IN (
                               SELECT id FROM telemetry_snapshots
                               WHERE timestamp < ?
                           )""",
                        (cutoff,),
                    )
                else:
                    cur = self.execute(
                        f"DELETE FROM {table} WHERE {ts_col} < ?",
                        (cutoff,),
                    )
                deleted[table] = cur.rowcount
        log.info("db.retention.applied", days=days, cutoff=cutoff, deleted=deleted)
        return deleted

    def db_size_mb(self) -> float:
        """Return the current database file size in megabytes."""
        try:
            return self.path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0


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
