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

CREATE TABLE IF NOT EXISTS workflow_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    workflow    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    steps_total INTEGER NOT NULL DEFAULT 0,
    steps_done  INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    duration_ms INTEGER
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

    # --- decision log helpers -------------------------------------------------

    def log_decision(
        self, agent: str, decision: str, *, reasoning: str = "",
        confidence: float = 0.0, outcome: str = "", query: str = "",
    ) -> int:
        cur = self.execute(
            """INSERT INTO decisions
                   (timestamp, agent, decision, reasoning_summary, confidence, outcome, query)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(UTC).isoformat(), agent, decision,
             reasoning, confidence, outcome, query),
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

    def pending_tasks(self, workspace: str | None = None) -> list[sqlite3.Row]:
        if workspace:
            return self.query(
                "SELECT * FROM tasks WHERE status = 'pending' AND related_workspace = ? "
                "ORDER BY priority DESC, created_at ASC",
                (workspace,),
            )
        return self.query(
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY priority DESC, created_at ASC"
        )

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
    ) -> None:
        existing = self.query_one("SELECT id FROM user_model WHERE key = ?", (key,))
        if existing:
            self.execute(
                "UPDATE user_model SET value = ?, confidence = ?, source = ?, updated_at = ? WHERE key = ?",
                (value, confidence, source, datetime.now(UTC).isoformat(), key),
            )
        else:
            self.execute(
                """INSERT INTO user_model (key, value, confidence, source, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, value, confidence, source, datetime.now(UTC).isoformat()),
            )

    def get_user_model_entry(self, key: str) -> dict | None:
        row = self.query_one("SELECT * FROM user_model WHERE key = ?", (key,))
        return dict(row) if row else None

    def all_user_model_entries(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM user_model ORDER BY key ASC")

    def delete_user_model_entry(self, key: str) -> None:
        self.execute("DELETE FROM user_model WHERE key = ?", (key,))

    # --- workflow run helpers -------------------------------------------------

    def start_workflow_run(self, workflow: str, steps_total: int) -> int:
        cur = self.execute(
            """INSERT INTO workflow_runs (timestamp, workflow, status, steps_total, steps_done)
               VALUES (?, ?, 'running', ?, 0)""",
            (datetime.now(UTC).isoformat(), workflow, steps_total),
        )
        return cur.lastrowid

    def update_workflow_run(self, run_id: int, steps_done: int, status: str = "running") -> None:
        self.execute(
            "UPDATE workflow_runs SET steps_done = ?, status = ? WHERE id = ?",
            (steps_done, status, run_id),
        )

    def finish_workflow_run(
        self, run_id: int, status: str, result: str = "", error: str = "", duration_ms: int = 0,
    ) -> None:
        self.execute(
            "UPDATE workflow_runs SET status = ?, result = ?, error = ?, duration_ms = ? WHERE id = ?",
            (status, result[:500], error[:500], duration_ms, run_id),
        )

    def recent_workflow_runs(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM workflow_runs ORDER BY timestamp DESC LIMIT ?", (limit,))


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
