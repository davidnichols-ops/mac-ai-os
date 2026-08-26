"""Event types for the Mac AI OS event bus.

Every component publishes events; every observer subscribes. This keeps
features decoupled — the telemetry collector doesn't know about the indexer,
the indexer doesn't know about the timeline, etc.

Events are typed dataclasses so subscribers can filter by type and access
payload fields with autocomplete. All events carry a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Event:
    """Base event. `type` is a string enum; `payload` is arbitrary metadata."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": self.payload, "timestamp": self.timestamp}


# --- Event type constants ----------------------------------------------------

FILE_INDEXED = "FILE_INDEXED"
FILE_SKIPPED = "FILE_SKIPPED"
FILE_DELETED = "FILE_DELETED"
GIT_COMMIT_INDEXED = "GIT_COMMIT_INDEXED"
GIT_REPO_INDEXED = "GIT_REPO_INDEXED"
MODEL_LOADED = "MODEL_LOADED"
MODEL_GENERATED = "MODEL_GENERATED"
COMMAND_RUN = "COMMAND_RUN"
INDEX_UPDATED = "INDEX_UPDATED"
SYSTEM_SNAPSHOT = "SYSTEM_SNAPSHOT"
WATCHER_STARTED = "WATCHER_STARTED"
WATCHER_STOPPED = "WATCHER_STOPPED"
ENTITY_EXTRACTED = "ENTITY_EXTRACTED"
CONVERSATION_MESSAGE = "CONVERSATION_MESSAGE"
ACTION_EXECUTED = "ACTION_EXECUTED"
ACTION_DENIED = "ACTION_DENIED"
TASK_CREATED = "TASK_CREATED"
TASK_COMPLETED = "TASK_COMPLETED"
SNAPSHOT_CAPTURED = "SNAPSHOT_CAPTURED"
DECISION_LOGGED = "DECISION_LOGGED"
WORKFLOW_STARTED = "WORKFLOW_STARTED"
WORKFLOW_STEP_COMPLETED = "WORKFLOW_STEP_COMPLETED"
WORKFLOW_STEP_FAILED = "WORKFLOW_STEP_FAILED"
WORKFLOW_COMPLETED = "WORKFLOW_COMPLETED"
WORKFLOW_CANCELLED = "WORKFLOW_CANCELLED"
WORKFLOW_CRASHED = "WORKFLOW_CRASHED"
WORKFLOW_PAUSED = "WORKFLOW_PAUSED"
WORKFLOW_RESUMED = "WORKFLOW_RESUMED"
MCP_RESOURCE_READ = "MCP_RESOURCE_READ"
MCP_TOOL_CALLED = "MCP_TOOL_CALLED"
SESSION_STARTED = "SESSION_STARTED"
SESSION_ENDED = "SESSION_ENDED"
AGENT_SESSION_STARTED = "AGENT_SESSION_STARTED"
AGENT_SESSION_COMPLETED = "AGENT_SESSION_COMPLETED"

# Network intelligence layer (Phase 5.7).
NETWORK_SAMPLE = "NETWORK_SAMPLE"
NETWORK_INTERRUPTED = "NETWORK_INTERRUPTED"
NETWORK_RECONNECTED = "NETWORK_RECONNECTED"
NETWORK_DEGRADED = "NETWORK_DEGRADED"
NETWORK_INTERFACE_CHANGED = "NETWORK_INTERFACE_CHANGED"


# --- Convenience factory functions ------------------------------------------


def file_indexed(path: str, chunks: int, source: str) -> Event:
    return Event(FILE_INDEXED, {"path": path, "chunks": chunks, "source": source})


def file_skipped(path: str, source: str) -> Event:
    return Event(FILE_SKIPPED, {"path": path, "source": source})


def file_deleted(path: str, source: str) -> Event:
    return Event(FILE_DELETED, {"path": path, "source": source})


def git_repo_indexed(repo: str, commits: int) -> Event:
    return Event(GIT_REPO_INDEXED, {"repo": repo, "commits": commits})


def model_loaded(provider: str, model: str) -> Event:
    return Event(MODEL_LOADED, {"provider": provider, "model": model})


def system_snapshot(cpu: float, mem_used: float, mem_total: float) -> Event:
    return Event(SYSTEM_SNAPSHOT, {
        "cpu_percent": cpu,
        "memory_used_gb": mem_used,
        "memory_total_gb": mem_total,
    })


def command_run(command: str, exit_code: int) -> Event:
    return Event(COMMAND_RUN, {"command": command, "exit_code": exit_code})
