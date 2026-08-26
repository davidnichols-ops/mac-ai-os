"""Durable identities for MAOS-connected AI sessions.

The registry is intentionally provider-neutral: Devin, Grok, Codex, local
models, and Slack can all announce the same agent/session identity.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def registry_path(data_dir: Path) -> Path:
    return data_dir / "agent_sessions.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sessions": []}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {"sessions": []}
    except (OSError, json.JSONDecodeError):
        return {"sessions": []}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def start_session(data_dir: Path, *, provider: str = "unknown", role: str = "operator", task: str = "") -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    short = uuid.uuid4().hex[:8]
    record = {
        "agent_id": f"maos-{role.lower().replace(' ', '-')}-{short}",
        "session_id": uuid.uuid4().hex,
        "provider": provider,
        "role": role,
        "task": task,
        "status": "active",
        "started_at": now,
        "ended_at": None,
        "completion": None,
    }
    path = registry_path(data_dir)
    data = _load(path)
    data.setdefault("sessions", []).append(record)
    _save(path, data)
    return record


def end_session(data_dir: Path, agent_id: str, *, status: str = "completed", summary: str = "", result: str = "") -> dict[str, Any] | None:
    path = registry_path(data_dir)
    data = _load(path)
    for record in reversed(data.get("sessions", [])):
        if record.get("agent_id") == agent_id and record.get("status") == "active":
            record["status"] = status
            record["ended_at"] = datetime.now(UTC).isoformat()
            record["completion"] = {"summary": summary, "result": result}
            _save(path, data)
            return record
    return None


def recent(data_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    return list(reversed(_load(registry_path(data_dir)).get("sessions", [])[-limit:]))
