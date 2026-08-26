"""Normalize external-agent sessions into provenance-bearing MAOS records.

Supports: Devin, Claude, Grok, Codex, Gemini, Cursor, Windsurf, OpenHands.
Each provider has a dedicated parser that extracts session metadata and
content without promoting claims (all records are classification=
observed_external_session, trusted=False).

The indexer also remembers where to look for nuggets of data via the
DATA_SOURCES registry — a structured list of all known AI tool data
locations on this machine. New tools can be added by appending to
DATA_SOURCES and implementing a _parse_<provider> function.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Data source registry — the indexer's map of where AI data hides.
# Each entry: provider name, root path (relative to home), glob pattern,
# and a flag indicating whether the source contains session transcripts
# vs. just config/metadata.
# ---------------------------------------------------------------------------

DATA_SOURCES: list[dict[str, str]] = [
    {"provider": "devin", "path": ".local/share/devin/cli/transcripts", "pattern": "*.json", "kind": "transcript"},
    {"provider": "devin", "path": ".local/share/devin/cli/logs", "pattern": "*.log.gz", "kind": "log"},
    {"provider": "devin", "path": ".local/share/devin/cli/sessions.db", "pattern": "*.db", "kind": "database"},
    {"provider": "grok", "path": ".grok/memory", "pattern": "*/sessions/*.md", "kind": "transcript"},
    {"provider": "grok", "path": ".grok/memtrace", "pattern": "*.jsonl", "kind": "trace"},
    {"provider": "grok", "path": ".grok/logs", "pattern": "unified.jsonl", "kind": "log"},
    {"provider": "grok", "path": ".grok/sessions", "pattern": "*", "kind": "session_dir"},
    {"provider": "claude", "path": ".claude/projects", "pattern": "**/*.jsonl", "kind": "transcript"},
    {"provider": "gemini", "path": ".gemini/history", "pattern": "**/*.json", "kind": "transcript"},
    {"provider": "gemini", "path": ".gemini/tmp", "pattern": "**/logs.json", "kind": "log"},
    {"provider": "codex", "path": ".codex", "pattern": ".codex-global-state.json", "kind": "state"},
    {"provider": "cursor", "path": ".cursor", "pattern": "**/*.json", "kind": "config"},
    {"provider": "windsurf", "path": ".windsurf", "pattern": "**/*.json", "kind": "config"},
    {"provider": "openhands", "path": ".openhands", "pattern": "**/*.json", "kind": "transcript"},
    {"provider": "continue", "path": ".continue", "pattern": "**/*.json", "kind": "config"},
    {"provider": "cline", "path": ".cline", "pattern": "**/*.json", "kind": "config"},
    {"provider": "echo", "path": ".echo", "pattern": "**/*.json", "kind": "config"},
    {"provider": "ollama", "path": ".ollama", "pattern": "**/*.json", "kind": "config"},
    {"provider": "lmstudio", "path": ".lmstudio", "pattern": "**/*.json", "kind": "config"},
]


class SessionParser(Protocol):
    """Protocol for provider-specific session parsers."""

    def __call__(self, path: Path, raw: str) -> tuple[str, str]: ...


def _record(
    provider: str,
    source: Path,
    content: str,
    *,
    session_id: str = "",
    kind: str = "transcript",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "session_id": session_id or source.stem,
        "source_path": str(source),
        "observed_at": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(),
        "classification": "observed_external_session",
        "trusted": False,
        "kind": kind,
        "content": content[:100_000],
    }


# ---------------------------------------------------------------------------
# Provider-specific parsers
# ---------------------------------------------------------------------------

def _parse_devin(path: Path, raw: str) -> tuple[str, str]:
    """Parse Devin ATIF-v1.7 transcript JSON."""
    parsed = json.loads(raw)
    content = json.dumps(
        {
            "agent": parsed.get("agent"),
            "steps": parsed.get("steps"),
            "final_metrics": parsed.get("final_metrics"),
        },
        ensure_ascii=False,
    )
    session_id = str(parsed.get("session_id", path.stem))
    return session_id, content


def _parse_grok_md(path: Path, raw: str) -> tuple[str, str]:
    """Parse Grok markdown session files."""
    return path.stem, raw


def _parse_grok_trace(path: Path, raw: str) -> tuple[str, str]:
    """Parse Grok memtrace JSONL files — one JSON object per line."""
    lines = raw.strip().splitlines()
    summaries: list[str] = []
    for line in lines[:200]:  # cap at 200 lines to stay within content limit
        try:
            obj = json.loads(line)
            role = obj.get("role", "?")
            content = obj.get("content", "")[:500]
            summaries.append(f"[{role}] {content}")
        except (json.JSONDecodeError, UnicodeError):
            continue
    return path.stem, "\n".join(summaries)


def _parse_claude(path: Path, raw: str) -> tuple[str, str]:
    """Parse Claude Code JSONL session files.

    Each line is a JSON object with types: queue-operation, user, assistant,
    attachment. We extract user prompts and assistant responses.
    """
    session_id = path.stem
    messages: list[dict[str, Any]] = []
    for line in raw.strip().splitlines():
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, UnicodeError):
            continue
        msg_type = obj.get("type", "")
        if msg_type == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                messages.append({"role": "user", "content": content[:1000], "timestamp": obj.get("timestamp")})
        elif msg_type == "assistant":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                content = " ".join(text_parts)
            if isinstance(content, str) and content:
                messages.append({"role": "assistant", "content": content[:1000], "timestamp": obj.get("timestamp")})
        elif msg_type == "queue-operation" and obj.get("operation") == "enqueue":
            session_id = obj.get("sessionId", session_id)
    content = json.dumps({"messages": messages[:100], "count": len(messages)}, ensure_ascii=False)
    return session_id, content


def _parse_gemini(path: Path, raw: str) -> tuple[str, str]:
    """Parse Gemini logs.json — typically an array of log entries."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError):
        return path.stem, raw[:5000]
    if isinstance(data, list):
        content = json.dumps({"entries": len(data), "sample": data[:20]}, ensure_ascii=False)
    else:
        content = json.dumps(data, ensure_ascii=False)
    return path.stem, content


def _parse_codex(path: Path, raw: str) -> tuple[str, str]:
    """Parse Codex global state JSON — Electron persisted state."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError):
        return path.stem, raw[:5000]
    # Extract just the meaningful parts, not the entire Electron state blob
    extracted = {
        "desktop_first_seen": data.get("desktop-first-seen-at-ms"),
        "last_completed_onboarding": data.get("last_completed_onboarding"),
        "agent_mode": data.get("agent-mode"),
    }
    return path.stem, json.dumps(extracted, ensure_ascii=False)


def _parse_jsonl_generic(path: Path, raw: str) -> tuple[str, str]:
    """Generic JSONL parser for unknown providers."""
    lines = raw.strip().splitlines()
    parsed_lines: list[str] = []
    for line in lines[:100]:
        try:
            obj = json.loads(line)
            parsed_lines.append(json.dumps(obj, ensure_ascii=False)[:500])
        except (json.JSONDecodeError, UnicodeError):
            continue
    return path.stem, "\n".join(parsed_lines)


def _parse_text_generic(path: Path, raw: str) -> tuple[str, str]:
    """Generic text parser for markdown/txt files."""
    return path.stem, raw[:50_000]


# ---------------------------------------------------------------------------
# Devin sessions.db parser
# ---------------------------------------------------------------------------

def _parse_devin_db(path: Path) -> list[tuple[str, str]]:
    """Read the Devin CLI SQLite sessions database into one record per session.

    Returns a list of (session_id, content_json) tuples. Each record is a
    bounded summary of a single session so it can be indexed without loading
    the full 4+ GB database into memory.
    """
    import sqlite3

    records: list[tuple[str, str]] = []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        sessions = cur.execute(
            "SELECT id, title, working_directory, backend_type, model, agent_mode, "
            "created_at, last_activity_at, metadata "
            "FROM sessions WHERE hidden=0 ORDER BY last_activity_at DESC LIMIT 500"
        ).fetchall()

        for srow in sessions:
            sid = str(srow["id"])
            msg_count = cur.execute(
                "SELECT COUNT(*) FROM message_nodes WHERE session_id=?", (sid,)
            ).fetchone()[0]
            prompt_count = cur.execute(
                "SELECT COUNT(*) FROM prompt_history WHERE session_id=?", (sid,)
            ).fetchone()[0]

            recent_messages = cur.execute(
                "SELECT chat_message, created_at FROM message_nodes "
                "WHERE session_id=? ORDER BY created_at DESC LIMIT 5",
                (sid,),
            ).fetchall()
            message_snippets: list[dict[str, Any]] = []
            for m in recent_messages:
                raw = m["chat_message"] or ""
                text = raw
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        text = obj.get("content") or str(obj)[:500]
                except json.JSONDecodeError:
                    pass
                message_snippets.append({"text": str(text)[:500], "created_at": m["created_at"]})

            recent_prompts = cur.execute(
                "SELECT content, timestamp FROM prompt_history "
                "WHERE session_id=? ORDER BY timestamp DESC LIMIT 5",
                (sid,),
            ).fetchall()
            prompt_snippets = [{"text": str(p["content"])[:500], "timestamp": p["timestamp"]} for p in recent_prompts]

            payload: dict[str, Any] = {
                "session_id": sid,
                "title": srow["title"],
                "working_directory": srow["working_directory"],
                "backend_type": srow["backend_type"],
                "model": srow["model"],
                "agent_mode": srow["agent_mode"],
                "created_at": srow["created_at"],
                "last_activity_at": srow["last_activity_at"],
                "metadata": srow["metadata"],
                "message_count": msg_count,
                "prompt_count": prompt_count,
                "recent_messages": message_snippets,
                "recent_prompts": prompt_snippets,
            }
            records.append((sid, json.dumps(payload, ensure_ascii=False, default=str)))

        conn.close()
    except Exception as e:
        records.append((
            f"{path.stem}-error",
            json.dumps({"error": str(e), "source": str(path)}, ensure_ascii=False),
        ))
    return records


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

PARSERS: dict[str, SessionParser] = {
    "devin": _parse_devin,
    "grok_md": _parse_grok_md,
    "grok_trace": _parse_grok_trace,
    "claude": _parse_claude,
    "gemini": _parse_gemini,
    "codex": _parse_codex,
    "jsonl_generic": _parse_jsonl_generic,
    "text_generic": _parse_text_generic,
}


def _parser_for(provider: str, kind: str) -> SessionParser:
    """Select the right parser for a provider + data kind."""
    if provider == "devin":
        return _parse_devin
    if provider == "grok":
        if kind == "trace":
            return _parse_grok_trace
        return _parse_grok_md
    if provider == "claude":
        return _parse_claude
    if provider == "gemini":
        return _parse_gemini
    if provider == "codex":
        return _parse_codex
    if kind == "transcript" or kind == "log" or kind == "trace":
        return _parse_jsonl_generic
    return _parse_text_generic


# ---------------------------------------------------------------------------
# Discovery: scan for data sources that exist on this machine
# ---------------------------------------------------------------------------

def discover_sources(*, home: Path | None = None) -> list[dict[str, Any]]:
    """Scan the home directory for known AI tool data locations.

    Returns a list of discovered sources with their paths, file counts,
    and sizes. This is the 'where to look for nuggets of data' map.
    """
    home = home or Path.home()
    discovered: list[dict[str, Any]] = []
    for entry in DATA_SOURCES:
        root = home / entry["path"]
        if not root.exists():
            continue
        if root.is_file():
            discovered.append({
                "provider": entry["provider"],
                "kind": entry["kind"],
                "path": str(root),
                "size_bytes": root.stat().st_size,
                "file_count": 1,
            })
        else:
            try:
                files = list(root.glob(entry["pattern"]))
            except OSError:
                continue
            if not files:
                continue
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            discovered.append({
                "provider": entry["provider"],
                "kind": entry["kind"],
                "path": str(root),
                "pattern": entry["pattern"],
                "file_count": len(files),
                "size_bytes": total_size,
            })
    return discovered


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest(data_dir: Path, *, home: Path | None = None) -> dict[str, Any]:
    """Collect readable AI session text from all known providers.

    Normalizes Devin, Claude, Grok, Codex, Gemini, and other AI tool
    sessions into provenance-bearing records without promoting claims.
    Also writes a data source discovery report.
    """
    home = home or Path.home()
    records: list[dict[str, Any]] = []
    providers_seen: set[str] = set()
    errors: list[dict[str, str]] = []

    for entry in DATA_SOURCES:
        provider = entry["provider"]
        root = home / entry["path"]
        if not root.exists():
            continue

        # Handle single-file sources (databases, state files)
        if root.is_file():
            if entry["kind"] == "database" and provider == "devin":
                for session_id, content in _parse_devin_db(root):
                    records.append(_record(provider, root, content, session_id=session_id, kind="database"))
                    providers_seen.add(provider)
                continue
            if entry["kind"] in ("database", "state", "config"):
                continue  # Don't try to parse binary/DB files as text
            try:
                raw = root.read_text(errors="ignore")
                parser = _parser_for(provider, entry["kind"])
                session_id, content = parser(root, raw)
                records.append(_record(provider, root, content, session_id=session_id, kind=entry["kind"]))
                providers_seen.add(provider)
            except (OSError, UnicodeError) as e:
                errors.append({"provider": provider, "path": str(root), "error": str(e)})
            continue

        # Handle directory sources
        try:
            files = sorted(root.glob(entry["pattern"]), key=lambda p: p.stat().st_mtime)
        except OSError:
            continue

        for path in files:
            if not path.is_file():
                continue
            if entry["kind"] in ("config", "session_dir"):
                continue  # Skip config-only sources for transcript ingestion
            if path.suffix == ".gz":
                continue  # Skip gzipped files — not text-parseable
            try:
                raw = path.read_text(errors="ignore")
                parser = _parser_for(provider, entry["kind"])
                session_id, content = parser(path, raw)
                records.append(_record(provider, path, content, session_id=session_id, kind=entry["kind"]))
                providers_seen.add(provider)
            except (OSError, json.JSONDecodeError, UnicodeError) as e:
                errors.append({"provider": provider, "path": str(path), "error": str(e)})
                continue

    # Write session records
    output = data_dir / "external_sessions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write data source discovery report
    sources = discover_sources(home=home)
    sources_output = data_dir / "data_sources.json"
    sources_output.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n")

    return {
        "records": len(records),
        "output": str(output),
        "providers": sorted(providers_seen),
        "data_sources": len(sources),
        "sources_report": str(sources_output),
        "errors": errors[:20],
    }
