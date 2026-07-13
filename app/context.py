"""Context builder — assembles a universal Context object for agents.

Instead of each agent fetching its own data from different sources, the
Context builder gathers everything relevant to a query into a single object.
Agents receive the Context and work with what's there.

This is intentionally a dataclass, not a framework. It grows as we add more
data sources. Agents that don't need a field just ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from app.db import Database
from app.log import get_logger
from app.retrieval.base import Retriever

log = get_logger(__name__)


@dataclass
class Context:
    """Universal context passed to agents."""

    user_query: str = ""
    active_session: dict[str, Any] | None = None
    workspace: dict[str, Any] | None = None
    relevant_files: list[dict[str, Any]] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    recent_commits: list[dict[str, Any]] = field(default_factory=list)
    recent_telemetry: list[dict[str, Any]] = field(default_factory=list)
    recent_commands: list[dict[str, Any]] = field(default_factory=list)
    graph_entities: list[dict[str, Any]] = field(default_factory=list)
    rag_results: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Render the context as a text block for LLM prompts."""
        parts: list[str] = []

        if self.active_session:
            s = self.active_session
            parts.append(f"Active session: {s.get('start_time', '?')[:19]} to {s.get('end_time', '?')[:19]}")
            if s.get("project"):
                parts.append(f"  Project: {s['project']}")
            if s.get("event_count"):
                parts.append(f"  Events: {s['event_count']}")

        if self.workspace:
            w = self.workspace
            parts.append(f"Workspace: {w.get('name', '?')} ({w.get('path', '?')})")
            if w.get("language"):
                parts.append(f"  Language: {w['language']}")

        if self.recent_commits:
            parts.append(f"\nRecent commits ({len(self.recent_commits)}):")
            for c in self.recent_commits[:5]:
                parts.append(f"  {c.get('hash', '?')[:8]} {c.get('author_date', '?')[:10]} {c.get('message', '?')[:60]}")

        if self.recent_events:
            parts.append(f"\nRecent events ({len(self.recent_events)}):")
            for e in self.recent_events[:10]:
                parts.append(f"  {e.get('timestamp', '?')[:19]} {e.get('type', '?')}")

        if self.recent_commands:
            parts.append(f"\nRecent commands ({len(self.recent_commands)}):")
            for c in self.recent_commands[:5]:
                parts.append(f"  {c.get('timestamp', '?')[:19]} {c.get('command', '?')[:60]}")

        if self.recent_telemetry:
            t = self.recent_telemetry[-1]
            parts.append(f"\nLatest telemetry: CPU={t.get('cpu_percent', 0):.1f}% "
                        f"Mem={t.get('memory_used_gb', 0):.1f}/{t.get('memory_total_gb', 0):.1f}GB")

        if self.graph_entities:
            parts.append(f"\nRelated concepts ({len(self.graph_entities)}):")
            for e in self.graph_entities[:5]:
                parts.append(f"  {e.get('name', '?')} ({e.get('type', '?')})")

        if self.rag_results:
            parts.append(f"\nRelevant documents ({len(self.rag_results)}):")
            for r in self.rag_results[:3]:
                parts.append(f"  [{r.get('score', 0):.2f}] {r.get('source', '?')}")

        return "\n".join(parts) if parts else "(no context available)"


def build_context(
    db: Database,
    query: str,
    *,
    retriever: Retriever | None = None,
    lookback_hours: int = 24,
    top_k: int = 5,
) -> Context:
    """Build a Context object for a user query."""
    ctx = Context(user_query=query)
    now = datetime.now(UTC)
    since = (now - timedelta(hours=lookback_hours)).isoformat()

    # Recent events.
    events = db.query_events(since=since, limit=50)
    ctx.recent_events = [dict(e) for e in events]

    # Recent commits.
    ctx.recent_commits = [dict(c) for c in db.query(
        "SELECT * FROM git_commits ORDER BY author_date DESC LIMIT 10"
    )]

    # Recent telemetry.
    ctx.recent_telemetry = [dict(t) for t in db.telemetry_since(since, limit=20)]

    # Recent shell commands.
    ctx.recent_commands = [dict(c) for c in db.shell_commands_since(since, limit=20)]

    # Latest session.
    latest = db.latest_session()
    if latest:
        ctx.active_session = dict(latest)

    # Recent sessions.
    ctx.sessions = [dict(s) for s in db.recent_sessions(limit=5)]

    # Active workspace.
    from app.workspace import detect_active_workspace

    ws = detect_active_workspace(db)
    if ws:
        ctx.workspace = ws

    # Graph entities related to the query.
    # Search for entities that match keywords in the query.
    words = [w for w in query.split() if len(w) > 2]
    for word in words[:3]:
        entities = db.search_entities(word, limit=5)
        for e in entities:
            ctx.graph_entities.append(dict(e))
    # Deduplicate by name.
    seen = set()
    unique = []
    for e in ctx.graph_entities:
        if e["name"] not in seen:
            seen.add(e["name"])
            unique.append(e)
    ctx.graph_entities = unique[:10]

    # RAG results.
    if retriever:
        try:
            results = retriever.search(query, top_k=top_k)
            ctx.rag_results = [
                {"text": r.text[:200], "source": r.source, "score": r.score,
                 "metadata": r.metadata}
                for r in results
            ]
        except Exception as e:
            log.warning("context.rag_failed", error=str(e))

    log.debug("context.built",
              events=len(ctx.recent_events),
              commits=len(ctx.recent_commits),
              telemetry=len(ctx.recent_telemetry),
              commands=len(ctx.recent_commands),
              entities=len(ctx.graph_entities),
              rag=len(ctx.rag_results))
    return ctx
