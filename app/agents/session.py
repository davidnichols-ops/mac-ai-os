"""Session Reconstruction Agent — the "what was I working on?" agent.

This agent synthesizes a session summary from multiple data sources:
  - Git commits (what changed)
  - Events (what files were touched, what commands ran)
  - Telemetry (what the machine was doing)
  - Shell commands (what the user typed)
  - Knowledge graph (what concepts were involved)

It first tries structured data (sessions table, git commits, events). If the
question needs interpretation or synthesis, it uses the planner LLM with the
assembled context to generate a human-readable summary.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from app.agents.base import Agent, AgentResult
from app.context import Context, build_context
from app.db import Database
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message
from app.retrieval.base import Retriever
from app.sessions import detect_sessions, summarize_session

log = get_logger(__name__)

_SESSION_RE = re.compile(
    r"\b(what|which|show|tell me|summarize|recall).*(was I working|did I do|last session|"
    r"before I stopped|what happened|working on|was going on)",
    re.IGNORECASE,
)
_LAST_WORK_RE = re.compile(
    r"\b(what|which).*(last|recent|before|previous).*(work|session|doing|did|changed|committed)",
    re.IGNORECASE,
)


class SessionAgent(Agent):
    """Reconstructs what the user was working on from system memory."""

    name = "session"
    description = "Reconstructs what you were working on from git, events, telemetry, and shell history."

    def __init__(
        self,
        planner: Model,
        db: Database,
        retriever: Retriever | None = None,
    ):
        self.planner = planner
        self.db = db
        self.retriever = retriever

    def plan(self, goal: str) -> list[str]:
        return [
            "Detect sessions from event gaps if needed",
            "Gather session data (commits, events, telemetry, commands)",
            "Synthesize a summary using the planner LLM",
        ]

    def describe(self) -> str:
        return "Reconstructs what you were working on from git, events, telemetry, and shell history."

    def health(self) -> dict[str, Any]:
        return {
            "healthy": True,
            "sessions": self.db.session_count(),
            "events": self.db.event_count(),
            "commits": self.db.commit_count(),
            "commands": self.db.shell_command_count(),
        }

    def execute(self, goal: str) -> AgentResult:
        """Answer a session reconstruction query."""
        # First, make sure sessions are detected.
        if self.db.session_count() == 0 or self.db.event_count() > 0:
            detect_sessions(self.db)

        # Determine the time range to look at.
        hours = self._extract_hours(goal)
        now = datetime.now(UTC)
        since = (now - timedelta(hours=hours)).isoformat()

        # Get the latest session, or create a summary from raw data.
        latest = self.db.latest_session()
        if latest:
            session_data = summarize_session(self.db, latest["id"])
        else:
            # No sessions detected — build from raw events.
            session_data = self._raw_summary(since)

        if not session_data or not session_data.get("events"):
            return AgentResult(
                ok=True,
                answer=("I don't have enough activity data to reconstruct a session yet. "
                       "Try running some commands, indexing files, or collecting telemetry first."),
                steps=["no session data found"],
            )

        # Build context for the LLM.
        ctx = build_context(self.db, goal, retriever=self.retriever, lookback_hours=hours)

        # Format the session data as a prompt.
        prompt = self._build_prompt(goal, session_data, ctx)
        steps = [f"analyzed session with {session_data['event_count']} events, "
                 f"{session_data['commit_count']} commits, "
                 f"{session_data['command_count']} commands"]

        # Generate the summary.
        try:
            resp = self.planner.chat([
                Message("system", self._system_prompt()),
                Message("user", prompt),
            ])
            answer = resp.text.strip()
            steps.append("synthesized summary with planner LLM")

            # Save the summary to the session if we have one.
            if latest and answer:
                self.db.update_session_summary(latest["id"], answer[:500])

            return AgentResult(ok=True, answer=answer, steps=steps,
                             data={"session_id": latest["id"] if latest else None,
                                   "event_count": session_data["event_count"],
                                   "commit_count": session_data["commit_count"]})
        except Exception as e:
            log.warning("session.agent.llm_failed", error=str(e))
            # Fall back to structured summary.
            structured = self._structured_summary(session_data)
            steps.append(f"fell back to structured summary (LLM error: {e})")
            return AgentResult(ok=True, answer=structured, steps=steps,
                             data={"session_id": latest["id"] if latest else None})

    def _extract_hours(self, goal: str) -> int:
        """Extract a time range from the query (default 24 hours)."""
        match = re.search(r"(\d+)\s*(hour|day|week)", goal, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            unit = match.group(2).lower()
            if unit.startswith("hour"):
                return n
            elif unit.startswith("day"):
                return n * 24
            elif unit.startswith("week"):
                return n * 168
        # "yesterday" = 48 hours, "last" = 24 hours.
        if "yesterday" in goal.lower():
            return 48
        return 24

    def _system_prompt(self) -> str:
        return (
            "You are a session reconstruction agent for a local AI operating system. "
            "Your job is to synthesize what the user was working on based on system data: "
            "git commits, file events, telemetry, shell commands, and knowledge graph entities. "
            "Produce a concise, readable summary that includes:\n"
            "1. What project(s) were active\n"
            "2. What changed (files, commits)\n"
            "3. What commands were run\n"
            "4. System state (memory, CPU if notable)\n"
            "5. Related concepts from the knowledge graph\n"
            "6. A suggested next action\n\n"
            "Be specific and factual. Use the data provided. Don't make things up."
        )

    def _build_prompt(self, goal: str, session_data: dict, ctx: Context) -> str:
        """Build the LLM prompt from session data and context."""
        parts = [f"User question: {goal}\n"]
        parts.append(f"Context:\n{ctx.to_prompt()}\n")

        s = session_data
        parts.append("\nSession data:")
        parts.append(f"  Events: {s['event_count']}")
        parts.append(f"  Commits: {s['commit_count']}")
        parts.append(f"  Commands: {s['command_count']}")
        parts.append(f"  Telemetry snapshots: {s['telemetry_count']}")

        if s.get("files_touched"):
            parts.append(f"\nFiles touched ({len(s['files_touched'])}):")
            for f in s["files_touched"][:10]:
                parts.append(f"  {f}")

        if s.get("commits"):
            parts.append(f"\nCommits ({len(s['commits'])}):")
            for c in s["commits"][:5]:
                parts.append(f"  {c['hash'][:8]} {c['author_date'][:10]} {c['message'][:60]}")

        if s.get("commands"):
            parts.append(f"\nCommands ({len(s['commands'])}):")
            for c in s["commands"][:10]:
                parts.append(f"  {c['timestamp'][:19]} {c['command'][:60]}")

        if s.get("telemetry"):
            t = s["telemetry"][-1]
            parts.append(f"\nLatest telemetry: CPU={t['cpu_percent']:.1f}% "
                        f"Mem={t['memory_used_gb']:.1f}/{t['memory_total_gb']:.1f}GB "
                        f"Procs={t['process_count']}")

        parts.append("\nProvide a concise session summary.")
        return "\n".join(parts)

    def _structured_summary(self, session_data: dict) -> str:
        """Generate a summary without the LLM (fallback)."""
        s = session_data
        lines = ["Session summary (structured):"]
        lines.append(f"  Events: {s['event_count']}")
        lines.append(f"  Commits: {s['commit_count']}")
        lines.append(f"  Commands: {s['command_count']}")

        if s.get("files_touched"):
            lines.append(f"\nFiles touched ({len(s['files_touched'])}):")
            for f in s["files_touched"][:10]:
                lines.append(f"  {f}")

        if s.get("commits"):
            lines.append(f"\nCommits ({len(s['commits'])}):")
            for c in s["commits"][:5]:
                lines.append(f"  {c['hash'][:8]} {c['message'][:60]}")

        if s.get("commands"):
            lines.append(f"\nCommands ({len(s['commands'])}):")
            for c in s["commands"][:5]:
                lines.append(f"  {c['command'][:60]}")

        return "\n".join(lines)

    def _raw_summary(self, since: str) -> dict[str, Any]:
        """Build a summary from raw events when no session is detected."""
        events = self.db.query_events(since=since, limit=100)
        commits = self.db.query(
            "SELECT * FROM git_commits WHERE author_date >= ? ORDER BY author_date ASC",
            (since,),
        )
        commands = self.db.shell_commands_since(since, limit=50)
        telemetry = self.db.telemetry_since(since, limit=20)


        files_touched: set[str] = set()
        for e in events:
            if e["type"] == "FILE_INDEXED":
                try:
                    payload = json.loads(e["payload"])
                    if "path" in payload:
                        files_touched.add(payload["path"])
                except Exception:
                    continue

        return {
            "session": {"start_time": since, "end_time": datetime.now(UTC).isoformat()},
            "events": [dict(e) for e in events],
            "commits": [dict(c) for c in commits],
            "telemetry": [dict(t) for t in telemetry],
            "commands": [dict(c) for c in commands],
            "files_touched": sorted(files_touched),
            "event_count": len(events),
            "commit_count": len(commits),
            "telemetry_count": len(telemetry),
            "command_count": len(commands),
        }

    @staticmethod
    def matches(goal: str) -> bool:
        """Check if a goal looks like a session reconstruction query."""
        return bool(_SESSION_RE.search(goal) or _LAST_WORK_RE.search(goal))
