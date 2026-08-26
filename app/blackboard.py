"""Blackboard — shared working memory for the cognitive system.

Inspired by the classical AI blackboard architecture. Instead of each agent
independently rediscovering context, the blackboard maintains a shared state
that every capability can read from and write to.

The blackboard is NOT a database table. It's an in-memory structure that is
periodically refreshed by the attention daemon and can be persisted to
SQLite for recovery after restart.

Structure:
  - current_goal: what the user is trying to accomplish
  - current_workspace: which project is active
  - current_context: assembled Context dataclass
  - open_problems: list of identified issues
  - recent_decisions: last few routing/action decisions
  - current_plan: the active task graph or daily plan
  - suggested_actions: what the system thinks the user should do next
  - confidence: how confident the system is in its current understanding
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)


@dataclass
class Blackboard:
    """Shared working memory. Thread-safe via a lock."""

    current_goal: str = ""
    goal_source: str = ""
    goal_updated_at: str = ""
    current_workspace: str = ""
    current_context: dict[str, Any] = field(default_factory=dict)
    open_problems: list[dict[str, Any]] = field(default_factory=list)
    recent_decisions: list[dict[str, Any]] = field(default_factory=list)
    current_plan: list[str] = field(default_factory=list)
    suggested_actions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    last_updated: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_goal(self, goal: str, *, source: str = "manual") -> None:
        with self._lock:
            self.current_goal = goal
            self.goal_source = source
            self.goal_updated_at = datetime.now(UTC).isoformat()
            self._touch()

    def set_workspace(self, workspace: str) -> None:
        with self._lock:
            self.current_workspace = workspace
            self._touch()

    def add_problem(self, problem: str, *, evidence: str = "") -> None:
        with self._lock:
            self.open_problems.append({
                "problem": problem, "evidence": evidence,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            self._touch()

    def resolve_problem(self, index: int) -> None:
        with self._lock:
            if 0 <= index < len(self.open_problems):
                self.open_problems.pop(index)
                self._touch()

    def add_decision(self, agent: str, decision: str, confidence: float = 0.0) -> None:
        with self._lock:
            self.recent_decisions.append({
                "agent": agent, "decision": decision, "confidence": confidence,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            # Keep only last 10.
            self.recent_decisions = self.recent_decisions[-10:]
            self._touch()

    def set_plan(self, plan: list[str]) -> None:
        with self._lock:
            self.current_plan = plan
            self._touch()

    def add_suggestion(self, action: str, reason: str = "") -> None:
        with self._lock:
            self.suggested_actions.append({
                "action": action, "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            self.suggested_actions = self.suggested_actions[-5:]
            self._touch()

    def set_context(self, ctx: dict[str, Any]) -> None:
        with self._lock:
            self.current_context = ctx
            self._touch()

    def set_confidence(self, confidence: float) -> None:
        with self._lock:
            self.confidence = confidence
            self._touch()

    def update_time_awareness(self, time_info: dict[str, Any]) -> None:
        """Update the time awareness state on the blackboard.

        This is called periodically (e.g. by the attention daemon or
        before each context packet assembly) to keep the blackboard's
        time info fresh. The time info is stored in current_context
        under the "time_awareness" key.
        """
        with self._lock:
            self.current_context["time_awareness"] = time_info
            self._touch()

    def clear(self) -> None:
        with self._lock:
            self.current_goal = ""
            self.goal_source = ""
            self.goal_updated_at = ""
            self.current_workspace = ""
            self.current_context = {}
            self.open_problems = []
            self.recent_decisions = []
            self.current_plan = []
            self.suggested_actions = []
            self.confidence = 0.0
            self.last_updated = datetime.now(UTC).isoformat()

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe copy of the blackboard state."""
        with self._lock:
            return {
                "current_goal": self.current_goal,
                "goal_source": self.goal_source,
                "goal_updated_at": self.goal_updated_at,
                "current_workspace": self.current_workspace,
                "current_context": dict(self.current_context),
                "open_problems": list(self.open_problems),
                "recent_decisions": list(self.recent_decisions),
                "current_plan": list(self.current_plan),
                "suggested_actions": list(self.suggested_actions),
                "confidence": self.confidence,
                "last_updated": self.last_updated,
            }

    def to_text(self) -> str:
        """Render as a readable text block for prompts or display."""
        s = self.snapshot()
        lines: list[str] = []

        if s["current_goal"]:
            source = f" (source: {s['goal_source']})" if s["goal_source"] else ""
            lines.append(f"Goal: {s['current_goal']}{source}")
        if s["current_workspace"]:
            lines.append(f"Workspace: {s['current_workspace']}")
        lines.append(f"Confidence: {s['confidence']:.2f}")

        # Time awareness (if available in current_context).
        time_info = s.get("current_context", {}).get("time_awareness", {})
        if time_info and isinstance(time_info, dict):
            sess = time_info.get("session", {})
            if sess.get("elapsed"):
                elapsed = sess["elapsed"]
                calls = sess.get("tool_call_count", 0)
                active = "active" if sess.get("is_active") else "idle"
                lines.append(f"Time: {time_info.get('current_time_human', '?')} | Session: {elapsed} ({calls} calls, {active})")

        if s["open_problems"]:
            lines.append(f"\nOpen problems ({len(s['open_problems'])}):")
            for i, p in enumerate(s["open_problems"]):
                lines.append(f"  [{i}] {p['problem']}")
                if p["evidence"]:
                    lines.append(f"      evidence: {p['evidence']}")

        if s["current_plan"]:
            lines.append("\nCurrent plan:")
            for item in s["current_plan"]:
                lines.append(f"  - {item}")

        if s["suggested_actions"]:
            lines.append("\nSuggested actions:")
            for a in s["suggested_actions"]:
                lines.append(f"  - {a['action']}")
                if a["reason"]:
                    lines.append(f"      reason: {a['reason']}")

        if s["recent_decisions"]:
            lines.append("\nRecent decisions:")
            for d in s["recent_decisions"][-3:]:
                lines.append(f"  {d['agent']}: {d['decision']} (conf={d['confidence']:.2f})")

        return "\n".join(lines) if lines else "(blackboard empty)"

    def persist(self, db: Database) -> None:
        """Save blackboard state to the database for recovery."""
        state = self.snapshot()
        db.set_user_model_entry(
            "_blackboard_state", json.dumps(state),
            confidence=1.0, source="system",
        )

    def restore(self, db: Database) -> bool:
        """Restore blackboard state from the database. Returns True if restored."""
        entry = db.get_user_model_entry("_blackboard_state")
        if not entry:
            return False
        try:
            state = json.loads(entry["value"])
            with self._lock:
                self.current_goal = state.get("current_goal", "")
                self.goal_source = state.get("goal_source") or (
                    "legacy" if self.current_goal else ""
                )
                self.goal_updated_at = state.get(
                    "goal_updated_at", state.get("last_updated", "")
                )
                self.current_workspace = state.get("current_workspace", "")
                self.current_context = state.get("current_context", {})
                self.open_problems = state.get("open_problems", [])
                self.recent_decisions = state.get("recent_decisions", [])
                self.current_plan = state.get("current_plan", [])
                self.suggested_actions = state.get("suggested_actions", [])
                self.confidence = state.get("confidence", 0.0)
                self.last_updated = state.get("last_updated", "")
            log.info("blackboard.restored", goal=self.current_goal,
                     workspace=self.current_workspace)
            return True
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("blackboard.restore_failed", error=str(e))
            return False

    def _touch(self) -> None:
        self.last_updated = datetime.now(UTC).isoformat()
