"""Attention — proactive context computation.

The system doesn't just react to requests. Every N seconds, the attention
daemon computes the current context and updates the blackboard. This means
agents inherit context rather than rediscovering it from scratch.

The attention cycle:
1. Detect active workspace (from recent file events, git activity, shell commands)
2. Detect current goal (from recent tasks, session history, blackboard state)
3. Check system health (from telemetry)
4. Identify open problems (from failed actions, high resource usage, blocked tasks)
5. Generate suggestions (from pending tasks, uncommitted changes, stale sessions)
6. Update the blackboard with all of this
7. Compute confidence based on how much context was successfully gathered

This is what makes the system feel like it's "paying attention" rather than
just responding to prompts.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from app.blackboard import Blackboard
from app.db import Database
from app.log import get_logger

log = get_logger(__name__)


class AttentionDaemon:
    """Periodically computes context and updates the blackboard."""

    def __init__(
        self,
        db: Database,
        blackboard: Blackboard,
        interval_seconds: float = 60.0,
    ):
        self.db = db
        self.blackboard = blackboard
        self.interval = interval_seconds
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_cycle: str = ""

    def start(self) -> None:
        """Start the attention daemon in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="attention")
        self._thread.start()
        log.info("attention.started", interval=self.interval)

    def stop(self) -> None:
        """Stop the attention daemon."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("attention.stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.cycle()
            except Exception as e:
                log.warning("attention.cycle.error", error=str(e))
            self._stop.wait(self.interval)

    def cycle(self) -> dict[str, Any]:
        """Run one attention cycle. Returns the computed context."""
        start = time.time()
        context: dict[str, Any] = {}
        confidence_parts: list[float] = []

        # 1. Detect active workspace.
        workspace = self._detect_workspace()
        if workspace:
            self.blackboard.set_workspace(workspace)
            context["workspace"] = workspace
            confidence_parts.append(0.2)
        else:
            confidence_parts.append(0.0)

        # 2. Detect current goal from pending tasks.
        goal = self._detect_goal()
        if goal:
            self.blackboard.set_goal(goal)
            context["goal"] = goal
            confidence_parts.append(0.2)
        else:
            confidence_parts.append(0.0)

        # 3. Check system health.
        health = self._check_health()
        context["health"] = health
        confidence_parts.append(0.15)

        # 4. Identify open problems.
        problems = self._identify_problems(health)
        # Replace problems on the blackboard.
        self.blackboard.open_problems = problems
        context["open_problems"] = len(problems)
        confidence_parts.append(0.15)

        # 5. Generate suggestions.
        suggestions = self._generate_suggestions(workspace, health)
        self.blackboard.suggested_actions = suggestions
        context["suggestions"] = len(suggestions)
        confidence_parts.append(0.15)

        # 6. Set context on blackboard.
        self.blackboard.set_context(context)

        # 7. Compute confidence.
        confidence = sum(confidence_parts)
        self.blackboard.set_confidence(confidence)

        # Persist to database.
        self.blackboard.persist(self.db)

        duration_ms = int((time.time() - start) * 1000)
        self._last_cycle = datetime.now(UTC).isoformat()
        log.info("attention.cycle", duration_ms=duration_ms,
                 workspace=workspace, goal=goal, confidence=confidence,
                 problems=len(problems), suggestions=len(suggestions))
        return context

    def _detect_workspace(self) -> str:
        """Detect the active workspace from recent activity."""
        from app.workspace import detect_active_workspace
        ws = detect_active_workspace(self.db)
        if ws:
            if not isinstance(ws, dict):
                ws = dict(ws)
            return ws.get("name") or ws.get("path", "")
        return ""

    def _detect_goal(self) -> str:
        """Detect the current goal from pending tasks."""
        tasks = self.db.pending_tasks()
        if tasks:
            # Highest priority pending task is the likely current goal.
            top = tasks[0]
            return top["title"]
        return ""

    def _check_health(self) -> dict[str, Any]:
        """Check system health from recent telemetry."""
        since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        telemetry = self.db.telemetry_since(since, limit=1)
        if not telemetry:
            return {"status": "no_data", "cpu": None, "memory": None}

        t = telemetry[-1]
        mem_pct = (t["memory_used_gb"] / t["memory_total_gb"] * 100) if t["memory_total_gb"] else 0
        status = "normal"
        if t["cpu_percent"] > 80:
            status = "high_cpu"
        if mem_pct > 85:
            status = "memory_pressure"

        return {
            "status": status,
            "cpu": round(t["cpu_percent"], 1),
            "memory_used": round(t["memory_used_gb"], 1),
            "memory_total": round(t["memory_total_gb"], 1),
            "memory_pct": round(mem_pct, 1),
            "process_count": t["process_count"],
        }

    def _identify_problems(self, health: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify open problems from system state."""
        problems: list[dict[str, Any]] = []
        now = datetime.now(UTC).isoformat()
        since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()

        # Health problems.
        if health["status"] == "memory_pressure":
            problems.append({
                "problem": "Memory pressure detected",
                "evidence": f"{health.get('memory_pct', 0)}% used",
                "timestamp": now,
            })
        if health["status"] == "high_cpu":
            problems.append({
                "problem": "High CPU usage",
                "evidence": f"{health.get('cpu', 0)}%",
                "timestamp": now,
            })

        # Blocked tasks.
        tasks = self.db.pending_tasks()
        for t in tasks[:10]:
            deps = self.db.task_dependencies(t["id"])
            if deps and self.db.task_is_blocked(t["id"]):
                problems.append({
                    "problem": f"Task '{t['title']}' is blocked by incomplete dependencies",
                    "evidence": f"depends on: {deps}",
                    "timestamp": now,
                })

        # Failed actions in the last hour.
        recent = self.db.recent_actions(limit=10)
        for a in recent:
            if not a["executed"] and a["timestamp"] > since:
                problems.append({
                    "problem": f"Action '{a['action_name']}' was denied",
                    "evidence": a.get("error", ""),
                    "timestamp": a["timestamp"],
                })

        return problems

    def _generate_suggestions(
        self, workspace: str, health: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate suggested actions based on current state."""
        suggestions: list[dict[str, Any]] = []
        now = datetime.now(UTC).isoformat()

        # Suggest continuing pending tasks.
        tasks = self.db.pending_tasks()
        if tasks:
            suggestions.append({
                "action": f"Continue: {tasks[0]['title']}",
                "reason": f"{len(tasks)} pending task(s)",
                "timestamp": now,
            })

        # Suggest reviewing uncommitted changes.
        if workspace:
            snapshot = self.db.latest_snapshot(workspace=workspace)
            if snapshot and snapshot.get("modified_files"):
                modified = snapshot["modified_files"]
                count = len(modified.split("\n")) if modified else 0
                if count > 0:
                    suggestions.append({
                        "action": f"Review {count} modified files in {workspace}",
                        "reason": "uncommitted changes detected",
                        "timestamp": now,
                    })

        # Suggest running tests if health is normal.
        if health["status"] == "normal":
            suggestions.append({
                "action": "Run test suite",
                "reason": "system health is normal, good time for tests",
                "timestamp": now,
            })

        # Suggest capturing a snapshot if none exists.
        if workspace:
            snapshot = self.db.latest_snapshot(workspace=workspace)
            if not snapshot:
                suggestions.append({
                    "action": f"Capture snapshot of {workspace}",
                    "reason": "no snapshot exists yet",
                    "timestamp": now,
                })

        return suggestions[:5]  # keep top 5

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_cycle(self) -> str:
        return self._last_cycle
