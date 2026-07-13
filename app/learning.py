"""Learning — record action outcomes and update the user model.

The final stage of the causal loop:

  Observe → Remember → Understand → Plan → Act → Verify → Learn

Learning here is not RL or self-modifying code. It's simple:
  - Action succeeded → store why, increase confidence in the pattern
  - Action failed → remember the failure, avoid next time

The user_model table is the seed. This module formalizes the feedback loop.
"""

from __future__ import annotations

from typing import Any

from app.db import Database
from app.log import get_logger

log = get_logger(__name__)


class Learner:
    """Records outcomes and updates procedural memory."""

    def __init__(self, db: Database):
        self.db = db

    def record_action_outcome(
        self,
        action_name: str,
        target: str,
        success: bool,
        *,
        error: str = "",
        duration_ms: int = 0,
    ) -> None:
        """Record the outcome of an action and update learned patterns."""
        key = f"action.{action_name}.success_rate"
        entry = self.db.get_user_model_entry(key)

        if entry:
            # Update running stats.
            parts = entry["value"].split("/")
            successes = int(parts[0]) if len(parts) >= 1 else 0
            total = int(parts[1]) if len(parts) >= 2 else 0
        else:
            successes = 0
            total = 0

        total += 1
        if success:
            successes += 1

        rate = successes / total if total > 0 else 0.0
        value = f"{successes}/{total}"
        confidence = min(0.99, 0.3 + (total * 0.05))  # confidence grows with data

        self.db.set_user_model_entry(key, value, confidence=confidence, source="learned")
        log.info("learn.action_outcome", action=action_name, success=success,
                 rate=round(rate, 2), total=total)

        # Record failure patterns.
        if not success and error:
            failure_key = f"action.{action_name}.common_errors"
            failure_entry = self.db.get_user_model_entry(failure_key)
            if failure_entry:
                errors = failure_entry["value"].split(" | ")
                if error not in errors:
                    errors.append(error)
                    errors = errors[-5:]  # keep last 5
                self.db.set_user_model_entry(
                    failure_key, " | ".join(errors), confidence=confidence, source="learned",
                )
            else:
                self.db.set_user_model_entry(
                    failure_key, error, confidence=0.5, source="learned",
                )

    def record_user_preference(
        self,
        key: str,
        value: str,
        *,
        confidence: float = 0.7,
        source: str = "observed",
    ) -> None:
        """Record an observed user preference."""
        existing = self.db.get_user_model_entry(key)
        if existing:
            # Increase confidence if the observation matches.
            if existing["value"] == value:
                new_confidence = min(0.99, existing["confidence"] + 0.1)
                self.db.set_user_model_entry(key, value, confidence=new_confidence, source=source)
                log.info("learn.preference.confirmed", key=key, confidence=new_confidence)
            else:
                # Different value — replace with lower confidence.
                self.db.set_user_model_entry(key, value, confidence=confidence, source=source)
                log.info("learn.preference.changed", key=key, value=value)
        else:
            self.db.set_user_model_entry(key, value, confidence=confidence, source=source)
            log.info("learn.preference.recorded", key=key, value=value)

    def record_workflow_outcome(
        self,
        workflow_name: str,
        success: bool,
        steps_completed: int,
        steps_total: int,
    ) -> None:
        """Record the outcome of a workflow run."""
        key = f"workflow.{workflow_name}.success_rate"
        entry = self.db.get_user_model_entry(key)

        if entry:
            parts = entry["value"].split("/")
            successes = int(parts[0]) if len(parts) >= 1 else 0
            total = int(parts[1]) if len(parts) >= 2 else 0
        else:
            successes = 0
            total = 0

        total += 1
        if success:
            successes += 1

        value = f"{successes}/{total}"
        confidence = min(0.99, 0.3 + (total * 0.05))

        self.db.set_user_model_entry(key, value, confidence=confidence, source="learned")
        log.info("learn.workflow_outcome", workflow=workflow_name,
                 success=success, steps=f"{steps_completed}/{steps_total}",
                 total_runs=total)

    def action_success_rate(self, action_name: str) -> dict[str, Any] | None:
        """Get the learned success rate for an action."""
        entry = self.db.get_user_model_entry(f"action.{action_name}.success_rate")
        if not entry:
            return None
        parts = entry["value"].split("/")
        successes = int(parts[0]) if len(parts) >= 1 else 0
        total = int(parts[1]) if len(parts) >= 2 else 0
        rate = successes / total if total > 0 else 0.0
        return {
            "successes": successes,
            "total": total,
            "rate": rate,
            "confidence": entry["confidence"],
        }

    def common_errors(self, action_name: str) -> list[str]:
        """Get common errors for an action."""
        entry = self.db.get_user_model_entry(f"action.{action_name}.common_errors")
        if not entry:
            return []
        return entry["value"].split(" | ")

    def learned_summary(self) -> dict[str, Any]:
        """Summarize everything the system has learned."""
        entries = self.db.all_user_model_entries()
        actions: list[dict[str, Any]] = []
        workflows: list[dict[str, Any]] = []
        preferences: list[dict[str, Any]] = []

        for e in entries:
            if e["key"].startswith("action.") and e["key"].endswith(".success_rate"):
                name = e["key"].split(".")[1]
                stats = self.action_success_rate(name)
                if stats:
                    actions.append({"action": name, **stats})
            elif e["key"].startswith("workflow.") and e["key"].endswith(".success_rate"):
                name = e["key"].split(".")[1]
                parts = e["value"].split("/")
                workflows.append({
                    "workflow": name,
                    "successes": int(parts[0]),
                    "total": int(parts[1]),
                    "confidence": e["confidence"],
                })
            elif not e["key"].startswith("_"):
                preferences.append({
                    "key": e["key"], "value": e["value"],
                    "confidence": e["confidence"], "source": e["source"],
                })

        return {
            "actions": actions,
            "workflows": workflows,
            "preferences": preferences,
            "total_entries": len(entries),
        }
