"""Cognitive loop — Observation → State → Cognition → Action integration.

Phase 5: This module provides the integration between the workflow engine
and MAOS's existing cognitive architecture:

  - Observation: EventBus, telemetry, session detection, workspace snapshots
  - State: Blackboard (shared state), context builder, user model
  - Cognition: LLM-based reasoning, intent routing, session agents
  - Action: Action proposals, authorization pipeline, action executor

The CognitiveLoop class provides reusable workflow steps that implement
each phase of the O→S→C→A cycle. These can be composed into workflows
that autonomously observe the system, reason about what's happening,
and take actions based on that reasoning.

Key components:
  - CognitiveLoop: Orchestrates the O→S→C→A cycle
  - observe_step: A workflow step that gathers observations
  - cognize_step: A workflow step that reasons about observations
  - act_step: A workflow step that proposes and executes actions
  - CognitiveContext: Enriched context for cognitive workflows
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.log import get_logger
from app.workflows.models import WorkflowStep

log = get_logger(__name__)


@dataclass
class Observation:
    """An observation gathered from the system.

    Observations are the input to the cognition phase. They represent
    what the system "sees" — events, telemetry, git state, etc.
    """

    source: str  # e.g., "events", "telemetry", "git", "session"
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""  # human-readable summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamp": self.timestamp,
            "data": self.data,
            "summary": self.summary,
        }


@dataclass
class CognitiveState:
    """The cognitive state of the system at a point in time.

    This is the "State" in the O→S→C→A loop. It aggregates observations
    and provides a structured representation that the cognition phase
    can reason about.
    """

    observations: list[Observation] = field(default_factory=list)
    workspace: str = ""
    session_id: str = ""
    active_tasks: list[dict[str, Any]] = field(default_factory=list)
    user_intent: str = ""  # inferred intent from cognition
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "workspace": self.workspace,
            "session_id": self.session_id,
            "active_tasks": self.active_tasks,
            "user_intent": self.user_intent,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class CognitiveResult:
    """The result of the cognition phase.

    Cognition takes a CognitiveState and produces a recommendation
    for what action(s) to take, if any.
    """

    reasoning: str = ""
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    should_act: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "recommended_actions": self.recommended_actions,
            "confidence": self.confidence,
            "should_act": self.should_act,
            "metadata": self.metadata,
        }


class CognitiveLoop:
    """Orchestrates the Observation → State → Cognition → Action cycle.

    This class provides reusable workflow step factories that implement
    each phase of the cognitive loop. Workflows can compose these steps
    to create autonomous behaviors.

    Usage:
        loop = CognitiveLoop(db=db, bus=bus, context_builder=ctx_builder)
        wf = Workflow(name="cognitive_cycle", description="O→S→C→A cycle")
        wf.add_step(loop.observe_step(sources=["events", "telemetry"]))
        wf.add_step(loop.cognize_step())
        wf.add_step(loop.act_step())
    """

    def __init__(
        self,
        *,
        db: Any | None = None,
        bus: Any | None = None,
        context_builder: Any | None = None,
        llm: Any | None = None,
        action_proposer: Any | None = None,
    ):
        self.db = db
        self.bus = bus
        self.context_builder = context_builder
        self.llm = llm
        self.action_proposer = action_proposer

    def observe_step(
        self, sources: list[str] | None = None,
        name: str = "observe",
    ) -> WorkflowStep:
        """Create a workflow step that gathers observations.

        Args:
            sources: List of observation sources to gather from.
                Supported: "events", "telemetry", "git", "session", "workspace".
            name: Step name.
        """
        sources = sources or ["events"]

        def handler(ctx: dict[str, Any]) -> dict[str, Any]:
            state = CognitiveState()
            t0 = time.time()

            for source in sources:
                obs = self._gather_observation(source, ctx)
                if obs:
                    state.add_observation(obs)

            # Extract workspace and session from context if available.
            state.workspace = ctx.get("workspace", "")
            state.session_id = ctx.get("session_id", "")

            duration_ms = int((time.time() - t0) * 1000)
            log.info(
                "cognitive.observe",
                sources=sources,
                observations=len(state.observations),
                duration_ms=duration_ms,
            )

            return {
                "ok": True,
                "cognitive_state": state.to_dict(),
                "observations_count": len(state.observations),
            }

        return WorkflowStep(
            name=name,
            description=f"Gather observations from: {', '.join(sources)}",
            handler=handler,
            side_effects=[],  # observation is read-only
        )

    def cognize_step(
        self, name: str = "cognize",
        prompt_template: str = "",
    ) -> WorkflowStep:
        """Create a workflow step that reasons about observations.

        Args:
            name: Step name.
            prompt_template: Optional custom prompt template for the LLM.
                If empty, a default template is used.
        """
        default_prompt = (
            "Based on the following observations, reason about what the user "
            "is likely doing and what actions (if any) should be taken.\n\n"
            "Observations:\n{observations}\n\n"
            "Provide:\n"
            "1. Your reasoning about the current situation\n"
            "2. Recommended actions (if any)\n"
            "3. Confidence level (0-1)\n"
        )

        def handler(ctx: dict[str, Any]) -> dict[str, Any]:
            state_dict = ctx.get("cognitive_state", {})
            observations = state_dict.get("observations", [])

            if not observations:
                return {
                    "ok": True,
                    "cognitive_result": CognitiveResult(
                        reasoning="No observations to reason about.",
                        should_act=False,
                    ).to_dict(),
                    "should_act": False,
                }

            # If LLM is available, use it for reasoning.
            reasoning = ""
            recommended_actions: list[dict[str, Any]] = []
            confidence = 0.0
            should_act = False

            if self.llm:
                prompt = (prompt_template or default_prompt).format(
                    observations=json.dumps(observations, indent=2),
                )
                try:
                    response = self.llm.generate(prompt)
                    reasoning = response
                    # Simple heuristic: if the response mentions "action" or "should",
                    # set should_act=True.
                    should_act = any(
                        kw in response.lower()
                        for kw in ["action", "should", "recommend", "suggest"]
                    )
                    confidence = 0.7 if should_act else 0.5
                except Exception as e:
                    log.warning("cognitive.llm_failed", error=str(e))
                    reasoning = f"LLM reasoning failed: {e}"
                    confidence = 0.0
            else:
                # No LLM — use simple heuristic reasoning.
                reasoning = self._heuristic_reasoning(observations)
                confidence = 0.5
                should_act = len(observations) > 0

            result = CognitiveResult(
                reasoning=reasoning,
                recommended_actions=recommended_actions,
                confidence=confidence,
                should_act=should_act,
            )

            log.info(
                "cognitive.cognize",
                observations=len(observations),
                should_act=should_act,
                confidence=confidence,
            )

            return {
                "ok": True,
                "cognitive_result": result.to_dict(),
                "should_act": should_act,
                "reasoning": reasoning,
            }

        return WorkflowStep(
            name=name,
            description="Reason about observations and decide on actions",
            handler=handler,
        )

    def act_step(
        self, name: str = "act",
        action_filter: Any | None = None,
    ) -> WorkflowStep:
        """Create a workflow step that executes recommended actions.

        Args:
            name: Step name.
            action_filter: Optional callable to filter which recommended
                actions to execute. Returns True for actions to execute.
        """

        def handler(ctx: dict[str, Any]) -> dict[str, Any]:
            result_dict = ctx.get("cognitive_result", {})
            should_act = result_dict.get("should_act", False)
            recommended = result_dict.get("recommended_actions", [])

            if not should_act or not recommended:
                return {
                    "ok": True,
                    "actions_executed": 0,
                    "actions_skipped": 0,
                    "message": "No actions to execute",
                }

            executed = 0
            skipped = 0
            errors: list[str] = []

            for action in recommended:
                if action_filter and not action_filter(action):
                    skipped += 1
                    continue

                # If we have an action_proposer (engine), use it.
                engine = ctx.get("engine")
                if engine and "action_name" in action:
                    from app.workflows.proposals import ActionProposal
                    proposal = ActionProposal(
                        action_name=action["action_name"],
                        target=action.get("target", ""),
                        parameters=action.get("parameters", {}),
                        reason=action.get("reason", "cognitive loop recommendation"),
                        risk_level=action.get("risk_level", "low"),
                        requires_approval=action.get("requires_approval", False),
                        idempotency_key=action.get("idempotency_key", ""),
                    )
                    outcome = engine.execute_action_proposal(
                        proposal, run_id=ctx.get("run_id"),
                    )
                    if outcome["ok"]:
                        executed += 1
                    else:
                        errors.append(
                            f"{action['action_name']}: {outcome.get('denied_reason', 'failed')}"
                        )
                else:
                    skipped += 1

            log.info(
                "cognitive.act",
                executed=executed,
                skipped=skipped,
                errors=len(errors),
            )

            return {
                "ok": len(errors) == 0,
                "actions_executed": executed,
                "actions_skipped": skipped,
                "errors": errors,
            }

        return WorkflowStep(
            name=name,
            description="Execute recommended actions from cognition phase",
            handler=handler,
        )

    def _gather_observation(
        self, source: str, ctx: dict[str, Any],
    ) -> Observation | None:
        """Gather an observation from a single source."""
        try:
            if source == "events" and self.db:
                events = self.db.query_events(limit=20)
                return Observation(
                    source="events",
                    data={"count": len(events), "recent": events[:5]},
                    summary=f"{len(events)} recent events",
                )
            elif source == "telemetry" and self.db:
                rows = self.db.query(
                    "SELECT * FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT 1"
                )
                if rows:
                    return Observation(
                        source="telemetry",
                        data=dict(rows[0]),
                        summary="Latest telemetry snapshot",
                    )
            elif source == "git" and self.db:
                commits = self.db.query(
                    "SELECT * FROM git_commits ORDER BY timestamp DESC LIMIT 5"
                )
                return Observation(
                    source="git",
                    data={"count": len(commits), "commits": commits},
                    summary=f"{len(commits)} recent commits",
                )
            elif source == "session" and self.db:
                sessions = self.db.query(
                    "SELECT * FROM sessions ORDER BY start_time DESC LIMIT 1"
                )
                if sessions:
                    return Observation(
                        source="session",
                        data=dict(sessions[0]),
                        summary="Latest session",
                    )
            elif source == "workspace" and self.db:
                workspaces = self.db.query(
                    "SELECT * FROM workspaces ORDER BY last_active DESC LIMIT 1"
                )
                if workspaces:
                    return Observation(
                        source="workspace",
                        data=dict(workspaces[0]),
                        summary="Active workspace",
                    )
        except Exception as e:
            log.warning("cognitive.observe_failed", source=source, error=str(e))

        return None

    def _heuristic_reasoning(self, observations: list[dict[str, Any]]) -> str:
        """Simple heuristic reasoning when no LLM is available."""
        sources = [o.get("source", "unknown") for o in observations]
        return (
            f"Observed {len(observations)} signals from: {', '.join(sources)}. "
            "No specific action recommended without LLM reasoning."
        )


def create_cognitive_workflow(
    loop: CognitiveLoop,
    *,
    name: str = "cognitive_cycle",
    sources: list[str] | None = None,
    description: str = "Observation → Cognition → Action cycle",
) -> Any:
    """Create a complete O→S→C→A workflow from a CognitiveLoop.

    Args:
        loop: The CognitiveLoop instance.
        name: Workflow name.
        sources: Observation sources.
        description: Workflow description.

    Returns:
        A Workflow with observe, cognize, and act steps.
    """
    from app.workflows.models import Workflow

    wf = Workflow(name=name, description=description)
    wf.add_step(loop.observe_step(sources=sources or ["events"]))
    wf.add_step(loop.cognize_step())
    wf.add_step(loop.act_step())
    return wf
