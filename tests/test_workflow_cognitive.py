"""Tests for Phase 5: Observation → State → Cognition → Action loop.

Tests:
  - Observation model
  - CognitiveState model
  - CognitiveResult model
  - CognitiveLoop.observe_step: gathers observations from DB
  - CognitiveLoop.cognize_step: reasons about observations (with/without LLM)
  - CognitiveLoop.act_step: executes recommended actions
  - create_cognitive_workflow: full O→S→C→A workflow
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.db import Database
from app.events.bus import EventBus
from app.workflows.cognitive import (
    CognitiveLoop,
    CognitiveResult,
    CognitiveState,
    Observation,
    create_cognitive_workflow,
)
from app.workflows.engine import WorkflowEngine

# --- Model tests ---

class TestObservation:
    def test_basic_observation(self):
        obs = Observation(source="events", summary="5 recent events")
        assert obs.source == "events"
        assert obs.summary == "5 recent events"
        assert obs.data == {}

    def test_observation_with_data(self):
        obs = Observation(
            source="telemetry",
            data={"cpu": 45.2, "memory": 60.0},
            summary="System telemetry",
        )
        assert obs.data["cpu"] == 45.2
        assert obs.data["memory"] == 60.0

    def test_to_dict(self):
        obs = Observation(
            source="git",
            timestamp="2026-08-18T10:00:00Z",
            data={"commits": 5},
            summary="5 recent commits",
        )
        d = obs.to_dict()
        assert d["source"] == "git"
        assert d["timestamp"] == "2026-08-18T10:00:00Z"
        assert d["data"]["commits"] == 5
        assert d["summary"] == "5 recent commits"


class TestCognitiveState:
    def test_empty_state(self):
        state = CognitiveState()
        assert state.observations == []
        assert state.workspace == ""
        assert state.user_intent == ""

    def test_add_observation(self):
        state = CognitiveState()
        obs = Observation(source="events", summary="test")
        state.add_observation(obs)
        assert len(state.observations) == 1
        assert state.observations[0].source == "events"

    def test_to_dict(self):
        state = CognitiveState(workspace="~/Projects/test")
        state.add_observation(Observation(source="events"))
        d = state.to_dict()
        assert d["workspace"] == "~/Projects/test"
        assert len(d["observations"]) == 1


class TestCognitiveResult:
    def test_empty_result(self):
        result = CognitiveResult()
        assert result.reasoning == ""
        assert result.recommended_actions == []
        assert result.should_act is False

    def test_result_with_actions(self):
        result = CognitiveResult(
            reasoning="User seems to be working on a bug fix",
            recommended_actions=[{"action_name": "create_task", "parameters": {}}],
            confidence=0.8,
            should_act=True,
        )
        assert len(result.recommended_actions) == 1
        assert result.confidence == 0.8

    def test_to_dict(self):
        result = CognitiveResult(reasoning="test", should_act=True, confidence=0.9)
        d = result.to_dict()
        assert d["reasoning"] == "test"
        assert d["should_act"] is True
        assert d["confidence"] == 0.9


# --- CognitiveLoop observe tests ---

class TestObserveStep:
    def test_observe_events(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        # Insert some events.
        from app.events.models import Event
        bus = EventBus(db=db)
        bus.publish(Event("TEST_EVENT", {"key": "value"}))
        bus.publish(Event("TEST_EVENT_2", {"key": "value2"}))

        loop = CognitiveLoop(db=db)
        step = loop.observe_step(sources=["events"])

        ctx = {"db": db}
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["observations_count"] > 0
        assert "cognitive_state" in result
        db.close()

    def test_observe_telemetry(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        # Insert a telemetry snapshot.
        db.execute(
            """INSERT INTO telemetry_snapshots
               (timestamp, cpu_percent, memory_percent, memory_used_gb)
               VALUES (?, 45.0, 60.0, 8.0)""",
            (datetime.now(UTC).isoformat(),),
        )

        loop = CognitiveLoop(db=db)
        step = loop.observe_step(sources=["telemetry"])

        ctx = {"db": db}
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["observations_count"] == 1
        state = result["cognitive_state"]
        assert state["observations"][0]["source"] == "telemetry"
        db.close()

    def test_observe_multiple_sources(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        from app.events.models import Event
        bus = EventBus(db=db)
        bus.publish(Event("TEST", {}))
        db.execute(
            """INSERT INTO telemetry_snapshots
               (timestamp, cpu_percent, memory_percent, memory_used_gb)
               VALUES (?, 45.0, 60.0, 8.0)""",
            (datetime.now(UTC).isoformat(),),
        )

        loop = CognitiveLoop(db=db)
        step = loop.observe_step(sources=["events", "telemetry"])

        ctx = {"db": db}
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["observations_count"] == 2
        sources = [o["source"] for o in result["cognitive_state"]["observations"]]
        assert "events" in sources
        assert "telemetry" in sources
        db.close()

    def test_observe_empty_source(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        loop = CognitiveLoop(db=db)
        step = loop.observe_step(sources=["git"])

        ctx = {"db": db}
        result = step.handler(ctx)

        assert result["ok"] is True
        # git_commits table may be empty — observation might be None.
        # The step should still succeed.
        db.close()

    def test_observe_no_db(self):
        loop = CognitiveLoop()  # no db
        step = loop.observe_step(sources=["events"])
        ctx = {}
        result = step.handler(ctx)
        assert result["ok"] is True
        assert result["observations_count"] == 0


# --- CognitiveLoop cognize tests ---

class TestCognizeStep:
    def test_cognize_no_observations(self):
        loop = CognitiveLoop()
        step = loop.cognize_step()

        ctx = {"cognitive_state": {"observations": []}}
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["should_act"] is False

    def test_cognize_with_observations_no_llm(self):
        loop = CognitiveLoop()  # no LLM
        step = loop.cognize_step()

        ctx = {
            "cognitive_state": {
                "observations": [
                    {"source": "events", "summary": "5 events"},
                    {"source": "telemetry", "summary": "CPU 45%"},
                ],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["should_act"] is True
        assert "Observed 2 signals" in result["reasoning"]

    def test_cognize_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="I recommend creating a task.")
        loop = CognitiveLoop(llm=mock_llm)
        step = loop.cognize_step()

        ctx = {
            "cognitive_state": {
                "observations": [{"source": "events", "summary": "test"}],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["should_act"] is True  # "recommend" in response
        assert "task" in result["reasoning"]
        mock_llm.generate.assert_called_once()

    def test_cognize_llm_failure(self):
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(side_effect=RuntimeError("LLM unavailable"))
        loop = CognitiveLoop(llm=mock_llm)
        step = loop.cognize_step()

        ctx = {
            "cognitive_state": {
                "observations": [{"source": "events"}],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert "LLM reasoning failed" in result["reasoning"]
        assert result["should_act"] is False

    def test_cognize_llm_no_action_keywords(self):
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Everything looks normal.")
        loop = CognitiveLoop(llm=mock_llm)
        step = loop.cognize_step()

        ctx = {
            "cognitive_state": {
                "observations": [{"source": "events"}],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["should_act"] is False  # no action keywords


# --- CognitiveLoop act tests ---

class TestActStep:
    def test_act_no_actions(self):
        loop = CognitiveLoop()
        step = loop.act_step()

        ctx = {
            "cognitive_result": {
                "should_act": False,
                "recommended_actions": [],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["actions_executed"] == 0
        assert "No actions to execute" in result["message"]

    def test_act_should_act_but_no_recommendations(self):
        loop = CognitiveLoop()
        step = loop.act_step()

        ctx = {
            "cognitive_result": {
                "should_act": True,
                "recommended_actions": [],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["actions_executed"] == 0

    def test_act_with_recommendations_no_engine(self):
        loop = CognitiveLoop()
        step = loop.act_step()

        ctx = {
            "cognitive_result": {
                "should_act": True,
                "recommended_actions": [
                    {"action_name": "create_task", "parameters": {"title": "test"}},
                ],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["actions_executed"] == 0
        assert result["actions_skipped"] == 1  # no engine to execute

    def test_act_with_filter(self):
        loop = CognitiveLoop()
        # Filter that rejects all actions.
        step = loop.act_step(action_filter=lambda a: False)

        ctx = {
            "cognitive_result": {
                "should_act": True,
                "recommended_actions": [
                    {"action_name": "create_task"},
                ],
            },
        }
        result = step.handler(ctx)

        assert result["ok"] is True
        assert result["actions_executed"] == 0
        assert result["actions_skipped"] == 1


# --- Full cognitive workflow tests ---

class TestCognitiveWorkflow:
    def test_create_cognitive_workflow(self):
        loop = CognitiveLoop()
        wf = create_cognitive_workflow(loop, sources=["events"])

        assert wf.name == "cognitive_cycle"
        assert len(wf.steps) == 3
        assert wf.steps[0].name == "observe"
        assert wf.steps[1].name == "cognize"
        assert wf.steps[2].name == "act"

    def test_run_cognitive_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        from app.events.models import Event
        bus = EventBus(db=db)
        bus.publish(Event("TEST", {"key": "value"}))

        loop = CognitiveLoop(db=db)
        wf = create_cognitive_workflow(loop, sources=["events"])
        engine = WorkflowEngine(db)

        result = engine.run(wf)
        assert result.ok
        assert len(result.steps) == 3
        assert result.steps[0].step_name == "observe"
        assert result.steps[1].step_name == "cognize"
        assert result.steps[2].step_name == "act"
        db.close()

    def test_cognitive_workflow_with_custom_name(self):
        loop = CognitiveLoop()
        wf = create_cognitive_workflow(loop, name="my_cycle", sources=["telemetry"])
        assert wf.name == "my_cycle"
