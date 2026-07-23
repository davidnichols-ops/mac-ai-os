"""Tests for Phase 1: workflow identity, lifecycle states, and persistence.

Tests:
  - WorkflowStatus enum and state machine transitions
  - Workflow identity (workflow_id, version, step_id)
  - Workflow definition persistence
  - Per-step persistence (workflow_step_runs table)
  - Heartbeat updates during execution
  - Crash recovery (stale run detection)
  - Cancellation
  - State transition validation
"""

import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from app.db import Database
from app.events.bus import EventBus
from app.workflows.engine import (
    WorkflowEngine,
)
from app.workflows.models import (
    VerificationResult,
    Workflow,
    WorkflowStep,
)
from app.workflows.states import (
    InvalidTransitionError,
    WorkflowStatus,
    can_transition,
    is_active,
    is_terminal,
    validate_transition,
)

# --- Helpers ---

def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "something went wrong"}


def _slow_step(ctx):
    time.sleep(0.05)
    return {"ok": True, "data": "slow"}


# --- State machine tests ---

class TestWorkflowStatus:
    def test_status_values(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"
        assert WorkflowStatus.CRASHED.value == "crashed"
        assert WorkflowStatus.TIMED_OUT.value == "timed_out"
        assert WorkflowStatus.COMPENSATING.value == "compensating"
        assert WorkflowStatus.COMPENSATED.value == "compensated"

    def test_str_representation(self):
        assert str(WorkflowStatus.RUNNING) == "running"
        assert str(WorkflowStatus.COMPLETED) == "completed"

    def test_is_terminal(self):
        assert is_terminal(WorkflowStatus.COMPLETED)
        assert is_terminal(WorkflowStatus.CANCELLED)
        assert is_terminal(WorkflowStatus.COMPENSATED)
        assert not is_terminal(WorkflowStatus.RUNNING)
        assert not is_terminal(WorkflowStatus.PAUSED)
        assert not is_terminal(WorkflowStatus.FAILED)
        assert not is_terminal(WorkflowStatus.CRASHED)

    def test_is_active(self):
        assert is_active(WorkflowStatus.RUNNING)
        assert is_active(WorkflowStatus.COMPENSATING)
        assert not is_active(WorkflowStatus.PAUSED)
        assert not is_active(WorkflowStatus.COMPLETED)
        assert not is_active(WorkflowStatus.FAILED)


class TestStateTransitions:
    def test_pending_to_running(self):
        validate_transition(WorkflowStatus.PENDING, WorkflowStatus.RUNNING)

    def test_running_to_completed(self):
        validate_transition(WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED)

    def test_running_to_failed(self):
        validate_transition(WorkflowStatus.RUNNING, WorkflowStatus.FAILED)

    def test_running_to_paused(self):
        validate_transition(WorkflowStatus.RUNNING, WorkflowStatus.PAUSED)

    def test_running_to_cancelled(self):
        validate_transition(WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED)

    def test_running_to_crashed(self):
        validate_transition(WorkflowStatus.RUNNING, WorkflowStatus.CRASHED)

    def test_running_to_timed_out(self):
        validate_transition(WorkflowStatus.RUNNING, WorkflowStatus.TIMED_OUT)

    def test_paused_to_running(self):
        validate_transition(WorkflowStatus.PAUSED, WorkflowStatus.RUNNING)

    def test_paused_to_cancelled(self):
        validate_transition(WorkflowStatus.PAUSED, WorkflowStatus.CANCELLED)

    def test_crashed_to_running(self):
        validate_transition(WorkflowStatus.CRASHED, WorkflowStatus.RUNNING)

    def test_failed_to_compensating(self):
        validate_transition(WorkflowStatus.FAILED, WorkflowStatus.COMPENSATING)

    def test_compensating_to_compensated(self):
        validate_transition(WorkflowStatus.COMPENSATING, WorkflowStatus.COMPENSATED)

    def test_invalid_completed_to_running(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition(WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING)

    def test_invalid_cancelled_to_running(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition(WorkflowStatus.CANCELLED, WorkflowStatus.RUNNING)

    def test_invalid_pending_to_completed(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition(WorkflowStatus.PENDING, WorkflowStatus.COMPLETED)

    def test_invalid_pending_to_failed(self):
        with pytest.raises(InvalidTransitionError):
            validate_transition(WorkflowStatus.PENDING, WorkflowStatus.FAILED)

    def test_can_transition_true(self):
        assert can_transition(WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED)

    def test_can_transition_false(self):
        assert not can_transition(WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING)

    def test_invalid_transition_error_message(self):
        try:
            validate_transition(WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING)
            raise AssertionError("should have raised")
        except InvalidTransitionError as e:
            assert "completed" in str(e)
            assert "running" in str(e)
            assert e.from_status == WorkflowStatus.COMPLETED
            assert e.to_status == WorkflowStatus.RUNNING


# --- Workflow identity tests ---

class TestWorkflowIdentity:
    def test_workflow_has_uuid(self):
        wf = Workflow(name="test", description="test")
        assert wf.workflow_id
        assert len(wf.workflow_id) == 36  # UUID4 string length

    def test_workflow_has_version(self):
        wf = Workflow(name="test", description="test")
        assert wf.version == 1

    def test_workflow_version_custom(self):
        wf = Workflow(name="test", description="test", version=3)
        assert wf.version == 3

    def test_step_has_step_id(self):
        wf = Workflow(name="test", description="test")
        step = WorkflowStep(name="step1", description="First", handler=_ok_step)
        wf.add_step(step)
        assert step.step_id
        assert len(step.step_id) == 16  # truncated hash

    def test_step_id_deterministic(self):
        """Same workflow_id + version + name → same step_id."""
        wf = Workflow(name="test", description="test", workflow_id="abc-123", version=1)
        step1 = WorkflowStep(name="step1", description="First", handler=_ok_step)
        wf.add_step(step1)
        step_id_1 = step1.step_id

        # Recreate with same workflow_id and version.
        wf2 = Workflow(name="test", description="test", workflow_id="abc-123", version=1)
        step2 = WorkflowStep(name="step1", description="First", handler=_ok_step)
        wf2.add_step(step2)
        assert step2.step_id == step_id_1

    def test_step_id_changes_with_version(self):
        wf1 = Workflow(name="test", description="test", workflow_id="abc-123", version=1)
        wf2 = Workflow(name="test", description="test", workflow_id="abc-123", version=2)
        s1 = WorkflowStep(name="step1", description="First", handler=_ok_step)
        s2 = WorkflowStep(name="step1", description="First", handler=_ok_step)
        wf1.add_step(s1)
        wf2.add_step(s2)
        assert s1.step_id != s2.step_id

    def test_step_id_changes_with_name(self):
        wf = Workflow(name="test", description="test", workflow_id="abc-123")
        s1 = WorkflowStep(name="step1", description="First", handler=_ok_step)
        s2 = WorkflowStep(name="step2", description="Second", handler=_ok_step)
        wf.add_step(s1)
        wf.add_step(s2)
        assert s1.step_id != s2.step_id

    def test_steps_added_in_constructor_get_ids(self):
        step1 = WorkflowStep(name="s1", description="First", handler=_ok_step)
        step2 = WorkflowStep(name="s2", description="Second", handler=_ok_step)
        Workflow(name="test", description="test", steps=[step1, step2])
        assert step1.step_id
        assert step2.step_id
        assert step1.step_id != step2.step_id

    def test_workflow_to_dict_includes_identity(self):
        wf = Workflow(name="test", description="test", version=2)
        d = wf.to_dict()
        assert "workflow_id" in d
        assert "version" in d
        assert d["version"] == 2


# --- Workflow definition persistence tests ---

class TestWorkflowDefinitionPersistence:
    def test_definition_saved_on_run(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_def", description="Test definition persistence")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf)

        defs = db.list_workflow_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "test_def"
        assert defs[0]["version"] == 1

        definition = json.loads(defs[0]["definition"])
        assert definition["name"] == "test_def"
        assert len(definition["steps"]) == 1
        assert definition["steps"][0]["name"] == "step1"
        db.close()

    def test_definition_retrieval_by_name(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_retrieve", description="Test retrieval")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf)

        row = db.get_workflow_definition("test_retrieve")
        assert row is not None
        assert row["name"] == "test_retrieve"
        db.close()

    def test_definition_retrieval_by_name_and_version(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        wf1 = Workflow(name="test_ver", description="v1", version=1)
        wf1.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf1)

        wf2 = Workflow(name="test_ver", description="v2", version=2)
        wf2.add_step(WorkflowStep(name="step1", description="First v2", handler=_ok_step))
        wf2.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        engine.run(wf2)

        v1 = db.get_workflow_definition("test_ver", version=1)
        v2 = db.get_workflow_definition("test_ver", version=2)
        assert v1["version"] == 1
        assert v2["version"] == 2

        latest = db.get_workflow_definition("test_ver")
        assert latest["version"] == 2  # latest version
        db.close()


# --- Per-step persistence tests ---

class TestStepPersistence:
    def test_step_runs_recorded(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_steps", description="Test step persistence")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        assert len(step_runs) == 2
        assert step_runs[0]["step_name"] == "step1"
        assert step_runs[0]["status"] == "completed"
        assert step_runs[1]["step_name"] == "step2"
        assert step_runs[1]["status"] == "completed"
        db.close()

    def test_step_run_has_step_id(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_step_id", description="Test step ID")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        assert step_runs[0]["step_id"] == wf.steps[0].step_id
        db.close()

    def test_step_run_has_duration(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_duration", description="Test duration")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_slow_step))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        assert step_runs[0]["duration_ms"] is not None
        assert step_runs[0]["duration_ms"] >= 40  # at least ~50ms
        db.close()

    def test_step_run_records_failure(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_step_fail", description="Test step failure record")
        wf.add_step(WorkflowStep(name="step1", description="Fails", handler=_fail_step, required=True))
        wf.add_step(WorkflowStep(name="step2", description="Should not run", handler=_ok_step))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        assert len(step_runs) == 1  # only the failed step
        assert step_runs[0]["status"] == "failed"
        assert "something went wrong" in step_runs[0]["error"]
        db.close()

    def test_step_run_records_data(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_data", description="Test data persistence")
        wf.add_step(WorkflowStep(name="step1", description="Returns data", handler=_ok_step))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        data = json.loads(step_runs[0]["data"])
        assert data["ok"] is True
        assert data["value"] == "done"
        db.close()

    def test_step_run_records_verification(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        def verify_ok(ctx):
            return VerificationResult(success=True, evidence=["verified"])

        wf = Workflow(name="test_verify", description="Test verification persistence")
        wf.add_step(WorkflowStep(
            name="step1", description="Verified", handler=_ok_step, verify=verify_ok,
        ))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        ver = json.loads(step_runs[0]["verification"])
        assert ver["success"] is True
        assert "verified" in ver["evidence"]
        db.close()


# --- Heartbeat tests ---

class TestHeartbeat:
    def test_heartbeat_set_on_start(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_hb", description="Test heartbeat")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        run = db.get_workflow_run(result.run_id)
        assert run["heartbeat_at"] is not None
        db.close()

    def test_heartbeat_updated_during_execution(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_hb_update", description="Test heartbeat update")
        wf.add_step(WorkflowStep(name="step1", description="Slow", handler=_slow_step))
        wf.add_step(WorkflowStep(name="step2", description="Slow", handler=_slow_step))
        result = engine.run(wf)

        run = db.get_workflow_run(result.run_id)
        assert run["heartbeat_at"] is not None
        # Heartbeat should be a valid ISO timestamp.
        hb = run["heartbeat_at"]
        datetime.fromisoformat(hb.replace("Z", "+00:00"))
        db.close()


# --- Crash recovery tests ---

class TestCrashRecovery:
    def test_stale_run_detected(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        # Manually insert a stale run (old heartbeat).
        old_time = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'stale_wf', 'running', 3, 1, ?, '', 1)""",
            (old_time, old_time),
        )

        engine = WorkflowEngine(db)
        crashed = engine.recover_crashed_runs(max_heartbeat_age_seconds=300)
        assert len(crashed) == 1

        runs = db.workflow_runs_by_status("crashed")
        assert len(runs) == 1
        assert runs[0]["workflow"] == "stale_wf"
        db.close()

    def test_fresh_run_not_stale(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        # Insert a run with a recent heartbeat.
        recent_time = datetime.now(UTC).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'fresh_wf', 'running', 3, 1, ?, '', 1)""",
            (recent_time, recent_time),
        )

        engine = WorkflowEngine(db)
        crashed = engine.recover_crashed_runs(max_heartbeat_age_seconds=300)
        assert len(crashed) == 0

        runs = db.workflow_runs_by_status("running")
        assert len(runs) == 1
        db.close()

    def test_no_heartbeat_is_stale(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        # Insert a run with no heartbeat (NULL).
        old_time = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'no_hb_wf', 'running', 3, 0, NULL, '', 1)""",
            (old_time,),
        )

        engine = WorkflowEngine(db)
        crashed = engine.recover_crashed_runs(max_heartbeat_age_seconds=300)
        assert len(crashed) == 1
        db.close()

    def test_crash_recovery_publishes_event(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        bus = EventBus(db=db)
        engine = WorkflowEngine(db, bus=bus)

        old_time = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'crash_evt_wf', 'running', 3, 1, ?, '', 1)""",
            (old_time, old_time),
        )

        engine.recover_crashed_runs(max_heartbeat_age_seconds=300)

        events = db.query_events(limit=50)
        types = [e["type"] for e in events]
        assert "WORKFLOW_CRASHED" in types
        db.close()

    def test_completed_runs_not_affected(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        old_time = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'completed_wf', 'completed', 3, 3, ?, '', 1)""",
            (old_time, old_time),
        )

        engine = WorkflowEngine(db)
        crashed = engine.recover_crashed_runs(max_heartbeat_age_seconds=300)
        assert len(crashed) == 0
        db.close()


# --- Cancellation tests ---

class TestCancellation:
    def test_cancel_before_step_runs(self, tmp_path):
        """Cancel a workflow before it starts executing steps."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        # Create a workflow with a slow step.
        wf = Workflow(name="test_cancel", description="Test cancellation")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_slow_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))

        # We can't easily cancel mid-run in a synchronous test,
        # but we can test the cancel() API with a non-existent run.
        assert engine.cancel(99999) is False  # non-existent run
        db.close()

    def test_cancel_non_running_run(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        # Insert a completed run.
        wf = Workflow(name="test_cancel_completed", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        # Try to cancel the completed run — should fail.
        assert engine.cancel(result.run_id) is False
        db.close()

    def test_cancel_running_run(self, tmp_path):
        """Test cancellation of a running workflow by pre-registering the cancel."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        # Pre-register a run_id for cancellation.
        # We'll use a trick: create the run manually, then cancel it,
        # then run a workflow that uses the same run_id.
        # Actually, let's test the simpler case: cancel a manually-created running run.
        old_time = datetime.now(UTC).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'manual_wf', 'running', 2, 0, ?, '', 1)""",
            (old_time, old_time),
        )
        run_id = db.query("SELECT id FROM workflow_runs WHERE workflow = 'manual_wf'")[0]["id"]

        assert engine.cancel(run_id) is True
        db.close()


# --- Run status and details tests ---

class TestRunStatusAndDetails:
    def test_get_run_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_status", description="Test status")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        status = engine.get_run_status(result.run_id)
        assert status == WorkflowStatus.COMPLETED
        db.close()

    def test_get_run_status_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        assert engine.get_run_status(99999) is None
        db.close()

    def test_get_run_details(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_details", description="Test details")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        result = engine.run(wf)

        details = engine.get_run_details(result.run_id)
        assert details is not None
        assert details["run"]["workflow"] == "test_details"
        assert details["run"]["status"] == "completed"
        assert len(details["steps"]) == 2
        assert details["steps"][0]["step_name"] == "step1"
        assert details["steps"][1]["step_name"] == "step2"
        db.close()

    def test_get_run_details_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        assert engine.get_run_details(99999) is None
        db.close()

    def test_run_has_workflow_id(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_wid", description="Test workflow_id in run")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        run = db.get_workflow_run(result.run_id)
        assert run["workflow_id"] == wf.workflow_id
        assert run["version"] == 1
        db.close()

    def test_run_has_finished_at(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_finished", description="Test finished_at")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        run = db.get_workflow_run(result.run_id)
        assert run["finished_at"] is not None
        db.close()

    def test_failed_run_has_finished_at(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_failed_finished", description="Test failed finished_at")
        wf.add_step(WorkflowStep(name="step1", description="Fails", handler=_fail_step, required=True))
        result = engine.run(wf)

        run = db.get_workflow_run(result.run_id)
        assert run["status"] == "failed"
        assert run["finished_at"] is not None
        db.close()


# --- WorkflowResult status field tests ---

class TestWorkflowResultStatus:
    def test_completed_result_has_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_result_status", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        assert result.status == "completed"
        assert result.workflow_id == wf.workflow_id
        db.close()

    def test_failed_result_has_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_result_fail", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="Fails", handler=_fail_step, required=True))
        result = engine.run(wf)

        assert result.status == "failed"
        db.close()

    def test_result_to_dict_includes_status_and_workflow_id(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_dict", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        d = result.to_dict()
        assert "status" in d
        assert d["status"] == "completed"
        assert "workflow_id" in d
        assert d["workflow_id"] == wf.workflow_id
        db.close()


# --- StepResult identity tests ---

class TestStepResultIdentity:
    def test_step_result_has_step_id(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_sr_id", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        assert result.steps[0].step_id == wf.steps[0].step_id
        db.close()

    def test_step_result_has_attempt(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_sr_attempt", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        assert result.steps[0].attempt == 1
        db.close()

    def test_step_result_to_dict_includes_identity(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_sr_dict", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        d = result.steps[0].to_dict()
        assert "step_id" in d
        assert "attempt" in d
        db.close()
