"""Tests for Phase 3: checkpoints and resumability.

Tests:
  - Checkpoint saved after each successful step
  - Checkpoint context restoration
  - Resume from checkpoint after crash
  - Resume creates child run with parent_run_id
  - Side effect tracking and idempotency
  - Resume from no checkpoint (fresh start)
  - Resume from wrong state (should fail)
"""

import json

from app.db import Database
from app.events.bus import EventBus
from app.workflows.engine import WorkflowEngine
from app.workflows.models import (
    FailureBehavior,
    Workflow,
    WorkflowStep,
)

# --- Helpers ---

def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "permanent failure"}


def _step_that_sets_data(ctx):
    return {"ok": True, "my_key": "my_value", "step_num": ctx.get("step_num", 0) + 1}


def _step_that_reads_data(ctx):
    """Reads data set by a previous step — verifies context is restored on resume."""
    my_key = ctx.get("my_key", "MISSING")
    return {"ok": True, "received_key": my_key}


# --- Checkpoint tests ---

class TestCheckpoints:
    def test_checkpoint_saved_after_each_step(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_checkpoint", description="Test checkpointing")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step3", description="Third", handler=_ok_step))
        result = engine.run(wf)

        checkpoints = db.checkpoints_for_workflow(result.run_id)
        assert len(checkpoints) == 3  # one after each step
        assert checkpoints[0]["step_index"] == 0
        assert checkpoints[1]["step_index"] == 1
        assert checkpoints[2]["step_index"] == 2
        db.close()

    def test_checkpoint_not_saved_on_failure(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_no_checkpoint_on_fail", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="OK", handler=_ok_step))
        wf.add_step(WorkflowStep(
            name="step2", description="Fails", handler=_fail_step,
            on_failure=FailureBehavior.ABORT,
        ))
        wf.add_step(WorkflowStep(name="step3", description="Never runs", handler=_ok_step))
        result = engine.run(wf)

        checkpoints = db.checkpoints_for_workflow(result.run_id)
        assert len(checkpoints) == 1  # only after step1
        assert checkpoints[0]["step_index"] == 0
        db.close()

    def test_checkpoint_contains_context(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_ctx_checkpoint", description="Test context in checkpoint")
        wf.add_step(WorkflowStep(name="step1", description="Sets data", handler=_step_that_sets_data))
        wf.add_step(WorkflowStep(name="step2", description="OK", handler=_ok_step))
        result = engine.run(wf)

        checkpoints = db.checkpoints_for_workflow(result.run_id)
        ctx = json.loads(checkpoints[0]["context"])
        assert ctx["my_key"] == "my_value"
        db.close()

    def test_latest_checkpoint(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_latest_cp", description="Test latest checkpoint")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        result = engine.run(wf)

        latest = db.latest_checkpoint(result.run_id)
        assert latest is not None
        assert latest["step_index"] == 1  # last step
        db.close()

    def test_no_checkpoint_for_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        assert db.latest_checkpoint(99999) is None
        db.close()


# --- Resume tests ---

class TestResume:
    def test_resume_from_crashed_run(self, tmp_path):
        """Simulate a crash after step 1, then resume from checkpoint."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        # Step 1 succeeds, step 2 would crash (we simulate by marking the run as crashed).
        step1_executed = {"ran": False}
        step2_executed = {"ran": False}

        def step1(ctx):
            step1_executed["ran"] = True
            return {"ok": True, "data_from_step1": "important_value"}

        def step2(ctx):
            step2_executed["ran"] = True
            # Verify context was restored from checkpoint
            assert ctx.get("data_from_step1") == "important_value"
            return {"ok": True, "step2_done": True}

        wf = Workflow(name="test_resume", description="Test resume")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=step1))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=step2))

        # Run step 1 only by making step 2 crash.
        def crashing_step2(ctx):
            step2_executed["ran"] = True
            raise RuntimeError("simulated crash")

        wf_crash = Workflow(name="test_resume", description="Test resume")
        wf_crash.add_step(WorkflowStep(name="step1", description="First", handler=step1))
        wf_crash.add_step(WorkflowStep(
            name="step2", description="Crashes", handler=crashing_step2,
            on_failure=FailureBehavior.ABORT,
        ))

        # First run — crashes at step 2.
        result1 = engine.run(wf_crash)
        assert not result1.ok
        assert step1_executed["ran"] is True

        # Mark the run as crashed (simulating process death).
        db.transition_workflow_run(result1.run_id, "crashed", error="simulated crash")

        # Resume with the correct workflow (step 2 doesn't crash this time).
        step2_executed["ran"] = False
        result2 = engine.resume(wf, result1.run_id)

        assert result2.ok
        assert step2_executed["ran"] is True
        assert result2.run_id != result1.run_id  # new run created
        db.close()

    def test_resume_creates_child_run(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        wf = Workflow(name="test_child_run", description="Test child run")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))

        # Run and crash.
        result1 = engine.run(wf)
        db.transition_workflow_run(result1.run_id, "crashed")

        # Resume.
        result2 = engine.resume(wf, result1.run_id)
        assert result2.ok

        # Verify parent_run_id is set.
        run2 = db.get_workflow_run(result2.run_id)
        assert run2["parent_run_id"] == result1.run_id
        db.close()

    def test_resume_skips_completed_steps(self, tmp_path):
        """Resume should only execute steps after the checkpoint."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        step1_count = {"n": 0}
        step2_count = {"n": 0}
        step3_count = {"n": 0}

        def step1(ctx):
            step1_count["n"] += 1
            return {"ok": True, "step1_data": "hello"}

        def step2(ctx):
            step2_count["n"] += 1
            raise RuntimeError("simulated crash at step 2")

        def step3(ctx):
            step3_count["n"] += 1
            return {"ok": True}

        wf_crash = Workflow(name="test_skip_completed", description="Test skip completed")
        wf_crash.add_step(WorkflowStep(name="step1", description="First", handler=step1))
        wf_crash.add_step(WorkflowStep(
            name="step2", description="Crashes", handler=step2,
            on_failure=FailureBehavior.ABORT,
        ))
        wf_crash.add_step(WorkflowStep(name="step3", description="Third", handler=step3))

        # Run — crashes at step 2. Only step 1 completes, checkpoint at step_index=0.
        result1 = engine.run(wf_crash)
        assert not result1.ok
        assert step1_count["n"] == 1

        # Mark as crashed.
        db.transition_workflow_run(result1.run_id, "crashed")

        # Create a non-crashing version of the workflow.
        def step2_ok(ctx):
            step2_count["n"] += 1
            return {"ok": True, "step2_data": "world"}

        wf_ok = Workflow(name="test_skip_completed", description="Test skip completed")
        wf_ok.add_step(WorkflowStep(name="step1", description="First", handler=step1))
        wf_ok.add_step(WorkflowStep(name="step2", description="Second", handler=step2_ok))
        wf_ok.add_step(WorkflowStep(name="step3", description="Third", handler=step3))

        # Resume — checkpoint at step_index=0, so resume_from=1.
        # Should execute steps 2 and 3, NOT step 1.
        result2 = engine.resume(wf_ok, result1.run_id)
        assert result2.ok

        # Step 1 should NOT have been re-executed.
        assert step1_count["n"] == 1  # not re-executed
        # Step 2 was attempted once (crashed) + once (resume) = 2.
        assert step2_count["n"] == 2
        # Step 3 only ran during resume.
        assert step3_count["n"] == 1
        db.close()

    def test_resume_restores_context_from_checkpoint(self, tmp_path):
        """Context from completed steps should be available on resume."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        received_data = {"value": None}

        def step1(ctx):
            return {"ok": True, "important_data": "secret123"}

        def step2(ctx):
            # This step reads data from step 1.
            received_data["value"] = ctx.get("important_data", "MISSING")
            return {"ok": True}

        wf = Workflow(name="test_ctx_restore", description="Test context restore")
        wf.add_step(WorkflowStep(name="step1", description="Sets data", handler=step1))
        wf.add_step(WorkflowStep(name="step2", description="Reads data", handler=step2))

        # Run step 1, then crash before step 2.
        def crashing_step2(ctx):
            raise RuntimeError("crash")

        wf_crash = Workflow(name="test_ctx_restore", description="Test")
        wf_crash.add_step(WorkflowStep(name="step1", description="Sets data", handler=step1))
        wf_crash.add_step(WorkflowStep(
            name="step2", description="Crashes", handler=crashing_step2,
            on_failure=FailureBehavior.ABORT,
        ))

        result1 = engine.run(wf_crash)
        db.transition_workflow_run(result1.run_id, "crashed")

        # Resume with the non-crashing workflow.
        result2 = engine.resume(wf, result1.run_id)
        assert result2.ok
        assert received_data["value"] == "secret123"  # context was restored
        db.close()

    def test_resume_from_nonexistent_run(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_no_run", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))

        result = engine.resume(wf, 99999)
        assert not result.ok
        assert "not found" in result.error
        db.close()

    def test_resume_from_running_run_fails(self, tmp_path):
        """Cannot resume a run that's still running."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_running", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        # Run is completed, not crashed — should fail.
        result2 = engine.resume(wf, result.run_id)
        assert not result2.ok
        assert "cannot resume" in result2.error
        db.close()

    def test_resume_from_no_checkpoint_starts_fresh(self, tmp_path):
        """If no checkpoint exists, resume should start from the beginning."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        # Create a crashed run with no checkpoints (step 1 failed immediately).
        wf = Workflow(name="test_no_cp", description="Test")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails", handler=_fail_step,
            on_failure=FailureBehavior.ABORT,
        ))
        result1 = engine.run(wf)
        assert not result1.ok

        # No checkpoint was saved (step 1 failed).
        assert db.latest_checkpoint(result1.run_id) is None

        # Mark as crashed.
        db.transition_workflow_run(result1.run_id, "crashed")

        # Resume with a workflow that succeeds.
        wf2 = Workflow(name="test_no_cp", description="Test")
        wf2.add_step(WorkflowStep(name="ok", description="OK", handler=_ok_step))
        result2 = engine.resume(wf2, result1.run_id)
        assert result2.ok
        db.close()

    def test_resume_publishes_resumed_event(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        bus = EventBus(db=db)
        engine = WorkflowEngine(db, bus=bus)

        wf = Workflow(name="test_resume_event", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))

        result1 = engine.run(wf)
        db.transition_workflow_run(result1.run_id, "crashed")

        engine.resume(wf, result1.run_id)

        events = db.query_events(limit=100)
        types = [e["type"] for e in events]
        assert "WORKFLOW_RESUMED" in types
        db.close()


# --- Side effect tracking tests ---

class TestSideEffects:
    def test_record_side_effect(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_se", description="Test side effects")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        recorded = engine.record_side_effect(
            result.run_id, wf.steps[0], "email.send",
            "email:session-summary:2026-08-18",
            details={"to": "david@example.com"},
        )
        assert recorded is True

        effects = db.side_effects_for_workflow(result.run_id)
        assert len(effects) == 1
        assert effects[0]["effect_type"] == "email.send"
        db.close()

    def test_duplicate_side_effect_rejected(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_dup_se", description="Test duplicate")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        key = "email:session-summary:2026-08-18"
        first = engine.record_side_effect(
            result.run_id, wf.steps[0], "email.send", key,
        )
        second = engine.record_side_effect(
            result.run_id, wf.steps[0], "email.send", key,
        )

        assert first is True
        assert second is False  # duplicate

        effects = db.side_effects_for_workflow(result.run_id)
        assert len(effects) == 1  # only one recorded
        db.close()

    def test_side_effect_idempotency_across_runs(self, tmp_path):
        """Side effect with same key should not be re-executed on resume."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)

        wf = Workflow(name="test_idem", description="Test idempotency")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result1 = engine.run(wf)

        # Record a side effect in the first run.
        key = "email:daily-report:2026-08-18"
        engine.record_side_effect(result1.run_id, wf.steps[0], "email.send", key)

        # Simulate crash and resume.
        db.transition_workflow_run(result1.run_id, "crashed")
        result2 = engine.resume(wf, result1.run_id)

        # Try to record the same side effect in the resumed run.
        # The idempotency key is global, so it should be rejected.
        second = engine.record_side_effect(
            result2.run_id, wf.steps[0], "email.send", key,
        )
        assert second is False  # already executed in the first run
        db.close()

    def test_side_effect_exists_check(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_exists", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        key = "task:create:important-task"
        assert db.side_effect_exists(key) is False

        engine.record_side_effect(result.run_id, wf.steps[0], "task.create", key)
        assert db.side_effect_exists(key) is True
        db.close()
