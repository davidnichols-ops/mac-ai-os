"""Tests for Phase 7: durable workflow external API.

Tests:
  - maos_workflow_list_definitions
  - maos_workflow_list_runs
  - maos_workflow_show
  - maos_workflow_cancel
  - maos_workflow_resume (informational)
  - maos_workflow_status
  - maos_workflow_recover_crashed
  - maos_workflow_dlq_list
  - maos_workflow_dlq_inspect
  - maos_workflow_dlq_mark_inspected
"""

from unittest.mock import MagicMock

from app.db import Database
from app.workflows.api import (
    maos_workflow_cancel,
    maos_workflow_dlq_inspect,
    maos_workflow_dlq_list,
    maos_workflow_dlq_mark_inspected,
    maos_workflow_list_definitions,
    maos_workflow_list_runs,
    maos_workflow_recover_crashed,
    maos_workflow_resume,
    maos_workflow_show,
    maos_workflow_status,
)
from app.workflows.engine import WorkflowEngine
from app.workflows.failure import DeadLetterQueue
from app.workflows.models import Workflow, WorkflowStep

# --- Helpers ---

def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "permanent failure"}


def _make_runtime(db: Database) -> MagicMock:
    """Create a mock runtime with a db."""
    rt = MagicMock()
    rt.db = db
    return rt


# --- List definitions tests ---

class TestListDefinitions:
    def test_no_definitions(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_list_definitions(runtime)
        assert "No workflow definitions" in result
        db.close()

    def test_with_definitions(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_def", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf)

        runtime = _make_runtime(db)
        result = maos_workflow_list_definitions(runtime)
        assert "test_def" in result
        assert "v1" in result
        db.close()


# --- List runs tests ---

class TestListRuns:
    def test_no_runs(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_list_runs(runtime)
        assert "No workflow runs" in result
        db.close()

    def test_with_runs(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_run", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf)

        runtime = _make_runtime(db)
        result = maos_workflow_list_runs(runtime)
        assert "test_run" in result
        assert "completed" in result
        db.close()

    def test_filter_by_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="ok_wf", description="OK")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf)

        wf_fail = Workflow(name="fail_wf", description="Fail")
        wf_fail.add_step(WorkflowStep(
            name="fails", description="Fails", handler=_fail_step,
            on_failure=__import__("app.workflows.models", fromlist=["FailureBehavior"]).FailureBehavior.ABORT,
        ))
        engine.run(wf_fail)

        runtime = _make_runtime(db)
        completed = maos_workflow_list_runs(runtime, status="completed")
        failed = maos_workflow_list_runs(runtime, status="failed")
        assert "ok_wf" in completed
        assert "fail_wf" not in completed
        assert "fail_wf" in failed
        db.close()


# --- Show workflow tests ---

class TestShowWorkflow:
    def test_show_existing_run(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_show", description="Test show")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        result = engine.run(wf)

        runtime = _make_runtime(db)
        output = maos_workflow_show(runtime, result.run_id)
        assert "test_show" in output
        assert "completed" in output
        assert "step1" in output
        assert "step2" in output
        db.close()

    def test_show_nonexistent_run(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        output = maos_workflow_show(runtime, 99999)
        assert "not found" in output
        db.close()

    def test_show_includes_checkpoints(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_cp", description="Test checkpoints")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        wf.add_step(WorkflowStep(name="step2", description="Second", handler=_ok_step))
        result = engine.run(wf)

        runtime = _make_runtime(db)
        output = maos_workflow_show(runtime, result.run_id)
        assert "Checkpoints" in output
        db.close()


# --- Cancel tests ---

class TestCancel:
    def test_cancel_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_cancel(runtime, 99999)
        assert "Cannot cancel" in result
        db.close()

    def test_cancel_completed_fails(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        runtime = _make_runtime(db)
        cancel_result = maos_workflow_cancel(runtime, result.run_id)
        assert "Cannot cancel" in cancel_result
        db.close()


# --- Resume tests ---

class TestResume:
    def test_resume_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_resume(runtime, 99999)
        assert "not found" in result
        db.close()

    def test_resume_returns_info(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_resume", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        result = engine.run(wf)

        runtime = _make_runtime(db)
        resume_result = maos_workflow_resume(runtime, result.run_id)
        assert "test_resume" in resume_result
        assert "CLI" in resume_result
        db.close()


# --- Status tests ---

class TestStatus:
    def test_empty_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_status(runtime)
        assert "Workflow Engine Status" in result
        assert "Running: 0" in result
        db.close()

    def test_status_with_runs(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
        engine.run(wf)

        runtime = _make_runtime(db)
        result = maos_workflow_status(runtime)
        assert "Running: 0" in result  # completed, not running
        db.close()


# --- Recover crashed tests ---

class TestRecoverCrashed:
    def test_no_crashed(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_recover_crashed(runtime)
        assert "No stale" in result
        db.close()

    def test_recover_stale(self, tmp_path):
        from datetime import UTC, datetime, timedelta
        db = Database(str(tmp_path / "test.db"))
        old_time = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        db.execute(
            """INSERT INTO workflow_runs
               (timestamp, workflow, status, steps_total, steps_done, heartbeat_at, workflow_id, version)
               VALUES (?, 'stale_wf', 'running', 3, 1, ?, '', 1)""",
            (old_time, old_time),
        )

        runtime = _make_runtime(db)
        result = maos_workflow_recover_crashed(runtime)
        assert "Recovered 1" in result
        db.close()


# --- DLQ tests ---

class TestDLQAPI:
    def test_dlq_list_empty(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_dlq_list(runtime)
        assert "No pending entries" in result
        db.close()

    def test_dlq_list_with_entries(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        run_id = db.start_workflow_run("failed_wf", 3)
        dlq = DeadLetterQueue(db)
        dlq.enqueue(run_id, "failed_wf", "step2 failed permanently")

        runtime = _make_runtime(db)
        result = maos_workflow_dlq_list(runtime)
        assert "failed_wf" in result
        assert "step2 failed" in result
        db.close()

    def test_dlq_inspect(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        run_id = db.start_workflow_run("failed_wf", 3)
        dlq = DeadLetterQueue(db)
        dlq_id = dlq.enqueue(run_id, "failed_wf", "permanent failure")

        runtime = _make_runtime(db)
        result = maos_workflow_dlq_inspect(runtime, dlq_id)
        assert "failed_wf" in result
        assert "permanent failure" in result
        db.close()

    def test_dlq_inspect_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        runtime = _make_runtime(db)
        result = maos_workflow_dlq_inspect(runtime, 99999)
        assert "not found" in result
        db.close()

    def test_dlq_mark_inspected(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        run_id = db.start_workflow_run("failed_wf", 3)
        dlq = DeadLetterQueue(db)
        dlq_id = dlq.enqueue(run_id, "failed_wf", "failed")

        runtime = _make_runtime(db)
        result = maos_workflow_dlq_mark_inspected(runtime, dlq_id, "investigated")
        assert "marked as inspected" in result

        entry = dlq.get_entry(dlq_id)
        assert entry["status"] == "inspected"
        assert entry["notes"] == "investigated"
        db.close()
