"""Tests for Phase 8: workflow engine as runtime.

Tests:
  - WorkflowRegistry: register, get, list, describe
  - WorkflowRuntime: start (sync), start (async), pause, resume, cancel, retry, inspect
  - Async execution with ThreadPoolExecutor
  - List active runs
"""

import time

import pytest

from app.db import Database
from app.workflows.models import Workflow, WorkflowStep
from app.workflows.runtime import WorkflowRegistry, WorkflowRuntime
from app.workflows.states import WorkflowStatus

# --- Helpers ---

def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "permanent failure"}


def _slow_step(ctx):
    time.sleep(0.1)
    return {"ok": True, "data": "slow"}


def _make_simple_workflow():
    wf = Workflow(name="simple", description="A simple workflow")
    wf.add_step(WorkflowStep(name="step1", description="First", handler=_ok_step))
    return wf


def _make_failing_workflow():
    wf = Workflow(name="failing", description="A failing workflow")
    wf.add_step(WorkflowStep(
        name="fails", description="Fails", handler=_fail_step,
        on_failure=__import__(
            "app.workflows.models", fromlist=["FailureBehavior"]
        ).FailureBehavior.ABORT,
    ))
    return wf


def _make_slow_workflow():
    wf = Workflow(name="slow", description="A slow workflow")
    wf.add_step(WorkflowStep(name="slow1", description="Slow", handler=_slow_step))
    wf.add_step(WorkflowStep(name="slow2", description="Slow", handler=_slow_step))
    return wf


# --- WorkflowRegistry tests ---

class TestWorkflowRegistry:
    def test_register_and_get(self):
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow, "A simple workflow")
        wf = reg.get("simple")
        assert wf is not None
        assert wf.name == "simple"
        assert len(wf.steps) == 1

    def test_get_nonexistent(self):
        reg = WorkflowRegistry()
        assert reg.get("nonexistent") is None

    def test_list_names(self):
        reg = WorkflowRegistry()
        reg.register("wf1", _make_simple_workflow)
        reg.register("wf2", _make_failing_workflow)
        names = reg.list_names()
        assert "wf1" in names
        assert "wf2" in names
        assert len(names) == 2

    def test_describe(self):
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow, "A simple workflow")
        assert reg.describe("simple") == "A simple workflow"

    def test_describe_nonexistent(self):
        reg = WorkflowRegistry()
        assert reg.describe("nonexistent") == ""

    def test_contains(self):
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        assert reg.contains("simple")
        assert not reg.contains("nonexistent")

    def test_factory_creates_fresh_instances(self):
        """Each call to get() should create a fresh Workflow instance."""
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        wf1 = reg.get("simple")
        wf2 = reg.get("simple")
        assert wf1 is not wf2  # different instances
        assert wf1.workflow_id != wf2.workflow_id  # different UUIDs


# --- WorkflowRuntime sync tests ---

class TestWorkflowRuntimeSync:
    def test_start_sync(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        result = runtime.start("simple")
        assert result.ok
        assert result.run_id is not None
        db.close()

    def test_start_unknown_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)

        with pytest.raises(ValueError, match="not registered"):
            runtime.start("nonexistent")
        db.close()

    def test_start_with_context(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        result = runtime.start("simple", context={"custom_key": "custom_value"})
        assert result.ok
        db.close()

    def test_start_failing_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("failing", _make_failing_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        result = runtime.start("failing")
        assert not result.ok
        db.close()


# --- WorkflowRuntime async tests ---

class TestWorkflowRuntimeAsync:
    def test_start_async(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        run_id = runtime.start("simple", async_exec=True)
        assert isinstance(run_id, int)

        # Wait for completion.
        result = runtime.get_result(run_id, timeout=5.0)
        assert result is not None
        assert result.ok
        db.close()

    def test_is_complete(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        run_id = runtime.start("simple", async_exec=True)
        # Wait for it to complete.
        runtime.get_result(run_id, timeout=5.0)
        assert runtime.is_complete(run_id)
        db.close()

    def test_is_complete_not_done(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("slow", _make_slow_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        run_id = runtime.start("slow", async_exec=True)
        # Immediately check — might not be done yet.
        # (This is a race condition, but the test is just checking the API works.)
        runtime.is_complete(run_id)  # should not raise
        # Wait for completion.
        runtime.get_result(run_id, timeout=10.0)
        db.close()

    def test_get_result_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)
        assert runtime.get_result(99999) is None
        db.close()


# --- WorkflowRuntime lifecycle tests ---

class TestWorkflowRuntimeLifecycle:
    def test_cancel(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        # Cancel a non-running workflow.
        assert runtime.cancel(99999) is False
        db.close()

    def test_retry(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("failing", _make_failing_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        # First run fails.
        result1 = runtime.start("failing")
        assert not result1.ok

        # Retry — also fails (same workflow).
        result2 = runtime.retry(result1.run_id)
        assert not result2.ok
        assert result2.run_id != result1.run_id  # new run
        db.close()

    def test_retry_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)

        result = runtime.retry(99999)
        assert not result.ok
        assert "not found" in result.error
        db.close()

    def test_retry_unregistered_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)

        # Create a run manually for a workflow that's not registered.
        run_id = db.start_workflow_run("unregistered_wf", 1)
        result = runtime.retry(run_id)
        assert not result.ok
        assert "not registered" in result.error
        db.close()

    def test_inspect(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        result = runtime.start("simple")
        details = runtime.inspect(result.run_id)
        assert details is not None
        assert details["run"]["workflow"] == "simple"
        db.close()

    def test_inspect_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)

        assert runtime.inspect(99999) is None
        db.close()

    def test_get_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        result = runtime.start("simple")
        status = runtime.get_status(result.run_id)
        assert status == WorkflowStatus.COMPLETED
        db.close()

    def test_get_status_nonexistent(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)

        assert runtime.get_status(99999) is None
        db.close()


# --- WorkflowRuntime listing tests ---

class TestWorkflowRuntimeListing:
    def test_list_active_empty(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        runtime = WorkflowRuntime(db=db, registry=reg)

        active = runtime.list_active()
        assert len(active) == 0
        db.close()

    def test_list_definitions(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        reg.register("failing", _make_failing_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        defs = runtime.list_definitions()
        assert "simple" in defs
        assert "failing" in defs
        db.close()

    def test_list_active_with_completed(self, tmp_path):
        """Completed workflows should not appear in list_active."""
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        runtime.start("simple")  # completes synchronously
        active = runtime.list_active()
        assert len(active) == 0  # completed, not active
        db.close()


# --- WorkflowRuntime shutdown tests ---

class TestWorkflowRuntimeShutdown:
    def test_shutdown(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        reg = WorkflowRegistry()
        reg.register("simple", _make_simple_workflow)
        runtime = WorkflowRuntime(db=db, registry=reg)

        runtime.start("simple", async_exec=True)
        runtime.shutdown(wait=True)
        # Should not raise.
        db.close()
