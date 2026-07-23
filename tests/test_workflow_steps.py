"""Tests for Phase 2: first-class workflow steps.

Tests:
  - Retry with exponential backoff
  - Per-step timeout
  - Per-step permission checks
  - Failure behavior (skip / abort / compensate)
  - Side effect declaration
  - All attempts tracked in StepResult.attempts
"""

import time
from unittest.mock import MagicMock

from app.db import Database
from app.events.bus import EventBus
from app.workflows.engine import WorkflowEngine
from app.workflows.models import (
    FailureBehavior,
    Workflow,
    WorkflowStep,
)

# --- Helpers ---

_call_count = 0


def _make_flaky_step(fail_n: int = 2):
    """Create a step that fails the first fail_n times, then succeeds."""
    counter = {"n": 0}

    def handler(ctx):
        counter["n"] += 1
        if counter["n"] <= fail_n:
            return {"ok": False, "error": f"transient failure #{counter['n']}"}
        return {"ok": True, "data": f"success on attempt {counter['n']}"}

    return handler


def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "permanent failure"}


def _slow_step(ctx):
    time.sleep(0.3)
    return {"ok": True, "data": "slow"}


def _exception_step(ctx):
    raise ValueError("handler crashed")


# --- Retry tests ---

class TestRetry:
    def test_retry_succeeds_on_second_attempt(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_retry", description="Test retry")
        wf.add_step(WorkflowStep(
            name="flaky", description="Fails twice then succeeds",
            handler=_make_flaky_step(fail_n=1),
            max_retries=2, retry_delay_ms=10,  # fast retry for tests
        ))
        result = engine.run(wf)

        assert result.ok
        assert result.steps[0].ok
        assert result.steps[0].attempt == 2  # succeeded on 2nd attempt
        assert len(result.steps[0].attempts) == 2
        assert result.steps[0].attempts[0]["ok"] is False
        assert result.steps[0].attempts[1]["ok"] is True
        db.close()

    def test_retry_exhausted(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_retry_exhausted", description="Test retry exhausted")
        wf.add_step(WorkflowStep(
            name="always_fails", description="Always fails",
            handler=_fail_step,
            max_retries=2, retry_delay_ms=10,
            on_failure=FailureBehavior.ABORT,
        ))
        result = engine.run(wf)

        assert not result.ok
        assert result.steps[0].attempt == 3  # 1 + 2 retries
        assert len(result.steps[0].attempts) == 3
        for attempt in result.steps[0].attempts:
            assert attempt["ok"] is False
        db.close()

    def test_no_retry_by_default(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_no_retry", description="Test no retry")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails once",
            handler=_fail_step,
            on_failure=FailureBehavior.ABORT,
        ))
        result = engine.run(wf)

        assert not result.ok
        assert result.steps[0].attempt == 1
        assert len(result.steps[0].attempts) == 1
        db.close()

    def test_retry_with_exception(self, tmp_path):
        """Retry should also apply to exceptions, not just ok=False."""
        call_count = {"n": 0}

        def flaky_exception(ctx):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("crash")
            return {"ok": True, "data": "recovered"}

        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_retry_exception", description="Test retry on exception")
        wf.add_step(WorkflowStep(
            name="crashy", description="Crashes then succeeds",
            handler=flaky_exception,
            max_retries=1, retry_delay_ms=10,
        ))
        result = engine.run(wf)

        assert result.ok
        assert result.steps[0].attempt == 2
        assert len(result.steps[0].attempts) == 2
        assert result.steps[0].attempts[0]["ok"] is False
        assert "crash" in result.steps[0].attempts[0]["error"]
        db.close()

    def test_retry_publishes_retry_event(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        bus = EventBus(db=db)
        engine = WorkflowEngine(db, bus=bus)

        wf = Workflow(name="test_retry_event", description="Test retry event")
        wf.add_step(WorkflowStep(
            name="flaky", description="Fails once",
            handler=_make_flaky_step(fail_n=1),
            max_retries=1, retry_delay_ms=10,
        ))
        engine.run(wf)

        events = db.query_events(limit=50)
        types = [e["type"] for e in events]
        assert "WORKFLOW_STEP_RETRY" in types
        db.close()

    def test_retry_persists_all_attempts(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_retry_persist", description="Test retry persistence")
        wf.add_step(WorkflowStep(
            name="flaky", description="Fails once",
            handler=_make_flaky_step(fail_n=1),
            max_retries=1, retry_delay_ms=10,
        ))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        assert len(step_runs) == 2  # two attempts persisted
        assert step_runs[0]["status"] == "failed"
        assert step_runs[1]["status"] == "completed"
        assert step_runs[0]["attempt"] == 1
        assert step_runs[1]["attempt"] == 2
        db.close()

    def test_exponential_backoff_delay(self, tmp_path):
        """Verify that retry delay doubles each time."""
        delays = []
        original_sleep = time.sleep

        def mock_sleep(seconds):
            delays.append(seconds)

        time.sleep = mock_sleep
        try:
            db = Database(str(tmp_path / "test.db"))
            engine = WorkflowEngine(db)
            wf = Workflow(name="test_backoff", description="Test backoff")
            wf.add_step(WorkflowStep(
                name="always_fails", description="Always fails",
                handler=_fail_step,
                max_retries=3, retry_delay_ms=100,
                on_failure=FailureBehavior.SKIP,  # don't abort
            ))
            engine.run(wf)

            # 3 retries → 3 sleeps with exponential backoff
            # 100ms, 200ms, 400ms
            assert len(delays) == 3
            assert delays[0] == 0.1
            assert delays[1] == 0.2
            assert delays[2] == 0.4
            db.close()
        finally:
            time.sleep = original_sleep


# --- Timeout tests ---

class TestTimeout:
    def test_timeout_fires(self, tmp_path):
        """A step that takes too long should time out."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_timeout", description="Test timeout")
        wf.add_step(WorkflowStep(
            name="slow", description="Takes 0.3s",
            handler=_slow_step,
            timeout_seconds=0.1,  # 100ms timeout
            on_failure=FailureBehavior.ABORT,
        ))
        result = engine.run(wf)

        assert not result.ok
        assert result.steps[0].timed_out is True
        assert "timed out" in result.steps[0].error.lower()
        db.close()

    def test_timeout_not_fired_when_fast(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_no_timeout", description="Test no timeout")
        wf.add_step(WorkflowStep(
            name="fast", description="Fast step",
            handler=_ok_step,
            timeout_seconds=5.0,  # 5s timeout, step is instant
        ))
        result = engine.run(wf)

        assert result.ok
        assert result.steps[0].timed_out is False
        db.close()

    def test_timeout_with_retry(self, tmp_path):
        """Timeout should be retried if retries are configured."""
        call_count = {"n": 0}

        def slow_then_fast(ctx):
            call_count["n"] += 1
            if call_count["n"] == 1:
                time.sleep(0.3)  # will timeout
            return {"ok": True, "data": "ok"}

        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_timeout_retry", description="Test timeout + retry")
        wf.add_step(WorkflowStep(
            name="slow_then_fast", description="Slow then fast",
            handler=slow_then_fast,
            timeout_seconds=0.1,
            max_retries=1, retry_delay_ms=10,
        ))
        result = engine.run(wf)

        # First attempt times out, second succeeds
        assert result.ok
        assert result.steps[0].attempt == 2
        assert len(result.steps[0].attempts) == 2
        assert result.steps[0].attempts[0].get("timed_out") is True
        db.close()

    def test_timeout_persists_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_timeout_persist", description="Test timeout persistence")
        wf.add_step(WorkflowStep(
            name="slow", description="Takes too long",
            handler=_slow_step,
            timeout_seconds=0.1,
            on_failure=FailureBehavior.ABORT,
        ))
        result = engine.run(wf)

        step_runs = db.step_runs_for_workflow(result.run_id)
        assert step_runs[0]["status"] == "timed_out"
        db.close()


# --- Permission tests ---

class TestPermissions:
    def test_permission_denied(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        checker = MagicMock()
        checker.check = MagicMock(return_value=False)
        engine = WorkflowEngine(db, permission_checker=checker)

        wf = Workflow(name="test_perm_denied", description="Test permission denied")
        wf.add_step(WorkflowStep(
            name="restricted", description="Needs permission",
            handler=_ok_step,
            permissions=["filesystem.write"],
        ))
        result = engine.run(wf)

        assert not result.ok
        assert "permission denied" in result.steps[0].error
        checker.check.assert_called_once_with(["filesystem.write"])
        db.close()

    def test_permission_granted(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        checker = MagicMock()
        checker.check = MagicMock(return_value=True)
        engine = WorkflowEngine(db, permission_checker=checker)

        wf = Workflow(name="test_perm_granted", description="Test permission granted")
        wf.add_step(WorkflowStep(
            name="restricted", description="Needs permission",
            handler=_ok_step,
            permissions=["filesystem.read"],
        ))
        result = engine.run(wf)

        assert result.ok
        checker.check.assert_called_once_with(["filesystem.read"])
        db.close()

    def test_no_permission_checker(self, tmp_path):
        """Without a permission checker, permissions are not checked."""
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)  # no permission_checker

        wf = Workflow(name="test_no_checker", description="Test no checker")
        wf.add_step(WorkflowStep(
            name="restricted", description="Has permissions but no checker",
            handler=_ok_step,
            permissions=["filesystem.write"],
        ))
        result = engine.run(wf)

        assert result.ok  # permissions ignored without checker
        db.close()

    def test_no_permissions_declared(self, tmp_path):
        """Steps without permissions don't trigger checks."""
        db = Database(str(tmp_path / "test.db"))
        checker = MagicMock()
        engine = WorkflowEngine(db, permission_checker=checker)

        wf = Workflow(name="test_no_perms", description="Test no permissions")
        wf.add_step(WorkflowStep(
            name="open", description="No permissions needed",
            handler=_ok_step,
        ))
        result = engine.run(wf)

        assert result.ok
        checker.check.assert_not_called()
        db.close()


# --- Failure behavior tests ---

class TestFailureBehavior:
    def test_abort_stops_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_abort", description="Test abort")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails and aborts",
            handler=_fail_step,
            on_failure=FailureBehavior.ABORT,
        ))
        wf.add_step(WorkflowStep(
            name="never_runs", description="Should not run",
            handler=_ok_step,
        ))
        result = engine.run(wf)

        assert not result.ok
        assert len(result.steps) == 1  # only the failed step
        db.close()

    def test_skip_continues_workflow(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_skip", description="Test skip")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails but skips",
            handler=_fail_step,
            on_failure=FailureBehavior.SKIP,
        ))
        wf.add_step(WorkflowStep(
            name="runs", description="Should run",
            handler=_ok_step,
        ))
        result = engine.run(wf)

        assert result.ok  # workflow succeeds because failure was skipped
        assert len(result.steps) == 2
        assert result.steps[0].ok is False
        assert result.steps[1].ok is True
        db.close()

    def test_compensate_runs_compensation(self, tmp_path):
        compensated = {"ran": False}

        def compensation(ctx, step_result):
            compensated["ran"] = True
            return {"ok": True, "compensated": True}

        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_compensate", description="Test compensation")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails and compensates",
            handler=_fail_step,
            on_failure=FailureBehavior.COMPENSATE,
            compensation=compensation,
        ))
        wf.add_step(WorkflowStep(
            name="runs", description="Should run after compensation",
            handler=_ok_step,
        ))
        result = engine.run(wf)

        assert result.ok  # compensation allows workflow to continue
        assert result.steps[0].ok is False
        assert result.steps[0].compensated is True
        assert compensated["ran"] is True
        assert result.steps[1].ok is True
        db.close()

    def test_compensate_without_handler_aborts(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_compensate_no_handler", description="Test compensate without handler")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails, compensate but no handler",
            handler=_fail_step,
            on_failure=FailureBehavior.COMPENSATE,
            # no compensation handler
        ))
        wf.add_step(WorkflowStep(
            name="never_runs", description="Should not run",
            handler=_ok_step,
        ))
        result = engine.run(wf)

        assert not result.ok  # aborts because no compensation handler
        assert len(result.steps) == 1
        db.close()

    def test_compensation_failure_aborts(self, tmp_path):
        def bad_compensation(ctx, step_result):
            raise RuntimeError("compensation failed")

        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_compensation_fails", description="Test compensation failure")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails, compensation also fails",
            handler=_fail_step,
            on_failure=FailureBehavior.COMPENSATE,
            compensation=bad_compensation,
        ))
        wf.add_step(WorkflowStep(
            name="never_runs", description="Should not run",
            handler=_ok_step,
        ))
        result = engine.run(wf)

        assert not result.ok
        assert len(result.steps) == 1
        db.close()

    def test_compensate_publishes_event(self, tmp_path):
        def compensation(ctx, step_result):
            return {"ok": True}

        db = Database(str(tmp_path / "test.db"))
        bus = EventBus(db=db)
        engine = WorkflowEngine(db, bus=bus)
        wf = Workflow(name="test_compensate_event", description="Test compensate event")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails and compensates",
            handler=_fail_step,
            on_failure=FailureBehavior.COMPENSATE,
            compensation=compensation,
        ))
        wf.add_step(WorkflowStep(name="ok", description="OK", handler=_ok_step))
        engine.run(wf)

        events = db.query_events(limit=50)
        types = [e["type"] for e in events]
        assert "WORKFLOW_STEP_COMPENSATED" in types
        db.close()


# --- Side effect declaration tests ---

class TestSideEffects:
    def test_step_with_side_effects(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_side_effects", description="Test side effects")
        wf.add_step(WorkflowStep(
            name="sends_email", description="Sends an email",
            handler=_ok_step,
            side_effects=["email.send"],
        ))
        result = engine.run(wf)

        assert result.ok
        assert result.steps[0].data.get("ok") is True
        db.close()

    def test_step_to_dict_includes_side_effects(self):
        step = WorkflowStep(
            name="test", description="Test",
            handler=_ok_step,
            side_effects=["email.send", "task.create"],
        )
        d = step.to_dict()
        assert "side_effects" in d
        assert "email.send" in d["side_effects"]
        assert "task.create" in d["side_effects"]


# --- StepResult attempts tracking tests ---

class TestAttemptsTracking:
    def test_single_attempt_tracked(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_single_attempt", description="Test single attempt")
        wf.add_step(WorkflowStep(name="ok", description="OK", handler=_ok_step))
        result = engine.run(wf)

        assert len(result.steps[0].attempts) == 1
        assert result.steps[0].attempts[0]["ok"] is True
        assert result.steps[0].attempts[0]["attempt"] == 1
        db.close()

    def test_multiple_attempts_tracked(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_multi_attempt", description="Test multi attempt")
        wf.add_step(WorkflowStep(
            name="flaky", description="Fails twice",
            handler=_make_flaky_step(fail_n=2),
            max_retries=3, retry_delay_ms=10,
        ))
        result = engine.run(wf)

        assert len(result.steps[0].attempts) == 3
        assert result.steps[0].attempts[0]["ok"] is False
        assert result.steps[0].attempts[1]["ok"] is False
        assert result.steps[0].attempts[2]["ok"] is True
        db.close()

    def test_step_result_to_dict_includes_attempts(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_dict_attempts", description="Test dict attempts")
        wf.add_step(WorkflowStep(
            name="flaky", description="Fails once",
            handler=_make_flaky_step(fail_n=1),
            max_retries=1, retry_delay_ms=10,
        ))
        result = engine.run(wf)

        d = result.steps[0].to_dict()
        assert "attempts" in d
        assert len(d["attempts"]) == 2
        db.close()

    def test_step_result_to_dict_includes_compensated(self, tmp_path):
        def compensation(ctx, step_result):
            return {"ok": True}

        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_dict_comp", description="Test dict compensated")
        wf.add_step(WorkflowStep(
            name="fails", description="Fails and compensates",
            handler=_fail_step,
            on_failure=FailureBehavior.COMPENSATE,
            compensation=compensation,
        ))
        wf.add_step(WorkflowStep(name="ok", description="OK", handler=_ok_step))
        result = engine.run(wf)

        d = result.steps[0].to_dict()
        assert d["compensated"] is True
        db.close()

    def test_step_result_to_dict_includes_timed_out(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        engine = WorkflowEngine(db)
        wf = Workflow(name="test_dict_timeout", description="Test dict timed_out")
        wf.add_step(WorkflowStep(
            name="slow", description="Takes too long",
            handler=_slow_step,
            timeout_seconds=0.1,
            on_failure=FailureBehavior.ABORT,
        ))
        result = engine.run(wf)

        d = result.steps[0].to_dict()
        assert d["timed_out"] is True
        db.close()
