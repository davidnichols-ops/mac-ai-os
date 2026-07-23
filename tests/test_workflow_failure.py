"""Tests for Phase 6: failure, retry, and compensation semantics.

Tests:
  - CompensationStrategy enum
  - FailurePolicy dataclass
  - CompensationRecord model
  - SagaCoordinator: compensates failed step + all completed steps in reverse order
  - DeadLetterQueue: enqueue, list, inspect, reprocess
"""

import json

from app.db import Database
from app.workflows.failure import (
    CompensationRecord,
    CompensationStrategy,
    DeadLetterQueue,
    FailurePolicy,
    SagaCoordinator,
)
from app.workflows.models import (
    StepResult,
    WorkflowStep,
)

# --- Helpers ---

def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "permanent failure"}


# --- CompensationStrategy tests ---

class TestCompensationStrategy:
    def test_values(self):
        assert CompensationStrategy.NONE.value == "none"
        assert CompensationStrategy.STEP_ONLY.value == "step_only"
        assert CompensationStrategy.SAGA.value == "saga"

    def test_str(self):
        assert str(CompensationStrategy.SAGA) == "saga"


# --- FailurePolicy tests ---

class TestFailurePolicy:
    def test_default_policy(self):
        policy = FailurePolicy()
        assert policy.strategy == CompensationStrategy.NONE
        assert policy.max_workflow_retries == 0
        assert policy.dead_letter_on_failure is True

    def test_saga_policy(self):
        policy = FailurePolicy(
            strategy=CompensationStrategy.SAGA,
            max_workflow_retries=2,
            workflow_retry_delay_ms=1000,
        )
        assert policy.strategy == CompensationStrategy.SAGA
        assert policy.max_workflow_retries == 2

    def test_to_dict(self):
        policy = FailurePolicy(strategy=CompensationStrategy.SAGA, timeout_seconds=30.0)
        d = policy.to_dict()
        assert d["strategy"] == "saga"
        assert d["timeout_seconds"] == 30.0
        assert d["dead_letter_on_failure"] is True


# --- CompensationRecord tests ---

class TestCompensationRecord:
    def test_success_record(self):
        record = CompensationRecord(
            step_name="step1", step_id="abc123", success=True,
            duration_ms=50, data={"result": "rolled back"},
        )
        assert record.success is True
        assert record.duration_ms == 50

    def test_failure_record(self):
        record = CompensationRecord(
            step_name="step2", step_id="def456", success=False,
            error="compensation failed",
        )
        assert record.success is False
        assert "compensation failed" in record.error

    def test_to_dict(self):
        record = CompensationRecord(
            step_name="step1", step_id="abc123", success=True,
            data={"key": "value"},
        )
        d = record.to_dict()
        assert d["step_name"] == "step1"
        assert d["step_id"] == "abc123"
        assert d["success"] is True
        assert d["data"]["key"] == "value"


# --- SagaCoordinator tests ---

class TestSagaCoordinator:
    def test_compensate_failed_step_only(self, tmp_path):
        """Compensate only the failed step (no completed steps)."""
        db = Database(str(tmp_path / "test.db"))
        coordinator = SagaCoordinator(db=db)

        compensated = {"ran": False}

        def compensation(ctx, step_result):
            compensated["ran"] = True
            return {"ok": True, "rolled_back": True}

        failed_step = WorkflowStep(
            name="failed_step", description="Fails",
            handler=_fail_step, compensation=compensation,
        )

        records = coordinator.compensate(failed_step, [], ctx={})
        assert len(records) == 1
        assert records[0].step_name == "failed_step"
        assert records[0].success is True
        assert compensated["ran"] is True
        db.close()

    def test_compensate_saga_reverse_order(self, tmp_path):
        """Saga: compensate failed step + all completed steps in reverse order."""
        db = Database(str(tmp_path / "test.db"))
        coordinator = SagaCoordinator(db=db)

        comp_order = []

        def comp1(ctx, sr):
            comp_order.append("step1")
            return {"ok": True}

        def comp2(ctx, sr):
            comp_order.append("step2")
            return {"ok": True}

        def comp_failed(ctx, sr):
            comp_order.append("failed_step")
            return {"ok": True}

        step1 = WorkflowStep(name="step1", description="S1", handler=_ok_step, compensation=comp1)
        step2 = WorkflowStep(name="step2", description="S2", handler=_ok_step, compensation=comp2)
        failed_step = WorkflowStep(
            name="failed_step", description="F", handler=_fail_step, compensation=comp_failed,
        )

        # Simulate: step1 and step2 completed, then failed_step failed.
        completed = [
            (step1, StepResult(step_name="step1", ok=True, step_id=step1.step_id)),
            (step2, StepResult(step_name="step2", ok=True, step_id=step2.step_id)),
        ]

        records = coordinator.compensate(failed_step, completed, ctx={})

        # Should compensate in order: failed_step, step2, step1 (reverse)
        assert len(records) == 3
        assert comp_order == ["failed_step", "step2", "step1"]
        assert all(r.success for r in records)
        db.close()

    def test_compensate_no_compensation_handlers(self, tmp_path):
        """Steps without compensation handlers are skipped."""
        db = Database(str(tmp_path / "test.db"))
        coordinator = SagaCoordinator(db=db)

        step1 = WorkflowStep(name="step1", description="S1", handler=_ok_step)
        step2 = WorkflowStep(name="step2", description="S2", handler=_ok_step)
        failed_step = WorkflowStep(name="failed", description="F", handler=_fail_step)

        completed = [
            (step1, StepResult(step_name="step1", ok=True)),
            (step2, StepResult(step_name="step2", ok=True)),
        ]

        records = coordinator.compensate(failed_step, completed, ctx={})
        assert len(records) == 0  # no compensation handlers
        db.close()

    def test_compensate_partial_failure(self, tmp_path):
        """Some compensations may fail — they should be recorded as failed."""
        db = Database(str(tmp_path / "test.db"))
        coordinator = SagaCoordinator(db=db)

        def good_comp(ctx, sr):
            return {"ok": True}

        def bad_comp(ctx, sr):
            raise RuntimeError("compensation crashed")

        step1 = WorkflowStep(name="step1", description="S1", handler=_ok_step, compensation=good_comp)
        step2 = WorkflowStep(name="step2", description="S2", handler=_ok_step, compensation=bad_comp)
        failed_step = WorkflowStep(name="failed", description="F", handler=_fail_step)

        completed = [
            (step1, StepResult(step_name="step1", ok=True)),
            (step2, StepResult(step_name="step2", ok=True)),
        ]

        records = coordinator.compensate(failed_step, completed, ctx={})
        assert len(records) == 2  # step2 and step1 (failed_step has no comp)
        # step2's comp fails, step1's comp succeeds
        assert records[0].step_name == "step2"
        assert records[0].success is False
        assert "compensation crashed" in records[0].error
        assert records[1].step_name == "step1"
        assert records[1].success is True
        db.close()

    def test_compensate_records_duration(self, tmp_path):
        """Compensation records should include duration."""
        import time
        db = Database(str(tmp_path / "test.db"))
        coordinator = SagaCoordinator(db=db)

        def slow_comp(ctx, sr):
            time.sleep(0.05)
            return {"ok": True}

        failed_step = WorkflowStep(
            name="failed", description="F", handler=_fail_step, compensation=slow_comp,
        )

        records = coordinator.compensate(failed_step, [], ctx={})
        assert records[0].duration_ms >= 40  # at least ~50ms
        db.close()


# --- DeadLetterQueue tests ---

class TestDeadLetterQueue:
    def test_enqueue(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        # Create a workflow run first.
        run_id = db.start_workflow_run("test_wf", 3)
        dlq_id = dlq.enqueue(run_id, "test_wf", "step2 failed permanently")

        assert dlq_id is not None
        entry = dlq.get_entry(dlq_id)
        assert entry is not None
        assert entry["workflow_name"] == "test_wf"
        assert entry["failure_reason"] == "step2 failed permanently"
        assert entry["status"] == "pending"
        db.close()

    def test_list_entries(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        run1 = db.start_workflow_run("wf1", 2)
        run2 = db.start_workflow_run("wf2", 2)
        dlq.enqueue(run1, "wf1", "failed 1")
        dlq.enqueue(run2, "wf2", "failed 2")

        entries = dlq.list_entries()
        assert len(entries) == 2
        db.close()

    def test_list_filtered_by_status(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        run1 = db.start_workflow_run("wf1", 2)
        run2 = db.start_workflow_run("wf2", 2)
        id1 = dlq.enqueue(run1, "wf1", "failed 1")
        dlq.enqueue(run2, "wf2", "failed 2")

        dlq.mark_inspected(id1, "checked by david")

        pending = dlq.list_entries(status="pending")
        inspected = dlq.list_entries(status="inspected")
        assert len(pending) == 1
        assert len(inspected) == 1
        assert pending[0]["workflow_name"] == "wf2"
        assert inspected[0]["workflow_name"] == "wf1"
        db.close()

    def test_mark_inspected(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        run_id = db.start_workflow_run("test_wf", 2)
        dlq_id = dlq.enqueue(run_id, "test_wf", "failed")

        dlq.mark_inspected(dlq_id, "investigated, root cause: bad data")
        entry = dlq.get_entry(dlq_id)
        assert entry["status"] == "inspected"
        assert "bad data" in entry["notes"]
        assert entry["inspected_at"] is not None
        db.close()

    def test_mark_reprocessed(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        run_id = db.start_workflow_run("test_wf", 2)
        dlq_id = dlq.enqueue(run_id, "test_wf", "failed")

        new_run_id = db.start_workflow_run("test_wf", 2)
        dlq.mark_reprocessed(dlq_id, new_run_id)

        entry = dlq.get_entry(dlq_id)
        assert entry["status"] == "reprocessed"
        assert entry["new_run_id"] == new_run_id
        assert entry["reprocessed_at"] is not None
        db.close()

    def test_count(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        assert dlq.count() == 0

        run1 = db.start_workflow_run("wf1", 2)
        run2 = db.start_workflow_run("wf2", 2)
        dlq.enqueue(run1, "wf1", "failed")
        dlq.enqueue(run2, "wf2", "failed")

        assert dlq.count() == 2
        assert dlq.count("pending") == 2
        assert dlq.count("inspected") == 0
        db.close()

    def test_enqueue_with_compensation_records(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        dlq = DeadLetterQueue(db)

        run_id = db.start_workflow_run("test_wf", 3)
        comp_records = [
            CompensationRecord(step_name="step2", step_id="abc", success=True),
            CompensationRecord(step_name="step1", step_id="def", success=True),
        ]
        dlq_id = dlq.enqueue(
            run_id, "test_wf", "step3 failed",
            compensation_records=comp_records,
            context_snapshot={"key": "value"},
        )

        entry = dlq.get_entry(dlq_id)
        comps = json.loads(entry["compensation_records"])
        assert len(comps) == 2
        assert comps[0]["step_name"] == "step2"
        ctx = json.loads(entry["context_snapshot"])
        assert ctx["key"] == "value"
        db.close()
