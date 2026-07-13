"""Tests for the workflow engine, verification, and morning routine."""

from app.db import Database
from app.events.bus import EventBus
from app.workflows.engine import WorkflowEngine
from app.workflows.models import (
    VerificationResult,
    Workflow,
    WorkflowStep,
)


def _ok_step(ctx):
    return {"ok": True, "value": "done"}


def _fail_step(ctx):
    return {"ok": False, "error": "something went wrong"}


def _always_ok_step(ctx):
    return {"ok": True, "data": "always succeeds"}


def _verify_ok(ctx):
    return VerificationResult(success=True, evidence=["verified"])


def _verify_fail(ctx):
    return VerificationResult(success=False, error="evidence not found")


def test_workflow_simple(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    engine = WorkflowEngine(db, bus=bus)

    wf = Workflow(name="test_simple", description="A simple test")
    wf.add_step(WorkflowStep(name="step1", description="First step", handler=_ok_step))
    wf.add_step(WorkflowStep(name="step2", description="Second step", handler=_always_ok_step))

    result = engine.run(wf)
    assert result.ok
    assert len(result.steps) == 2
    assert result.steps_completed == 2
    assert result.steps_failed == 0
    assert result.run_id is not None

    # Check workflow run was recorded.
    runs = db.recent_workflow_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    db.close()


def test_workflow_required_step_failure(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    engine = WorkflowEngine(db)

    wf = Workflow(name="test_fail", description="A failing workflow")
    wf.add_step(WorkflowStep(name="step1", description="OK step", handler=_ok_step))
    wf.add_step(WorkflowStep(name="step2", description="Failing step", handler=_fail_step, required=True))
    wf.add_step(WorkflowStep(name="step3", description="Should not run", handler=_ok_step))

    result = engine.run(wf)
    assert not result.ok
    assert len(result.steps) == 2  # step3 should not run
    assert result.steps[1].ok is False
    assert "required step" in result.error

    runs = db.recent_workflow_runs()
    assert runs[0]["status"] == "failed"
    db.close()


def test_workflow_optional_step_failure(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    engine = WorkflowEngine(db)

    wf = Workflow(name="test_optional_fail", description="Optional step fails")
    wf.add_step(WorkflowStep(name="step1", description="OK step", handler=_ok_step))
    wf.add_step(WorkflowStep(name="step2", description="Optional failing", handler=_fail_step, required=False))
    wf.add_step(WorkflowStep(name="step3", description="Should still run", handler=_ok_step))

    result = engine.run(wf)
    assert result.ok  # workflow succeeds because the failing step is optional
    assert len(result.steps) == 3
    assert result.steps[1].ok is False
    assert result.steps[2].ok is True
    db.close()


def test_workflow_verification_success(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    engine = WorkflowEngine(db)

    wf = Workflow(name="test_verify_ok", description="Verification passes")
    wf.add_step(WorkflowStep(
        name="step1", description="Verified step",
        handler=_ok_step, verify=_verify_ok,
    ))

    result = engine.run(wf)
    assert result.ok
    assert result.steps[0].verification is not None
    assert result.steps[0].verification.success is True
    assert "verified" in result.steps[0].verification.evidence
    db.close()


def test_workflow_verification_failure(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    engine = WorkflowEngine(db)

    wf = Workflow(name="test_verify_fail", description="Verification fails")
    wf.add_step(WorkflowStep(
        name="step1", description="Step that fails verification",
        handler=_ok_step, verify=_verify_fail, required=True,
    ))

    result = engine.run(wf)
    assert not result.ok
    assert result.steps[0].verification is not None
    assert result.steps[0].verification.success is False
    assert "verification failed" in result.steps[0].error
    db.close()


def test_workflow_events_published(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    engine = WorkflowEngine(db, bus=bus)

    wf = Workflow(name="test_events", description="Events test")
    wf.add_step(WorkflowStep(name="step1", description="Step", handler=_ok_step))

    engine.run(wf)

    events = db.query_events(limit=20)
    types = [e["type"] for e in events]
    assert "WORKFLOW_STARTED" in types
    assert "WORKFLOW_STEP_COMPLETED" in types
    assert "WORKFLOW_COMPLETED" in types
    db.close()


def test_morning_workflow_structure():
    from app.workflows.morning import morning_workflow

    wf = morning_workflow()
    assert wf.name == "morning_workspace_restore"
    assert len(wf.steps) == 6

    step_names = [s.name for s in wf.steps]
    assert "reconstruct_last_session" in step_names
    assert "summarize_changes" in step_names
    assert "check_pending_tasks" in step_names
    assert "check_system_health" in step_names
    assert "prepare_workspace" in step_names
    assert "create_daily_plan" in step_names


def test_morning_workflow_runs(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    engine = WorkflowEngine(db, bus=bus)

    from app.workflows.morning import morning_workflow

    wf = morning_workflow()
    result = engine.run(wf, context={"action_executor": None})

    # All steps are optional except the last, so it should succeed
    # even with no data.
    assert result.ok
    assert len(result.steps) == 6

    # The daily plan step should produce a plan.
    plan_step = next(s for s in result.steps if s.step_name == "create_daily_plan")
    assert plan_step.ok
    assert len(plan_step.data["plan"]) > 0
    db.close()


def test_morning_report_format(tmp_path):
    from app.workflows.morning import format_morning_report, morning_workflow

    db = Database(str(tmp_path / "test.db"))
    engine = WorkflowEngine(db)

    wf = morning_workflow()
    result = engine.run(wf)
    report = format_morning_report(result)

    assert isinstance(report, str)
    assert "Good morning" in report
    db.close()
