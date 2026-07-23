"""Tests for Phase 4: action proposal, authorization, and execution pipeline.

Tests:
  - ActionProposal model
  - AuthorizationPipeline: permission checks, reversibility, danger scoring
  - TrustcardIntegrationPoint: danger scoring and thresholds
  - WorkflowEngine.execute_action_proposal: full integration
  - Idempotency: duplicate action proposals are skipped
  - Approval queue: irreversible actions route to approval queue
"""

from unittest.mock import MagicMock

from app.actions.executor import ActionExecutor
from app.actions.models import Action, ActionRequest, ActionResult
from app.actions.permissions import PermissionChecker
from app.actions.registry import ActionRegistry
from app.db import Database
from app.events.bus import EventBus
from app.workflows.engine import WorkflowEngine
from app.workflows.models import Workflow, WorkflowStep
from app.workflows.proposals import (
    ActionProposal,
    AuthorizationPipeline,
    TrustcardIntegrationPoint,
)

# --- Helpers ---

def _make_registry_with_actions(*actions: Action) -> ActionRegistry:
    """Create a registry populated with the given actions.

    Each action is registered with a no-op handler (tests that need a real
    handler should pass it via the handler_map argument).
    """
    reg = ActionRegistry()
    def noop(req, ctx):
        return ActionResult(
            ok=True, action=req.action, target=req.target, result="noop", executed=True,
        )
    for action in actions:
        reg.register(action, noop)
    return reg


def _make_registry_with_handlers(action: Action, handler) -> ActionRegistry:
    """Create a registry with a single action and its handler."""
    reg = ActionRegistry()
    reg.register(action, handler)
    return reg


def _make_executor(
    registry: ActionRegistry, permissions: PermissionChecker,
    db: Database, bus: EventBus | None = None,
    *, allow_irreversible: bool = False,
) -> ActionExecutor:
    return ActionExecutor(
        registry=registry, permissions=permissions, db=db, bus=bus,
        allow_irreversible=allow_irreversible,
    )


# --- ActionProposal model tests ---

class TestActionProposal:
    def test_basic_proposal(self):
        p = ActionProposal(action_name="send_email", target="user@example.com")
        assert p.action_name == "send_email"
        assert p.target == "user@example.com"
        assert p.parameters == {}
        assert p.risk_level == "low"
        assert p.requires_approval is False

    def test_proposal_with_parameters(self):
        p = ActionProposal(
            action_name="create_task",
            parameters={"title": "Buy groceries", "priority": "high"},
            reason="User requested task creation",
            risk_level="low",
        )
        assert p.parameters["title"] == "Buy groceries"
        assert p.reason == "User requested task creation"

    def test_to_action_request(self):
        p = ActionProposal(
            action_name="open_workspace",
            target="~/Projects/myproject",
            parameters={"editor": "vscode"},
        )
        req = p.to_action_request()
        assert isinstance(req, ActionRequest)
        assert req.action == "open_workspace"
        assert req.target == "~/Projects/myproject"
        assert req.parameters == {"editor": "vscode"}

    def test_to_dict(self):
        p = ActionProposal(
            action_name="test_action",
            target="test_target",
            parameters={"key": "value"},
            reason="testing",
            risk_level="medium",
            requires_approval=True,
            idempotency_key="test-key-123",
            side_effect_type="test.effect",
        )
        d = p.to_dict()
        assert d["action_name"] == "test_action"
        assert d["target"] == "test_target"
        assert d["parameters"] == {"key": "value"}
        assert d["reason"] == "testing"
        assert d["risk_level"] == "medium"
        assert d["requires_approval"] is True
        assert d["idempotency_key"] == "test-key-123"
        assert d["side_effect_type"] == "test.effect"


# --- TrustcardIntegrationPoint tests ---

class TestTrustcardIntegrationPoint:
    def test_default_no_danger(self):
        tc = TrustcardIntegrationPoint()
        score, flags = tc.score_action("any_action")
        assert score == 0.0
        assert flags == []

    def test_is_dangerous_below_threshold(self):
        tc = TrustcardIntegrationPoint(danger_threshold=70.0)
        assert tc.is_dangerous(50.0) is False
        assert tc.is_dangerous(69.9) is False

    def test_is_dangerous_at_threshold(self):
        tc = TrustcardIntegrationPoint(danger_threshold=70.0)
        assert tc.is_dangerous(70.0) is True
        assert tc.is_dangerous(100.0) is True

    def test_custom_threshold(self):
        tc = TrustcardIntegrationPoint(danger_threshold=30.0)
        assert tc.is_dangerous(35.0) is True
        assert tc.is_dangerous(25.0) is False

    def test_custom_scoring_subclass(self):
        """Subclasses can override score_action for real danger scoring."""
        class CustomTrustcard(TrustcardIntegrationPoint):
            def score_action(self, action_name, target="", parameters=None):
                if "delete" in action_name.lower():
                    return 85.0, ["destructive_action"]
                if "send" in action_name.lower():
                    return 40.0, ["external_communication"]
                return 10.0, []

        tc = CustomTrustcard()
        score, flags = tc.score_action("delete_file", "/important.txt")
        assert score == 85.0
        assert "destructive_action" in flags
        assert tc.is_dangerous(score) is True

        score, flags = tc.score_action("send_email")
        assert score == 40.0
        assert tc.is_dangerous(score) is False


# --- AuthorizationPipeline tests ---

class TestAuthorizationPipeline:
    def test_unknown_action_denied(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        registry = _make_registry_with_actions()
        pipeline = AuthorizationPipeline(
            action_registry=registry, db=db,
        )
        proposal = ActionProposal(action_name="nonexistent")
        result = pipeline.authorize(proposal)
        assert not result.approved
        assert "unknown action" in result.denied_reason
        db.close()

    def test_permission_denied(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="restricted_action", description="Restricted",
            permissions=["admin.write"], reversible=True,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=False)
        pipeline = AuthorizationPipeline(
            action_registry=registry, permission_checker=checker, db=db,
        )
        proposal = ActionProposal(action_name="restricted_action")
        result = pipeline.authorize(proposal)
        assert not result.approved
        assert "permission denied" in result.denied_reason
        db.close()

    def test_permission_granted(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="safe_action", description="Safe",
            permissions=["filesystem.read"], reversible=True,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=True)
        pipeline = AuthorizationPipeline(
            action_registry=registry, permission_checker=checker, db=db,
        )
        proposal = ActionProposal(action_name="safe_action")
        result = pipeline.authorize(proposal)
        assert result.approved
        db.close()

    def test_irreversible_without_approval_denied(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="delete_file", description="Delete a file",
            permissions=["filesystem.write"], reversible=False,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=True)
        pipeline = AuthorizationPipeline(
            action_registry=registry, permission_checker=checker, db=db,
            allow_irreversible=False,
        )
        proposal = ActionProposal(
            action_name="delete_file", target="/important.txt",
            requires_approval=False,  # don't route to approval queue
        )
        result = pipeline.authorize(proposal)
        assert not result.approved
        assert "irreversible" in result.denied_reason
        db.close()

    def test_irreversible_with_approval_routes_to_queue(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="delete_file", description="Delete a file",
            permissions=["filesystem.write"], reversible=False,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=True)
        pipeline = AuthorizationPipeline(
            action_registry=registry, permission_checker=checker, db=db,
            allow_irreversible=False,
        )
        proposal = ActionProposal(
            action_name="delete_file", target="/important.txt",
            requires_approval=True, reason="User wants to clean up",
            risk_level="high",
        )
        result = pipeline.authorize(proposal, run_id=42)
        assert not result.approved
        assert result.needs_approval is True
        assert result.approval_id is not None

        # Verify approval was submitted to the queue.
        approvals = db.pending_approvals()
        assert len(approvals) == 1
        assert approvals[0]["action_name"] == "delete_file"
        assert approvals[0]["workflow_id"] == 42
        db.close()

    def test_irreversible_allowed_when_flag_set(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="delete_file", description="Delete a file",
            permissions=["filesystem.write"], reversible=False,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=True)
        pipeline = AuthorizationPipeline(
            action_registry=registry, permission_checker=checker, db=db,
            allow_irreversible=True,
        )
        proposal = ActionProposal(action_name="delete_file")
        result = pipeline.authorize(proposal)
        assert result.approved
        db.close()

    def test_dangerous_action_denied(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="dangerous_action", description="Dangerous",
            permissions=[], reversible=True,
        )
        registry = _make_registry_with_actions(action)

        class DangerousTrustcard(TrustcardIntegrationPoint):
            def score_action(self, action_name, target="", parameters=None):
                if action_name == "dangerous_action":
                    return 85.0, ["high_risk"]
                return 0.0, []

        pipeline = AuthorizationPipeline(
            action_registry=registry, db=db,
            trustcard=DangerousTrustcard(danger_threshold=70.0),
        )
        proposal = ActionProposal(action_name="dangerous_action")
        result = pipeline.authorize(proposal)
        assert not result.approved
        assert result.danger_score == 85.0
        assert "high_risk" in result.danger_flags
        assert "dangerous" in result.denied_reason.lower()
        db.close()

    def test_safe_action_with_low_danger_score(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="safe_action", description="Safe",
            permissions=[], reversible=True,
        )
        registry = _make_registry_with_actions(action)

        class LowDangerTrustcard(TrustcardIntegrationPoint):
            def score_action(self, action_name, target="", parameters=None):
                return 20.0, ["low_risk_note"]

        pipeline = AuthorizationPipeline(
            action_registry=registry, db=db,
            trustcard=LowDangerTrustcard(danger_threshold=70.0),
        )
        proposal = ActionProposal(action_name="safe_action")
        result = pipeline.authorize(proposal)
        assert result.approved
        assert result.danger_score == 20.0
        assert "low_risk_note" in result.danger_flags
        db.close()

    def test_execute_authorized_action(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="test_action", description="Test",
            permissions=[], reversible=True,
        )

        def handler(req, ctx):
            return ActionResult(
                ok=True, action=req.action, target=req.target,
                result="executed successfully", executed=True,
            )

        registry = _make_registry_with_handlers(action, handler)
        executor = ActionExecutor(
            registry=registry, permissions=PermissionChecker([]), db=db,
            allow_irreversible=True,
        )
        pipeline = AuthorizationPipeline(
            action_executor=executor, action_registry=registry, db=db,
            allow_irreversible=True,
        )
        proposal = ActionProposal(action_name="test_action")
        result = pipeline.execute(proposal)
        assert result.ok
        assert result.executed
        assert result.result == "executed successfully"
        db.close()

    def test_execute_denied_action(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="restricted", description="Restricted",
            permissions=["admin"], reversible=True,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=False)
        pipeline = AuthorizationPipeline(
            action_registry=registry, permission_checker=checker, db=db,
        )
        proposal = ActionProposal(action_name="restricted")
        result = pipeline.execute(proposal)
        assert not result.ok
        assert "permission denied" in result.error
        db.close()


# --- WorkflowEngine.execute_action_proposal tests ---

class TestEngineActionProposal:
    def test_execute_approved_proposal(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="test_action", description="Test",
            permissions=[], reversible=True,
        )

        def handler(req, ctx):
            return ActionResult(
                ok=True, action=req.action, target=req.target,
                result="done", executed=True,
            )

        registry = _make_registry_with_handlers(action, handler)
        executor = ActionExecutor(
            registry=registry, permissions=PermissionChecker([]), db=db,
            allow_irreversible=True,
        )
        engine = WorkflowEngine(
            db, action_executor=executor, action_registry=registry,
        )

        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="Step", handler=lambda ctx: {"ok": True}))
        result = engine.run(wf)

        proposal = ActionProposal(
            action_name="test_action",
            idempotency_key="test-action-run1",
        )
        outcome = engine.execute_action_proposal(
            proposal, run_id=result.run_id, step=wf.steps[0],
        )
        assert outcome["ok"] is True
        assert outcome["skipped"] is False
        assert outcome["result"].ok is True
        db.close()

    def test_duplicate_proposal_skipped(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="test_action", description="Test",
            permissions=[], reversible=True,
        )

        call_count = {"n": 0}

        def handler(req, ctx):
            call_count["n"] += 1
            return ActionResult(
                ok=True, action=req.action, target=req.target,
                result="done", executed=True,
            )

        registry = _make_registry_with_handlers(action, handler)
        executor = ActionExecutor(
            registry=registry, permissions=PermissionChecker([]), db=db,
            allow_irreversible=True,
        )
        engine = WorkflowEngine(
            db, action_executor=executor, action_registry=registry,
        )

        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="Step", handler=lambda ctx: {"ok": True}))
        result = engine.run(wf)

        key = "unique-key-123"
        proposal = ActionProposal(
            action_name="test_action", idempotency_key=key,
        )

        # First execution.
        outcome1 = engine.execute_action_proposal(
            proposal, run_id=result.run_id, step=wf.steps[0],
        )
        assert outcome1["ok"] is True
        assert outcome1["skipped"] is False
        assert call_count["n"] == 1

        # Second execution with same key — should be skipped.
        outcome2 = engine.execute_action_proposal(
            proposal, run_id=result.run_id, step=wf.steps[0],
        )
        assert outcome2["ok"] is True
        assert outcome2["skipped"] is True
        assert call_count["n"] == 1  # handler not called again
        db.close()

    def test_denied_proposal(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="restricted", description="Restricted",
            permissions=["admin"], reversible=True,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=False)
        engine = WorkflowEngine(
            db, permission_checker=checker, action_registry=registry,
        )

        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="Step", handler=lambda ctx: {"ok": True}))
        result = engine.run(wf)

        proposal = ActionProposal(action_name="restricted")
        outcome = engine.execute_action_proposal(
            proposal, run_id=result.run_id, step=wf.steps[0],
        )
        assert outcome["ok"] is False
        assert outcome["authorization"].approved is False
        assert "permission denied" in outcome["denied_reason"]
        db.close()

    def test_needs_approval_proposal(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="delete_file", description="Delete",
            permissions=["filesystem.write"], reversible=False,
        )
        registry = _make_registry_with_actions(action)
        checker = MagicMock()
        checker.check = MagicMock(return_value=True)
        engine = WorkflowEngine(
            db, permission_checker=checker, action_registry=registry,
        )

        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="Step", handler=lambda ctx: {"ok": True}))
        result = engine.run(wf)

        proposal = ActionProposal(
            action_name="delete_file", target="/important.txt",
            requires_approval=True, risk_level="high",
        )
        outcome = engine.execute_action_proposal(
            proposal, run_id=result.run_id, step=wf.steps[0],
        )
        assert outcome["ok"] is False
        assert outcome["needs_approval"] is True
        assert outcome["approval_id"] is not None

        # Verify approval is in the queue.
        approvals = db.pending_approvals()
        assert len(approvals) == 1
        assert approvals[0]["action_name"] == "delete_file"
        db.close()

    def test_side_effect_recorded_on_success(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="send_email", description="Send email",
            permissions=[], reversible=True,
        )

        def handler(req, ctx):
            return ActionResult(
                ok=True, action=req.action, target=req.target,
                result="email sent", executed=True,
            )

        registry = _make_registry_with_handlers(action, handler)
        executor = ActionExecutor(
            registry=registry, permissions=PermissionChecker([]), db=db,
            allow_irreversible=True,
        )
        engine = WorkflowEngine(
            db, action_executor=executor, action_registry=registry,
        )

        wf = Workflow(name="test", description="Test")
        wf.add_step(WorkflowStep(name="step1", description="Step", handler=lambda ctx: {"ok": True}))
        result = engine.run(wf)

        proposal = ActionProposal(
            action_name="send_email", target="user@example.com",
            idempotency_key="email:daily:2026-08-18",
            side_effect_type="email.send",
        )
        outcome = engine.execute_action_proposal(
            proposal, run_id=result.run_id, step=wf.steps[0],
        )
        assert outcome["ok"] is True

        # Verify side effect was recorded.
        effects = db.side_effects_for_workflow(result.run_id)
        assert len(effects) == 1
        assert effects[0]["effect_type"] == "email.send"
        assert effects[0]["idempotency_key"] == "email:daily:2026-08-18"
        db.close()


# --- Integration: workflow step using action proposal ---

class TestWorkflowWithActions:
    def test_step_executes_action_via_proposal(self, tmp_path):
        """A workflow step handler can use the engine to execute an action."""
        db = Database(str(tmp_path / "test.db"))
        action = Action(
            name="create_task", description="Create a task",
            permissions=[], reversible=True,
        )

        def create_task_handler(req, ctx):
            return ActionResult(
                ok=True, action=req.action, target=req.target,
                result=f"task created: {req.parameters.get('title')}",
                executed=True,
            )

        registry = _make_registry_with_handlers(action, create_task_handler)
        executor = ActionExecutor(
            registry=registry, permissions=PermissionChecker([]), db=db,
            allow_irreversible=True,
        )
        engine = WorkflowEngine(
            db, action_executor=executor, action_registry=registry,
        )

        # The step handler uses the engine to execute an action proposal.
        def step_handler(ctx):
            engine = ctx.get("engine")
            proposal = ActionProposal(
                action_name="create_task",
                parameters={"title": "Buy groceries"},
                idempotency_key="task:buy-groceries:2026-08-18",
            )
            outcome = engine.execute_action_proposal(
                proposal, run_id=ctx.get("run_id"),
            )
            return {"ok": outcome["ok"], "action_result": outcome.get("result", None)}

        wf = Workflow(name="test_action_workflow", description="Test")
        wf.add_step(WorkflowStep(name="create_task_step", description="Creates a task", handler=step_handler))

        # We need to pass the engine and run_id into the context.
        # For this test, we'll run manually.
        run_id = db.start_workflow_run("test_action_workflow", 1)
        ctx = {"engine": engine, "run_id": run_id, "db": db}

        step_result = engine._execute_step(wf, wf.steps[0], 0, ctx, run_id)
        assert step_result.ok
        assert step_result.data["ok"] is True
        db.close()
