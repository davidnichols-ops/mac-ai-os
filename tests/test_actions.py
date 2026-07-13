"""Tests for the action layer: registry, permissions, executor, built-in actions."""

from app.actions import (
    Action,
    ActionExecutor,
    ActionRegistry,
    ActionRequest,
    ActionResult,
    PermissionChecker,
    builtin_actions,
)
from app.db import Database
from app.events.bus import EventBus


def _ok_handler(req, ctx):
    return ActionResult(ok=True, action="test_action", target=req.target)


def _ok_handler_named(action_name):
    def handler(req, ctx):
        return ActionResult(ok=True, action=action_name, target=req.target)
    return handler


def _done_handler(req, ctx):
    return ActionResult(ok=True, action="safe_action", target=req.target, result="done")


def test_action_registry():
    registry = ActionRegistry()
    action = Action(
        name="test_action",
        description="A test action",
        permissions=["test.perm"],
        reversible=True,
    )
    registry.register(action, _ok_handler)

    assert registry.exists("test_action")
    assert registry.get("test_action").name == "test_action"
    assert len(registry.list_actions()) == 1
    assert registry.names() == ["test_action"]


def test_permission_checker():
    checker = PermissionChecker(granted={"a.b", "c.d"})

    assert checker.has("a.b")
    assert checker.has("c.d")
    assert not checker.has("e.f")

    assert checker.check(["a.b"])
    assert checker.check(["a.b", "c.d"])
    assert not checker.check(["a.b", "e.f"])

    # Wildcard grants everything.
    checker.grant("*")
    assert checker.has("anything")


def test_permission_denied(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    registry = ActionRegistry()
    action = Action(
        name="dangerous",
        description="Needs a permission we don't have",
        permissions=["admin.root"],
        reversible=True,
    )
    handler = _ok_handler_named("dangerous")
    registry.register(action, handler)

    permissions = PermissionChecker(granted={"basic.access"})
    executor = ActionExecutor(registry, permissions, db, bus=bus)

    result = executor.execute(ActionRequest(action="dangerous", target="x"))
    assert not result.ok
    assert not result.approved
    assert "permission denied" in result.error

    # Check it was logged.
    actions = db.recent_actions()
    assert len(actions) == 1
    assert actions[0]["action_name"] == "dangerous"
    assert actions[0]["approved"] == 0
    db.close()


def test_action_executed(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    registry = ActionRegistry()
    action = Action(
        name="safe_action",
        description="A safe action",
        permissions=["basic.access"],
        reversible=True,
    )
    registry.register(action, _done_handler)

    permissions = PermissionChecker(granted={"basic.access"})
    executor = ActionExecutor(registry, permissions, db, bus=bus)

    result = executor.execute(ActionRequest(action="safe_action", target="test"))
    assert result.ok
    assert result.approved
    assert result.executed
    assert result.result == "done"

    # Check it was logged.
    actions = db.recent_actions()
    assert len(actions) == 1
    assert actions[0]["action_name"] == "safe_action"
    assert actions[0]["approved"] == 1
    assert actions[0]["executed"] == 1

    # Check event was published.
    events = db.query_events(limit=10)
    assert any(e["type"] == "ACTION_EXECUTED" for e in events)
    db.close()


def test_unknown_action(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    registry = ActionRegistry()
    permissions = PermissionChecker(granted={"*"})
    executor = ActionExecutor(registry, permissions, db)

    result = executor.execute(ActionRequest(action="nonexistent", target="x"))
    assert not result.ok
    assert "unknown action" in result.error
    db.close()


def test_irreversible_action_blocked(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    registry = ActionRegistry()
    action = Action(
        name="irreversible",
        description="Cannot be undone",
        permissions=["basic.perm"],
        reversible=False,
    )
    handler = _ok_handler_named("irreversible")
    registry.register(action, handler)

    permissions = PermissionChecker(granted={"basic.perm"})
    executor = ActionExecutor(registry, permissions, db, allow_irreversible=False)

    result = executor.execute(ActionRequest(action="irreversible", target="x"))
    assert not result.ok
    assert "irreversible" in result.error
    assert result.approved  # permissions were OK
    assert not result.executed  # but execution was blocked
    db.close()


def test_builtin_actions_registered():
    actions = builtin_actions()
    assert len(actions) >= 5

    names = [a.name for a, _ in actions]
    assert "open_workspace" in names
    assert "create_todo" in names
    assert "complete_todo" in names
    assert "workspace_summary" in names
    assert "list_tasks" in names


def test_create_todo_action(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = EventBus(db=db)
    registry = ActionRegistry()
    for action, handler in builtin_actions():
        registry.register(action, handler)

    permissions = PermissionChecker(granted={"tasks.write"})
    executor = ActionExecutor(registry, permissions, db, bus=bus)

    result = executor.execute(
        ActionRequest(action="create_todo", parameters={"title": "Test the action layer"}),
        context={"db": db},
    )
    assert result.ok
    assert "created task" in result.result

    tasks = db.pending_tasks()
    assert len(tasks) == 1
    assert tasks[0]["title"] == "Test the action layer"
    assert tasks[0]["source"] == "ai"
    db.close()


def test_list_tasks_action(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_task("Task 1")
    db.create_task("Task 2")

    registry = ActionRegistry()
    for action, handler in builtin_actions():
        registry.register(action, handler)

    permissions = PermissionChecker(granted={"tasks.read"})
    executor = ActionExecutor(registry, permissions, db)

    result = executor.execute(
        ActionRequest(action="list_tasks"),
        context={"db": db},
    )
    assert result.ok
    assert "Task 1" in result.result
    assert "Task 2" in result.result
    db.close()


def test_decision_log(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    decision_id = db.log_decision(
        agent="IntentRouter",
        decision="route to session",
        reasoning="query matched session keywords",
        confidence=0.85,
        query="what was I working on?",
    )
    assert decision_id > 0

    decisions = db.recent_decisions()
    assert len(decisions) == 1
    assert decisions[0]["agent"] == "IntentRouter"
    assert decisions[0]["confidence"] == 0.85
    assert decisions[0]["query"] == "what was I working on?"

    db.update_decision_outcome(decision_id, "success")
    decisions = db.recent_decisions()
    assert decisions[0]["outcome"] == "success"
    db.close()
