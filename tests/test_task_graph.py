"""Tests for the task graph, approval queue, and user model."""

from app.db import Database


def test_task_graph_parent_child(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    # Create a parent task.
    parent_id = db.create_task_with_parent("Implement workflow engine", source="ai")
    assert parent_id > 0

    # Create child tasks.
    db.create_task_with_parent("Add workflow schema", parent_id=parent_id, source="ai")
    db.create_task_with_parent("Add executor", parent_id=parent_id, source="ai")
    db.create_task_with_parent("Add tests", parent_id=parent_id, source="ai")

    # Check children.
    children = db.task_children(parent_id)
    assert len(children) == 3

    # Check tree.
    tree = db.task_tree()
    assert len(tree) == 1
    assert tree[0]["id"] == parent_id
    assert len(tree[0]["children"]) == 3
    db.close()


def test_task_dependencies(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    # Create tasks with dependencies.
    task1 = db.create_task_with_parent("First task", source="ai")
    task2 = db.create_task_with_parent("Second task", dependencies=[task1], source="ai")
    task3 = db.create_task_with_parent("Third task", dependencies=[task1, task2], source="ai")

    # task2 is blocked by task1 (not completed).
    assert db.task_is_blocked(task2)

    # Complete task1.
    db.complete_task(task1)
    assert not db.task_is_blocked(task2)  # task2 is now unblocked

    # task3 is still blocked by task2.
    assert db.task_is_blocked(task3)

    # Complete task2.
    db.complete_task(task2)
    assert not db.task_is_blocked(task3)
    db.close()


def test_approval_queue(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    # Submit an approval request.
    approval_id = db.submit_approval(
        "git_commit", target="mac-ai-os",
        parameters='{"message": "test"}',
        reason="3 files changed", risk="medium",
    )
    assert approval_id > 0

    # Check it's pending.
    pending = db.pending_approvals()
    assert len(pending) == 1
    assert pending[0]["action_name"] == "git_commit"
    assert pending[0]["risk"] == "medium"
    assert db.approval_count() == 1

    # Approve it.
    db.decide_approval(approval_id, approved=True)
    assert db.approval_count() == 0  # no longer pending

    approval = db.approval_by_id(approval_id)
    assert approval["status"] == "approved"
    assert approval["decided_by"] == "user"
    db.close()


def test_approval_deny(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    approval_id = db.submit_approval("dangerous_action", risk="high")
    db.decide_approval(approval_id, approved=False)

    approval = db.approval_by_id(approval_id)
    assert approval["status"] == "denied"
    db.close()


def test_user_model(tmp_path):
    db = Database(str(tmp_path / "test.db"))

    # Set a preference.
    db.set_user_model_entry("coding_style", "prefers typed Python", confidence=0.9, source="observed")
    db.set_user_model_entry("workflow", "runs pytest before commit", confidence=0.85, source="observed")

    # Get it back.
    entry = db.get_user_model_entry("coding_style")
    assert entry is not None
    assert entry["value"] == "prefers typed Python"
    assert entry["confidence"] == 0.9

    # List all.
    entries = db.all_user_model_entries()
    assert len(entries) == 2

    # Update existing (upsert).
    db.set_user_model_entry("coding_style", "prefers typed Python with dataclasses", confidence=0.95)
    entry = db.get_user_model_entry("coding_style")
    assert entry["value"] == "prefers typed Python with dataclasses"
    assert entry["confidence"] == 0.95
    assert len(db.all_user_model_entries()) == 2  # still 2, not 3

    # Delete.
    db.delete_user_model_entry("coding_style")
    assert db.get_user_model_entry("coding_style") is None
    db.close()
