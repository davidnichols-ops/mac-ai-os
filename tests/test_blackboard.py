"""Tests for the blackboard, attention daemon, and learner."""

import json
from types import SimpleNamespace

from app.blackboard import Blackboard
from app.db import Database
from app.learning import Learner


def test_blackboard_basic():
    bb = Blackboard()
    assert bb.current_goal == ""
    assert bb.confidence == 0.0

    bb.set_goal("Implement workflow engine")
    bb.set_workspace("mac-ai-os")
    bb.set_confidence(0.85)

    assert bb.current_goal == "Implement workflow engine"
    assert bb.goal_source == "manual"
    assert bb.goal_updated_at != ""
    assert bb.current_workspace == "mac-ai-os"
    assert bb.confidence == 0.85
    assert bb.last_updated != ""


def test_blackboard_problems():
    bb = Blackboard()
    bb.add_problem("Memory pressure", evidence="85% used")
    bb.add_problem("Blocked task", evidence="depends on #3")

    assert len(bb.open_problems) == 2
    assert bb.open_problems[0]["problem"] == "Memory pressure"
    assert bb.open_problems[0]["evidence"] == "85% used"

    bb.resolve_problem(0)
    assert len(bb.open_problems) == 1
    assert bb.open_problems[0]["problem"] == "Blocked task"


def test_blackboard_decisions():
    bb = Blackboard()
    for i in range(15):
        bb.add_decision("router", f"decision_{i}", confidence=0.9)

    # Should keep only last 10.
    assert len(bb.recent_decisions) == 10
    assert bb.recent_decisions[-1]["decision"] == "decision_14"


def test_blackboard_suggestions():
    bb = Blackboard()
    for i in range(7):
        bb.add_suggestion(f"action_{i}", reason="test")

    # Should keep only last 5.
    assert len(bb.suggested_actions) == 5


def test_blackboard_to_text():
    bb = Blackboard()
    bb.set_goal("Test goal")
    bb.set_workspace("test_ws")
    bb.set_confidence(0.9)
    bb.add_problem("Test problem")
    bb.add_suggestion("Test suggestion", reason="because")
    bb.add_decision("router", "routed to session agent", confidence=0.8)

    text = bb.to_text()
    assert "Test goal" in text
    assert "test_ws" in text
    assert "Test problem" in text
    assert "Test suggestion" in text
    assert "session agent" in text


def test_blackboard_persist_restore(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bb = Blackboard()
    bb.set_goal("Persistent goal")
    bb.set_workspace("persistent_ws")
    bb.set_confidence(0.95)
    bb.add_problem("Persistent problem")
    bb.persist(db)

    # Create a new blackboard and restore.
    bb2 = Blackboard()
    restored = bb2.restore(db)
    assert restored
    assert bb2.current_goal == "Persistent goal"
    assert bb2.goal_source == "manual"
    assert bb2.goal_updated_at != ""
    assert bb2.current_workspace == "persistent_ws"
    assert bb2.confidence == 0.95
    assert len(bb2.open_problems) == 1
    db.close()


def test_blackboard_restores_legacy_goal_provenance(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.set_user_model_entry(
        "_blackboard_state",
        json.dumps({"current_goal": "Legacy goal", "last_updated": "2026-08-20T00:00:00+00:00"}),
        confidence=1.0,
        source="system",
    )
    bb = Blackboard()

    assert bb.restore(db)
    assert bb.goal_source == "legacy"
    assert bb.goal_updated_at == "2026-08-20T00:00:00+00:00"
    db.close()


def test_blackboard_clear():
    bb = Blackboard()
    bb.set_goal("Temp")
    bb.set_workspace("temp_ws")
    bb.add_problem("Temp problem")
    bb.clear()

    assert bb.current_goal == ""
    assert bb.goal_source == ""
    assert bb.goal_updated_at == ""
    assert bb.current_workspace == ""
    assert len(bb.open_problems) == 0
    assert bb.confidence == 0.0


def test_blackboard_thread_safe():
    """Test that the blackboard can be safely accessed from multiple threads."""
    import threading

    bb = Blackboard()
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(100):
                bb.add_decision("test", f"d{i}", confidence=0.5)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(100):
                s = bb.snapshot()
                assert "recent_decisions" in s
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0
    assert len(bb.recent_decisions) <= 10


def test_attention_cycle(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    bb = Blackboard()
    daemon = AttentionDaemon(db, bb, interval_seconds=1.0)

    context = daemon.cycle()

    # Context always has health, problems, and suggestions.
    assert "health" in context
    assert "open_problems" in context
    assert "suggestions" in context
    assert bb.last_updated != ""
    assert bb.confidence >= 0
    db.close()


def test_attention_detects_workspace(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    bb = Blackboard()
    daemon = AttentionDaemon(db, bb)

    # No workspaces in a fresh DB.
    ws = daemon._detect_workspace()
    # Could be empty or a detected workspace depending on environment.
    assert isinstance(ws, str)
    db.close()


def test_attention_identifies_blocked_tasks(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    bb = Blackboard()
    daemon = AttentionDaemon(db, bb)

    # Create a blocked task.
    task1 = db.create_task_with_parent("First task", source="ai")
    db.create_task_with_parent("Second task", dependencies=[task1], source="ai")

    problems = daemon._identify_problems(health={"status": "normal"})
    blocked = [p for p in problems if "blocked" in p["problem"].lower()]
    assert len(blocked) >= 1
    db.close()


def test_attention_generates_suggestions(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    bb = Blackboard()
    daemon = AttentionDaemon(db, bb)

    # Add a pending task.
    db.create_task_with_parent("Important task", source="ai")

    suggestions = daemon._generate_suggestions(workspace="", health={"status": "normal"})
    assert len(suggestions) > 0
    # Should suggest continuing the pending task.
    assert any("Important task" in s["action"] for s in suggestions)
    db.close()


def test_attention_preserves_explicit_goal(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    db.create_task("Stale high-priority task", source="ai", priority=100)
    bb = Blackboard()
    bb.set_goal("User-selected goal")
    daemon = AttentionDaemon(db, bb)

    daemon.cycle()

    assert bb.current_goal == "User-selected goal"
    assert bb.goal_source == "manual"
    db.close()


def test_attention_prefers_workspace_task_over_global_and_inbox(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    db.create_task("Global task", source="ai", priority=100)
    db.create_task("GitHub: unrelated issue", source="github", priority=200)
    db.create_task(
        "Fix current workspace",
        source="ai",
        related_workspace="/Users/testuser/mac-ai-os",
        priority=5,
    )
    daemon = AttentionDaemon(db, Blackboard())

    task = daemon._focus_task("mac-ai-os")

    assert task is not None
    assert task["title"] == "Fix current workspace"
    db.close()


def test_attention_deduplicates_repeated_denials(tmp_path):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    for _ in range(3):
        db.log_action(
            "complete_todo",
            approved=False,
            executed=False,
            error="irreversible action requires explicit confirmation",
        )
    daemon = AttentionDaemon(db, Blackboard())

    problems = daemon._identify_problems({"status": "normal"})
    denials = [problem for problem in problems if "complete_todo" in problem["problem"]]

    assert len(denials) == 1
    db.close()


def test_attention_reports_disk_pressure_without_telemetry(tmp_path, monkeypatch):
    from app.attention import AttentionDaemon

    db = Database(str(tmp_path / "test.db"))
    daemon = AttentionDaemon(db, Blackboard())
    monkeypatch.setattr(
        "app.attention.shutil.disk_usage",
        lambda _: SimpleNamespace(total=100, used=92, free=8),
    )

    health = daemon._check_health()
    problems = daemon._identify_problems(health)
    suggestions = daemon._generate_suggestions(workspace="", health=health)

    assert health["status"] == "disk_pressure"
    assert health["disk_pct"] == 92.0
    assert any("Disk pressure" in problem["problem"] for problem in problems)
    assert any("disk pressure" in suggestion["action"].lower() for suggestion in suggestions)
    db.close()


def test_learner_record_action_success(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    learner = Learner(db)

    learner.record_action_outcome("open_workspace", "mac-ai-os", success=True)
    learner.record_action_outcome("open_workspace", "other", success=True)
    learner.record_action_outcome("open_workspace", "fail", success=False, error="not found")

    stats = learner.action_success_rate("open_workspace")
    assert stats is not None
    assert stats["total"] == 3
    assert stats["successes"] == 2
    assert stats["rate"] == 2 / 3
    db.close()


def test_learner_records_errors(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    learner = Learner(db)

    learner.record_action_outcome("git_commit", "repo", success=False, error="nothing to commit")
    errors = learner.common_errors("git_commit")
    assert "nothing to commit" in errors
    db.close()


def test_learner_record_preference(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    learner = Learner(db)

    # First observation.
    learner.record_user_preference("coding_style", "typed Python", confidence=0.6)
    entry = db.get_user_model_entry("coding_style")
    assert entry["value"] == "typed Python"
    assert entry["confidence"] == 0.6

    # Second observation confirms — confidence should increase.
    learner.record_user_preference("coding_style", "typed Python")
    entry = db.get_user_model_entry("coding_style")
    assert entry["confidence"] > 0.6

    # Different observation — replace with lower confidence.
    learner.record_user_preference("coding_style", "functional Python", confidence=0.5)
    entry = db.get_user_model_entry("coding_style")
    assert entry["value"] == "functional Python"
    db.close()


def test_learner_record_workflow_outcome(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    learner = Learner(db)

    learner.record_workflow_outcome("morning_workspace_restore", True, 6, 6)
    learner.record_workflow_outcome("morning_workspace_restore", False, 3, 6)

    entries = db.all_user_model_entries()
    wf_entry = next(e for e in entries if e["key"] == "workflow.morning_workspace_restore.success_rate")
    parts = wf_entry["value"].split("/")
    assert parts[0] == "1"  # successes
    assert parts[1] == "2"  # total
    db.close()


def test_learner_summary(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    learner = Learner(db)

    learner.record_action_outcome("open_workspace", "ws", success=True)
    learner.record_workflow_outcome("morning", True, 6, 6)
    learner.record_user_preference("editor", "vim", confidence=0.8)

    summary = learner.learned_summary()
    assert len(summary["actions"]) == 1
    assert len(summary["workflows"]) == 1
    assert len(summary["preferences"]) >= 1
    assert summary["total_entries"] >= 3
    db.close()
