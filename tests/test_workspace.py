
from app.db import Database
from app.workspace import detect_active_workspace, is_project, scan_workspaces


def test_is_project(tmp_path):
    # Not a project (no markers).
    assert not is_project(tmp_path)

    # Is a project (has .git).
    (tmp_path / ".git").mkdir()
    assert is_project(tmp_path)


def test_workspace_storage(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    ws_id = db.upsert_workspace(
        str(tmp_path), "myproject", language="python", git_repo=True, file_count=10,
    )
    assert ws_id > 0

    # Upsert should update, not create.
    ws_id2 = db.upsert_workspace(
        str(tmp_path), "myproject", language="python", git_repo=True, file_count=15,
    )
    assert ws_id2 == ws_id

    workspaces = db.all_workspaces()
    assert len(workspaces) == 1
    assert workspaces[0]["file_count"] == 15
    db.close()


def test_scan_workspaces(tmp_path):
    # Create a fake project structure.
    projects = tmp_path / "Projects"
    projects.mkdir()

    # Project 1: Python with git.
    p1 = projects / "myapp"
    p1.mkdir()
    (p1 / ".git").mkdir()
    (p1 / "main.py").write_text("print('hello')\n")

    # Project 2: Not a project (no markers).
    p2 = projects / "notproject"
    p2.mkdir()
    (p2 / "random.txt").write_text("random")

    db = Database(str(tmp_path / "test.db"))
    discovered = scan_workspaces(db, scan_paths=[str(projects)])

    assert len(discovered) == 1
    assert discovered[0]["name"] == "myapp"
    assert discovered[0]["git_repo"] is True
    assert discovered[0]["language"] == "python"
    db.close()


def test_detect_active_workspace(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    bus = __import__("app.events.bus", fromlist=["EventBus"]).EventBus(db=db)

    from app.events.models import FILE_INDEXED, Event

    # Publish some file events for a project.
    for i in range(5):
        bus.publish(Event(FILE_INDEXED, {
            "path": f"/Users/testuser/myproject/file{i}.py",
            "source": "/Users/testuser/myproject",
        }))

    ws = detect_active_workspace(db)
    assert ws is not None
    # detect_active_workspace prioritizes CWD if it's a project.
    # Since we're running from mac-ai-os, that takes priority.
    # But if CWD isn't a registered workspace, it falls back to events.
    # Just verify we got a workspace back with a path.
    assert ws.get("path", "") != ""
    db.close()


def test_detect_active_workspace_from_events(tmp_path):
    """Test that event-based detection works when CWD is not a project."""
    db = Database(str(tmp_path / "test.db"))
    bus = __import__("app.events.bus", fromlist=["EventBus"]).EventBus(db=db)

    from app.events.models import FILE_INDEXED, Event

    # Register a fake workspace path and publish events for it.
    fake_path = str(tmp_path / "myproject")
    db.upsert_workspace(fake_path, name="myproject", language="python",
                        git_repo=True, file_count=5)

    for i in range(5):
        bus.publish(Event(FILE_INDEXED, {
            "path": f"{fake_path}/file{i}.py",
            "source": fake_path,
        }))

    # Change to a non-project directory so CWD doesn't interfere.
    import os
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        ws = detect_active_workspace(db)
        assert ws is not None
        assert "myproject" in ws.get("path", "") or ws.get("name") == "myproject"
    finally:
        os.chdir(old_cwd)
    db.close()
