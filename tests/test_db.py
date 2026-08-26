from app.db import Database


def test_file_hashes(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    assert db.get_file_hash("/a/b.py", "src1") is None
    db.set_file_hash("/a/b.py", "src1", "abc123", 5)
    assert db.get_file_hash("/a/b.py", "src1") == "abc123"
    # Update existing.
    db.set_file_hash("/a/b.py", "src1", "def456", 3)
    assert db.get_file_hash("/a/b.py", "src1") == "def456"
    assert db.file_hash_count("src1") == 1
    db.remove_file_hash("/a/b.py", "src1")
    assert db.get_file_hash("/a/b.py", "src1") is None
    db.close()


def test_git_commits(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.upsert_commit(
        repo="/repo1", hash="abc123", author="alice", author_date="2025-01-15T10:00:00",
        message="fix bug", files_changed="src/a.py\nsrc/b.py", insertions=10, deletions=2,
    )
    db.upsert_commit(
        repo="/repo1", hash="def456", author="bob", author_date="2025-01-16T12:00:00",
        message="add feature", files_changed="src/a.py\nsrc/c.py", insertions=20, deletions=5,
    )
    assert db.commit_count("/repo1") == 2

    recent = db.recent_commits("/repo1", limit=1)
    assert len(recent) == 1
    assert recent[0]["hash"] == "def456"

    since = db.commits_since("/repo1", "2025-01-16T00:00:00")
    assert len(since) == 1
    assert since[0]["hash"] == "def456"

    files = db.most_modified_files("/repo1")
    assert files[0][0] == "src/a.py"
    assert files[0][1] == 2
    db.close()


def test_retention_prunes_old_rows(tmp_path):
    """apply_retention deletes rows older than the cutoff but keeps recent ones."""
    from datetime import UTC, datetime, timedelta

    db = Database(str(tmp_path / "test.db"))
    now = datetime.now(UTC)
    old = now - timedelta(days=60)
    recent = now - timedelta(days=5)

    # Insert old + recent events (schema: type, payload, timestamp).
    db.execute(
        "INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)",
        ("TEST", "{}", old.isoformat()),
    )
    db.execute(
        "INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)",
        ("TEST", "{}", recent.isoformat()),
    )
    # Insert old + recent shell commands (schema: timestamp, command, directory, exit_code, session_id).
    db.execute(
        "INSERT INTO shell_commands (timestamp, command, directory, exit_code, session_id) VALUES (?, ?, ?, ?, ?)",
        (old.isoformat(), "old_cmd", "/tmp", 0, None),
    )
    db.execute(
        "INSERT INTO shell_commands (timestamp, command, directory, exit_code, session_id) VALUES (?, ?, ?, ?, ?)",
        (recent.isoformat(), "new_cmd", "/tmp", 0, None),
    )
    # Insert old telemetry + process snapshot (child table).
    cur = db.execute(
        "INSERT INTO telemetry_snapshots (timestamp, cpu_percent, cpu_cores, memory_total_gb, memory_used_gb, memory_percent, swap_used_gb, swap_total_gb, disk_read_mb, disk_write_mb, battery_percent, battery_plugged, thermal_state, process_count, gpu_usage, neural_engine) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (old.isoformat(), 50.0, 10, 16.0, 8.0, 50.0, 1.0, 4.0, 10.0, 5.0, 80, 1, "ok", 100, None, None),
    )
    old_tel_id = cur.lastrowid
    db.execute(
        "INSERT INTO process_snapshots (telemetry_id, pid, name, cpu_percent, memory_mb) VALUES (?, ?, ?, ?, ?)",
        (old_tel_id, 1234, "test", 5.0, 100.0),
    )

    # file_hashes should NOT be pruned (canonical table).
    db.set_file_hash("/keep.py", "src", "hash123", 3)

    # Apply 30-day retention.
    deleted = db.apply_retention(days=30)

    assert deleted["events"] == 1
    assert deleted["shell_commands"] == 1
    assert deleted["telemetry_snapshots"] == 1
    assert deleted["process_snapshots"] == 1

    # Recent rows survive.
    assert db.query_one("SELECT COUNT(*) as n FROM events WHERE type='TEST'")["n"] == 1
    assert db.query_one("SELECT COUNT(*) as n FROM shell_commands")["n"] == 1

    # Canonical table untouched.
    assert db.get_file_hash("/keep.py", "src") == "hash123"

    # DB size is reported.
    assert db.db_size_mb() >= 0.0
    db.close()


def test_retention_zero_days_keeps_nothing(tmp_path):
    """With days=0, all rows with timestamps in the past are pruned."""
    from datetime import UTC, datetime, timedelta

    db = Database(str(tmp_path / "test.db"))
    old = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    db.execute(
        "INSERT INTO events (type, payload, timestamp) VALUES (?, ?, ?)",
        ("TEST", "{}", old),
    )
    deleted = db.apply_retention(days=0)
    assert deleted["events"] == 1
    assert db.query_one("SELECT COUNT(*) as n FROM events")["n"] == 0
    db.close()
