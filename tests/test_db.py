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
