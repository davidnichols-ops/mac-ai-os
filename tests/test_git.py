import subprocess
from pathlib import Path

import pytest

from app.agents.git import GitAgent
from app.db import Database
from app.retrieval.indexers.git import GitIndexer
from tests.conftest import FakeModel, FakeRetriever


def _make_repo(tmp_path: Path) -> Path:
    """Create a small git repo with a few commits."""
    repo = tmp_path / "myrepo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@test.com",
           "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@test.com",
           "GIT_AUTHOR_DATE": "2025-01-15T10:00:00",
           "GIT_COMMITTER_DATE": "2025-01-15T10:00:00"}

    def _git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env={**env, "PATH": "/usr/bin:/bin"})

    _git("init")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@test.com")
    (repo / "a.py").write_text("print('a')\n")
    _git("add", ".")
    _git("commit", "-m", "first commit")

    # Second commit with different date.
    env2 = {**env, "GIT_AUTHOR_DATE": "2025-01-16T12:00:00", "GIT_COMMITTER_DATE": "2025-01-16T12:00:00"}
    (repo / "a.py").write_text("print('a changed')\n")
    (repo / "b.py").write_text("print('b')\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, env={**env2, "PATH": "/usr/bin:/bin"})
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "second commit"], check=True, env={**env2, "PATH": "/usr/bin:/bin"})

    return repo


@pytest.fixture
def git_repo(tmp_path):
    return _make_repo(tmp_path)


def test_git_indexer(git_repo, tmp_path):
    db = Database(str(tmp_path / "test.db"))
    indexer = GitIndexer(db)
    chunks = indexer.index(str(git_repo))
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["kind"] == "git_commit"
    assert db.commit_count(str(git_repo.resolve())) == 2
    db.close()


def test_git_agent_structured(git_repo, tmp_path):
    db = Database(str(tmp_path / "test.db"))
    indexer = GitIndexer(db)
    indexer.index(str(git_repo))

    agent = GitAgent(planner=FakeModel(), db=db)
    result = agent.execute("show me the last 2 commits")
    assert result.ok
    assert "first commit" in result.answer or "second commit" in result.answer


def test_git_agent_most_modified(git_repo, tmp_path):
    db = Database(str(tmp_path / "test.db"))
    indexer = GitIndexer(db)
    indexer.index(str(git_repo))

    agent = GitAgent(planner=FakeModel(), db=db)
    result = agent.execute("which files have I modified most?")
    assert result.ok
    assert "a.py" in result.answer


def test_git_agent_rag(git_repo, tmp_path):
    db = Database(str(tmp_path / "test.db"))
    retriever = FakeRetriever()
    indexer = GitIndexer(db)
    chunks = indexer.index(str(git_repo))
    retriever.index(str(git_repo.resolve()), chunks)

    agent = GitAgent(planner=FakeModel(), db=db, retriever=retriever)
    result = agent.execute("explain my recent commits")
    assert result.ok
    db.close()
