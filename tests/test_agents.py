from app.agents.filesystem import FilesystemAgent
from app.agents.repository import RepositoryAgent
from app.agents.research import ResearchAgent
from app.agents.system import SystemAgent
from tests.conftest import FakeModel


def test_repository_agent_answers(fake_retriever):
    fake_retriever.index("repo", [
        {"text": "def foo(): return 42", "metadata": {"source": "src/foo.py", "chunk_index": 0}}
    ])
    agent = RepositoryAgent(planner=FakeModel(), retriever=fake_retriever)
    result = agent.execute("what does foo do?")
    assert result.ok
    assert "src/foo.py" in result.data["sources"]


def test_filesystem_agent_no_retriever():
    agent = FilesystemAgent(planner=FakeModel(), retriever=None)
    result = agent.execute("find x")
    assert not result.ok


def test_research_agent_cites_sources(fake_retriever):
    fake_retriever.index("docs", [
        {"text": "Metal is Apple's GPU API.", "metadata": {"source": "metal.md", "chunk_index": 0}}
    ])
    agent = ResearchAgent(planner=FakeModel(), retriever=fake_retriever)
    result = agent.execute("what is Metal?")
    assert result.ok
    assert "metal.md" in result.data["local_sources"]


def test_system_agent_snapshot():
    agent = SystemAgent(planner=None)
    result = agent.execute("status")
    assert result.ok
    assert "macOS" in result.answer
