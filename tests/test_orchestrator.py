from app.agents.registry import default_agents
from app.conversation import Conversation
from app.db import Database
from app.orchestrator import AgentOrchestrator
from tests.conftest import FakeModel, FakeRetriever


def _build_orchestrator():
    planner = FakeModel()
    retriever = FakeRetriever()
    db = Database(":memory:")
    agents = default_agents(planner=planner, retriever=retriever, db=db)
    return AgentOrchestrator(planner=planner, agents=agents)


def test_route_keyword():
    orch = _build_orchestrator()
    assert orch.route("show me cpu usage") == "system"
    assert orch.route("read this file") == "filesystem"
    assert orch.route("explain this repo's code") == "repository"
    assert orch.route("what changed since yesterday") == "git"
    assert orch.route("explain my last 5 commits") == "git"


def test_orchestrator_run():
    orch = _build_orchestrator()
    result = orch.run("what is my cpu usage?")
    assert result.ok
    assert result.steps[0].startswith("routed to")


def test_conversation_send():
    orch = _build_orchestrator()
    convo = Conversation(orchestrator=orch)
    answer = convo.send("show system status")
    assert isinstance(answer, str) and answer
    assert len(convo.history) == 3  # system + user + assistant
