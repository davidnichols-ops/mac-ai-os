"""Agent registry — maps agent names to factories and routes goals.

The orchestrator asks the registry to build agents with the right
dependencies injected (planner, retriever, tools, db).
"""

from __future__ import annotations

from app.agents.base import Agent
from app.agents.filesystem import FilesystemAgent
from app.agents.general import GeneralAgent
from app.agents.git import GitAgent
from app.agents.repository import RepositoryAgent
from app.agents.research import ResearchAgent
from app.agents.session import SessionAgent
from app.agents.system import SystemAgent
from app.db import Database
from app.models.base import Model
from app.perception import Perception
from app.retrieval.base import Retriever
from app.tools.registry import ToolRouter


def default_agents(
    *,
    planner: Model | None = None,
    utility: Model | None = None,
    retriever: Retriever | None = None,
    tools: ToolRouter | None = None,
    db: Database | None = None,
    perception: Perception | None = None,
) -> dict[str, Agent]:
    return {
        GeneralAgent.name: GeneralAgent(planner=planner, utility=utility),
        RepositoryAgent.name: RepositoryAgent(planner=planner, utility=utility, retriever=retriever, tools=tools),
        FilesystemAgent.name: FilesystemAgent(planner=planner, utility=utility, retriever=retriever, tools=tools),
        ResearchAgent.name: ResearchAgent(planner=planner, utility=utility, retriever=retriever, tools=tools, perception=perception),
        SystemAgent.name: SystemAgent(planner=planner, utility=utility, retriever=retriever, tools=tools),
        GitAgent.name: GitAgent(planner=planner, db=db, retriever=retriever),
        SessionAgent.name: SessionAgent(planner=planner, db=db, retriever=retriever),
    }


def register_agent(name: str, factory) -> None:
    # Future: external agent plugins
    pass
