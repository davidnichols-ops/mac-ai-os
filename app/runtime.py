"""Runtime assembly — builds the orchestrator + conversation from settings.

Keeps wiring in one place so the CLI and tests can construct a working stack
without duplicating boilerplate. No globals: `build_runtime()` returns a
`Runtime` dataclass holding all the wired objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agents.registry import default_agents
from app.config.settings import Settings, load_settings
from app.db import Database
from app.log import configure_logging
from app.models.base import EmbeddingModel, Model
from app.models.registry import build_embedding_model, build_model
from app.orchestrator import AgentOrchestrator
from app.retrieval.base import Retriever
from app.retrieval.registry import build_retriever
from app.tools.builtin import builtin_tools
from app.tools.registry import ToolRouter


@dataclass
class Runtime:
    settings: Settings
    planner: Model
    utility: Model
    embedding_model: EmbeddingModel
    retriever: Retriever
    tools: ToolRouter
    db: Database
    orchestrator: AgentOrchestrator


def build_runtime(
    *,
    config_path: str | None = None,
    configure_logs: bool = True,
) -> Runtime:
    settings = load_settings(config_path)
    if configure_logs:
        configure_logging(settings)

    planner = build_model("planner", settings)
    utility = build_model("utility", settings)
    embedding_model = build_embedding_model(settings)
    retriever = build_retriever(settings, embedding_model)

    db_path = str(Path(settings.paths.data_dir) / "maos.db")
    db = Database(db_path)

    tools = ToolRouter(builtin_tools(), granted_permissions={
        "shell.exec", "file.read", "file.write", "git.exec",
    })
    agents = default_agents(
        planner=planner, utility=utility, retriever=retriever, tools=tools, db=db
    )
    orchestrator = AgentOrchestrator(planner=planner, agents=agents, tools=tools)

    return Runtime(
        settings=settings,
        planner=planner,
        utility=utility,
        embedding_model=embedding_model,
        retriever=retriever,
        tools=tools,
        db=db,
        orchestrator=orchestrator,
    )
