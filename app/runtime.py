"""Runtime assembly — builds the orchestrator + conversation from settings.

Keeps wiring in one place so the CLI and tests can construct a working stack
without duplicating boilerplate. No globals: `build_runtime()` returns a
`Runtime` dataclass holding all the wired objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.actions import (
    ActionExecutor,
    ActionRegistry,
    PermissionChecker,
    builtin_actions,
)
from app.agents.registry import default_agents
from app.attention import AttentionDaemon
from app.blackboard import Blackboard
from app.config.settings import Settings, load_settings
from app.db import Database
from app.events.bus import EventBus
from app.knowledge.graph import KnowledgeGraph
from app.learning import Learner
from app.log import configure_logging
from app.models.base import EmbeddingModel, Model
from app.models.manager import ModelManager
from app.models.registry import build_embedding_model, build_model
from app.orchestrator import AgentOrchestrator
from app.retrieval.base import Retriever
from app.retrieval.registry import build_retriever
from app.router import IntentRouter
from app.tools.builtin import builtin_tools
from app.tools.registry import ToolRouter
from app.workflows.engine import WorkflowEngine


@dataclass
class Runtime:
    settings: Settings
    planner: Model
    utility: Model
    embedding_model: EmbeddingModel
    retriever: Retriever
    tools: ToolRouter
    db: Database
    bus: EventBus
    graph: KnowledgeGraph
    orchestrator: AgentOrchestrator
    action_registry: ActionRegistry
    action_executor: ActionExecutor
    permissions: PermissionChecker
    router: IntentRouter
    model_manager: ModelManager
    workflow_engine: WorkflowEngine
    blackboard: Blackboard
    attention: AttentionDaemon
    learner: Learner


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

    # Event bus — persists all events to the events table.
    bus = EventBus(db=db)

    # Knowledge graph — uses the planner for entity extraction.
    graph = KnowledgeGraph(db, planner=planner)

    tools = ToolRouter(builtin_tools(), granted_permissions={
        "shell.exec", "file.read", "file.write", "git.exec",
    })
    agents = default_agents(
        planner=planner, utility=utility, retriever=retriever, tools=tools, db=db
    )
    orchestrator = AgentOrchestrator(planner=planner, agents=agents, tools=tools)

    # Action layer — registry, permissions, executor.
    action_registry = ActionRegistry()
    for action, handler in builtin_actions():
        action_registry.register(action, handler)

    permissions = PermissionChecker(granted={
        "filesystem.read", "application.launch", "tasks.read", "tasks.write",
        "git.exec",
    })

    action_executor = ActionExecutor(
        action_registry, permissions, db, bus=bus,
    )

    # Intent router — uses utility model for classification.
    router = IntentRouter(utility=utility, db=db)

    # Model lifecycle manager.
    model_manager = ModelManager(max_memory_gb=12.0)
    model_manager.register("planner", planner, estimated_memory_gb=9.0)
    model_manager.register("utility", utility, estimated_memory_gb=2.0)

    # Workflow engine.
    workflow_engine = WorkflowEngine(db, bus=bus)

    # Blackboard — shared working memory.
    blackboard = Blackboard()
    blackboard.restore(db)

    # Attention daemon — proactive context computation.
    attention = AttentionDaemon(db, blackboard, interval_seconds=60.0)

    # Learner — records outcomes and updates procedural memory.
    learner = Learner(db)

    return Runtime(
        settings=settings,
        planner=planner,
        utility=utility,
        embedding_model=embedding_model,
        retriever=retriever,
        tools=tools,
        db=db,
        bus=bus,
        graph=graph,
        orchestrator=orchestrator,
        action_registry=action_registry,
        action_executor=action_executor,
        permissions=permissions,
        router=router,
        model_manager=model_manager,
        workflow_engine=workflow_engine,
        blackboard=blackboard,
        attention=attention,
        learner=learner,
    )
