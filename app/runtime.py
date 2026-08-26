"""Runtime assembly — builds the orchestrator + conversation from settings.

Keeps wiring in one place so the CLI and tests can construct a working stack
without duplicating boilerplate. No globals: `build_runtime()` returns a
`Runtime` dataclass holding all the wired objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.actions import (
    ActionExecutor,
    ActionRegistry,
    PermissionChecker,
    builtin_actions,
)
from app.adaptive_router import AdaptiveRouter
from app.agents.registry import default_agents
from app.attention import AttentionDaemon
from app.blackboard import Blackboard
from app.config.settings import Settings, load_settings
from app.db import Database
from app.events.bus import EventBus
from app.knowledge.graph import KnowledgeGraph
from app.learning import Learner
from app.log import configure_logging, get_logger
from app.memory.service import PersonalMemoryService
from app.models.api_provider import APIModel
from app.models.base import EmbeddingModel, Model
from app.models.manager import ModelManager
from app.models.registry import _resolve_api_key, build_embedding_model, build_model
from app.network import NetworkProvider
from app.orchestrator import AgentOrchestrator
from app.persistence.lan_vault import LanVaultClient
from app.retrieval.base import Retriever
from app.retrieval.registry import build_retriever
from app.router import IntentRouter
from app.tools.builtin import builtin_tools
from app.tools.registry import ToolRouter
from app.workflows.engine import WorkflowEngine

if TYPE_CHECKING:
    from app.slack.bridge import SlackBridge

log = get_logger(__name__)


@dataclass
class Runtime:
    settings: Settings
    planner: Model
    planner_alt: Model | None
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
    network: NetworkProvider
    memory: PersonalMemoryService
    vault: LanVaultClient
    qa_model: Model | None = None
    slack: SlackBridge | None = None
    adaptive_router: AdaptiveRouter | None = None


def build_runtime(
    *,
    config_path: str | None = None,
    configure_logs: bool = True,
) -> Runtime:
    settings = load_settings(config_path)
    if configure_logs:
        configure_logging(settings)

    planner = build_model("planner", settings)
    # Alt planner: second model for switching via --model flag.
    # Only built if configured (planner_alt.model is non-empty).
    planner_alt: Model | None = None
    if settings.planner_alt.model:
        try:
            planner_alt = build_model("planner_alt", settings)
        except Exception as exc:
            log.warning("runtime.planner_alt.build_failed", error=str(exc))
    # Utility model: lazy-load to avoid 2GB memory hit at startup.
    # The adaptive router uses sklearn, not an LLM — utility is only
    # needed for IntentRouter classification, which is rare.
    utility = _LazyUtility(settings)
    embedding_model = build_embedding_model(settings)
    retriever = build_retriever(settings, embedding_model)

    data_dir = settings.resolved_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(data_dir / "maos.db")
    db = Database(db_path)

    # Event bus — persists all events to the events table.
    bus = EventBus(db=db)

    # Knowledge graph — uses the planner for entity extraction.
    graph = KnowledgeGraph(db, planner=planner)

    tools = ToolRouter(builtin_tools(), granted_permissions={
        "shell.exec", "file.read", "file.write", "git.exec",
        "web.search", "web.fetch", "todo.write", "user.interact",
    })
    # Note: file_edit and file_list use file.write and file.read permissions
    # which are already granted above.
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
        "git.exec", "memory.write",
    })

    action_executor = ActionExecutor(
        action_registry, permissions, db, bus=bus,
    )

    # Intent router — uses utility model for classification.
    router = IntentRouter(utility=utility, db=db)

    # Model lifecycle manager.
    model_manager = ModelManager(max_memory_gb=12.0)
    model_manager.register("planner", planner, estimated_memory_gb=9.0)
    # Utility model is lazy — don't register it until it's actually loaded

    # Workflow engine.
    workflow_engine = WorkflowEngine(db, bus=bus)

    # Blackboard — shared working memory.
    blackboard = Blackboard()
    blackboard.restore(db)

    # Attention daemon — proactive context computation.
    attention = AttentionDaemon(db, blackboard, interval_seconds=60.0)

    # Learner — records outcomes and updates procedural memory.
    learner = Learner(db)

    # Network intelligence provider — read-only view over network_samples.
    # The collector is NOT started here (it's a background sampler started
    # explicitly via `maos network collect` or by a daemon). The provider
    # just reads what's already in the DB.
    network = NetworkProvider(db)

    # Durable personal memory: local SQLite + RAG are authoritative. The LAN
    # vault is a verified secondary archive and may be unavailable without
    # blocking local capture.
    repo_root = Path(__file__).resolve().parents[1]
    vault = LanVaultClient(settings.lan_vault, base_dir=repo_root)
    memory = PersonalMemoryService(
        db,
        retriever,
        bus=bus,
        vault=vault,
        notes_dir=settings.resolved_data_dir() / "personal-memory",
    )

    # Slack bridge — NOT constructed here. It needs the full runtime, so
    # it's built lazily in the CLI (`maos slack start`) or MCP layer.
    # The config is available via settings.slack.

    # Adaptive router — FAST/THINK mode selection.
    # For v2 models with control tokens, the model self-selects mode.
    # The router is disabled by default; enable via config if using v1.
    adaptive_router = None  # v2: model self-selects via control tokens

    # Optional Slack Q&A model — man-in-the-middle for Slack `ask`.
    qa_model: Model | None = None
    slack_cfg = settings.slack
    if slack_cfg.qa_model and slack_cfg.qa_api_key:
        try:
            qa_model = APIModel(
                slack_cfg.qa_model,
                base_url=slack_cfg.qa_base_url,
                api_key=_resolve_api_key(slack_cfg.qa_api_key),
                temperature=0.6,
                max_tokens=2048,
            )
            log.info("runtime.qa_model.built", model=slack_cfg.qa_model)
        except Exception as exc:
            log.warning("runtime.qa_model.build_failed", error=str(exc))

    return Runtime(
        settings=settings,
        planner=planner,
        planner_alt=planner_alt,
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
        network=network,
        memory=memory,
        vault=vault,
        qa_model=qa_model,
        slack=None,
        adaptive_router=adaptive_router,
    )


class _LazyUtility:
    """Lazy proxy for the utility model — only loads on first use.

    The utility model (Llama 3.2 3B 4-bit, ~2GB) is only needed for
    IntentRouter classification. The adaptive router uses sklearn, not
    an LLM. Deferring the load saves 2GB of memory at startup.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._inner: Model | None = None

    def _resolve(self) -> Model:
        if self._inner is None:
            self._inner = build_model("utility", self._settings)
        return self._inner

    @property
    def name(self) -> str:
        return self._settings.utility.model if self._inner is None else self._inner.name

    @property
    def provider(self) -> str:
        return self._settings.utility.provider if self._inner is None else self._inner.provider

    def generate(self, prompt: str, **kwargs):
        return self._resolve().generate(prompt, **kwargs)

    def chat(self, messages, *, tools=None, **kwargs):
        return self._resolve().chat(messages, tools=tools, **kwargs)

    def health(self) -> bool:
        try:
            return self._resolve().health()
        except Exception:
            return False

