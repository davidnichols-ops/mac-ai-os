"""MAOS Load Workbench — pressure-test every MAOS stack at scale.

The workbench drives synthetic user interactions through the local MAOS
runtime, hitting observation, cognitive, agency, workflow, and governance
layers. It is designed to find the practical break point on the path to
50 million interactions and either fix the gap or record it as a task.
"""

from __future__ import annotations

import random
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.actions.builtin import builtin_actions
from app.actions.executor import ActionExecutor
from app.actions.models import ActionRequest
from app.actions.permissions import PermissionChecker
from app.actions.registry import ActionRegistry
from app.blackboard import Blackboard
from app.context import build_context
from app.db import Database
from app.evaluation import evaluate_context
from app.events.bus import EventBus
from app.events.models import Event
from app.guards import scan_vibes
from app.knowledge.graph import KnowledgeGraph
from app.learning import Learner
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message, ModelResponse
from app.network.provider import NetworkProvider
from app.slack.auth import SlackAuth
from app.slack.commands import SlackCommandRouter
from app.umbrella import build_snapshot
from app.workflows.engine import WorkflowEngine
from app.workflows.models import Workflow, WorkflowStep

log = get_logger(__name__)


class LoadModel(Model):
    """No-op model that returns canned JSON for knowledge-graph extraction."""

    name = "load"
    provider = "load"

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
        return ModelResponse(text='{"entities": [], "relationships": []}', model=self.name, provider=self.provider)

    def chat(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None, **kwargs) -> ModelResponse:
        return ModelResponse(text='{"entities": [], "relationships": []}', model=self.name, provider=self.provider)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class LoadReport:
    """Result of a load run."""

    interactions: int
    duration_s: float
    throughput_ips: float
    errors: dict[str, int] = field(default_factory=dict)
    stack_hits: dict[str, int] = field(default_factory=dict)
    step_times: dict[str, float] = field(default_factory=dict)
    db_size_mb: float = 0.0
    projected_50m_s: float = 0.0
    projected_50m_h: float = 0.0
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LoadWorkbench:
    """Drive synthetic user interactions through every MAOS stack."""

    def __init__(
        self,
        db_path: str = "data/load.db",
        *,
        home: Path | None = None,
        batch_size: int = 1000,
        real_telemetry: bool = False,
    ) -> None:
        self.db_path = db_path
        self.home = home or Path.home()
        self.batch_size = batch_size
        self.real_telemetry = real_telemetry

        self.db = Database(db_path)
        self.bus = EventBus(db=self.db)
        self.blackboard = Blackboard()
        self.blackboard.restore(self.db)
        self.learner = Learner(self.db)

        # Agency layer.
        self.registry = ActionRegistry()
        for action, handler in builtin_actions():
            self.registry.register(action, handler)
        self.permissions = PermissionChecker(granted={"*"}, auto_approve=True)
        self.action_executor = ActionExecutor(
            self.registry, self.permissions, self.db, bus=self.bus, allow_irreversible=True,
        )

        # Cognitive / knowledge.
        self.knowledge = KnowledgeGraph(self.db, planner=LoadModel())
        self.network = NetworkProvider(self.db)

        # Workflow.
        self.workflow_engine = WorkflowEngine(
            self.db,
            bus=self.bus,
            permission_checker=self.permissions,
            action_executor=self.action_executor,
            action_registry=self.registry,
        )
        self._load_workflow = self._make_load_workflow()

        # Slack (with a fake runtime bound to the workbench home).
        self.slack_router = self._make_slack_router()

        # Step registry: name -> (weight, callable).
        self.steps: dict[str, tuple[int, Callable[[], None]]] = {
            "event": (20, self._step_event),
            "task": (15, self._step_task),
            "workspace": (10, self._step_workspace),
            "action": (10, self._step_action),
            "decision": (5, self._step_decision),
            "session": (5, self._step_session),
            "telemetry": (10, self._step_telemetry),
            "network": (10, self._step_network),
            "shell_command": (5, self._step_shell_command),
            "graph": (5, self._step_graph),
            "learn": (5, self._step_learn),
            "blackboard": (5, self._step_blackboard),
            "context": (3, self._step_context),
            "evaluate": (2, self._step_evaluate),
            "ask": (1, self._step_ask),
            "umbrella": (1, self._step_umbrella),
            "guard": (3, self._step_guard),
            "workflow": (3, self._step_workflow),
            "slack": (1, self._step_slack),
        }
        self.stack_hits: Counter = Counter()
        self.errors: Counter = Counter()
        self.step_times: defaultdict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ setup
    def _make_load_workflow(self) -> Workflow:
        def handler(ctx: dict[str, Any]) -> dict[str, Any]:
            db: Database = ctx["db"]
            db.create_task("load workflow task", source="ai")
            return {"ok": True}

        def verify(data: dict[str, Any]) -> Any:
            from app.workflows.models import VerificationResult
            return VerificationResult(success=data.get("ok", False))

        return Workflow(
            name="load_workflow",
            description="Create a task within a workflow.",
            steps=[
                WorkflowStep(
                    name="create_load_task",
                    description="Create a synthetic task.",
                    handler=handler,
                    verify=verify,
                ),
            ],
            context={},
        )

    def _make_slack_router(self) -> SlackCommandRouter:
        rt = SimpleNamespace()
        rt.db = self.db
        rt.network = self.network
        rt.settings = SimpleNamespace(slack=SimpleNamespace(require_approval=False))
        rt.retriever = None
        rt.action_executor = self.action_executor
        rt.blackboard = self.blackboard
        rt.orchestrator = None
        rt.home = self.home
        rt.qa_model = LoadModel()
        auth = SlackAuth(allowlist=[], rate_limit_seconds=0.0)
        return SlackCommandRouter(rt, auth)

    # ------------------------------------------------------------------ steps
    def _step_event(self) -> None:
        self.bus.publish(Event("USER_INTERACTION", {"load": True, "ts": _now()}))

    def _step_task(self) -> None:
        self.db.create_task(f"load task {uuid.uuid4().hex[:8]}", source="ai")

    def _step_workspace(self) -> None:
        name = f"ws-{uuid.uuid4().hex[:8]}"
        self.db.upsert_workspace(
            str(self.home / "load" / name),
            name=name,
            language="python",
            git_repo=False,
            file_count=1,
            last_activity=_now(),
        )

    def _step_action(self) -> None:
        req = ActionRequest(action="list_tasks", target="all", parameters={})
        self.action_executor.execute(req, context={"db": self.db})

    def _step_decision(self) -> None:
        self.db.log_decision(
            agent="load",
            decision="continue",
            reasoning="synthetic load decision",
            confidence=0.9,
            outcome="ok",
            query="load",
        )

    def _step_session(self) -> None:
        now = _now()
        self.db.create_session(now, now, project="load", workspace="load", summary="synthetic", event_count=1)

    def _step_telemetry(self) -> None:
        if self.real_telemetry:
            from app.telemetry.collector import _collect_snapshot, _collect_top_processes
            snap = _collect_snapshot()
            procs = _collect_top_processes(3)
        else:
            snap = {
                "timestamp": _now(),
                "cpu_percent": 1.0,
                "cpu_cores": 8,
                "memory_total_gb": 16.0,
                "memory_used_gb": 8.0,
                "memory_percent": 50.0,
                "swap_used_gb": 0.0,
                "swap_total_gb": 0.0,
                "disk_read_mb": 0.0,
                "disk_write_mb": 0.0,
                "process_count": 100,
            }
            procs = [{"pid": i, "name": f"proc{i}", "cpu": 0.0, "memory_mb": 1.0} for i in range(3)]
        self.db.insert_telemetry(snap, procs)

    def _step_network(self) -> None:
        # Insert a synthetic sample and also read the provider.
        self.db.insert_network_sample({
            "timestamp": _now(),
            "interface": "en0",
            "network_type": "wifi",
            "connection_state": "connected",
            "rx_bytes": 1000,
            "tx_bytes": 1000,
            "rx_rate_bps": 1000.0,
            "tx_rate_bps": 1000.0,
            "latency_ms": 10.0,
            "packet_loss": 0.0,
        })
        self.network.get_status()

    def _step_shell_command(self) -> None:
        self.db.insert_shell_command(_now(), f"echo load-{uuid.uuid4().hex[:8]}", exit_code=0)

    def _step_graph(self) -> None:
        name = f"concept-{uuid.uuid4().hex[:8]}"
        self.knowledge.extract_from_text(
            f"The {name} project uses MAOS and trustcard for secure agent operations.",
            source="load",
        )

    def _step_learn(self) -> None:
        self.learner.record_action_outcome("list_tasks", "all", success=True)

    def _step_blackboard(self) -> None:
        self.blackboard.set_goal(f"load goal {uuid.uuid4().hex[:8]}")
        self.blackboard.persist(self.db)

    def _step_context(self) -> None:
        build_context(self.db, "what is the load state?", retriever=None, lookback_hours=0, top_k=2)

    def _step_evaluate(self) -> None:
        evaluate_context(self.db)

    def _step_ask(self) -> None:
        # Ensure home surface exists for umbrella-style context.
        (self.home / "Projects" / "load").mkdir(parents=True, exist_ok=True)
        (self.home / "bin").mkdir(parents=True, exist_ok=True)
        self.slack_router.handle("U_LOAD", "ask what is the current load state?")

    def _step_umbrella(self) -> None:
        build_snapshot(self.db, home=self.home)

    def _step_guard(self) -> None:
        scan_vibes("the load harness is running and everything is verified with evidence")

    def _step_workflow(self) -> None:
        self.workflow_engine.run(self._load_workflow)

    def _step_slack(self) -> None:
        # Ensure a minimal home surface exists for the umbrella Slack command.
        (self.home / "Projects" / "load").mkdir(parents=True, exist_ok=True)
        (self.home / "bin").mkdir(parents=True, exist_ok=True)
        self.slack_router.handle("U_LOAD", "umbrella")

    # ------------------------------------------------------------------ run
    def _flush_batch(self) -> None:
        """Commit and checkpoint telemetry / blackboard state."""
        with self.db.transaction():
            self.blackboard.persist(self.db)

    def _run_batch(self, batch_size: int) -> None:
        """Execute a batch of interactions inside a single DB transaction."""
        with self.db.transaction():
            for _ in range(batch_size):
                name, fn = self._pick_step()
                t0 = time.perf_counter()
                try:
                    fn()
                    self.stack_hits[name] += 1
                except Exception as e:
                    self.errors[name] += 1
                    log.warning("load.step_failed", step=name, error=str(e))
                self.step_times[name] += time.perf_counter() - t0
            self.blackboard.persist(self.db)

    def _pick_step(self) -> tuple[str, Callable[[], None]]:
        names = list(self.steps.keys())
        weights = [w for w, _ in self.steps.values()]
        name = random.choices(names, weights=weights, k=1)[0]
        return name, self.steps[name][1]

    def run(self, interactions: int) -> LoadReport:
        """Run the load workbench for N interactions."""
        log.info("load.start", interactions=interactions, db=self.db_path)
        start = time.perf_counter()

        remaining = interactions
        while remaining > 0:
            batch = min(remaining, self.batch_size)
            self._run_batch(batch)
            remaining -= batch

        # Final checkpoint.
        self._flush_batch()

        elapsed = time.perf_counter() - start
        db_size = Path(self.db_path).stat().st_size / (1024 * 1024)

        report = LoadReport(
            interactions=interactions,
            duration_s=round(elapsed, 3),
            throughput_ips=round(interactions / elapsed, 2) if elapsed > 0 else 0.0,
            errors=dict(self.errors),
            stack_hits=dict(self.stack_hits),
            step_times={k: round(v, 3) for k, v in self.step_times.items()},
            db_size_mb=round(db_size, 2),
            projected_50m_s=round(elapsed * (50_000_000 / interactions), 2) if interactions > 0 else 0.0,
            projected_50m_h=round(elapsed * (50_000_000 / interactions) / 3600, 2) if interactions > 0 else 0.0,
            blockers=self._detect_blockers(elapsed, interactions, db_size),
        )
        log.info("load.done", report=report.to_dict())
        return report

    def _detect_blockers(self, elapsed: float, interactions: int, db_size_mb: float) -> list[str]:
        """Surface practical blockers for 50 million interactions."""
        blockers: list[str] = []
        if interactions == 0:
            return blockers

        throughput = interactions / elapsed
        if throughput < 100:
            blockers.append(
                f"Throughput ({throughput:.1f} ips) is too low for 50M in a reasonable window. "
                "Investigate batching, WAL tuning, or alternative persistence."
            )
        if db_size_mb > 0 and (db_size_mb / interactions) * 50_000_000 > 10_000:
            blockers.append(
                "Projected DB size exceeds 10 GB for 50M interactions. Add retention/rotation."
            )
        if self.errors:
            blockers.append(
                f"Steps failed: {dict(self.errors)}. Fix error paths before scaling."
            )
        if throughput > 0:
            projected_h = (50_000_000 / throughput) / 3600
            if projected_h > 24:
                blockers.append(
                    f"Projected 50M run time is {projected_h:.1f} hours. Parallelize or shard."
                )
        return blockers
