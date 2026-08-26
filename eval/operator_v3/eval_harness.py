"""Evaluation harness for Adaptive Operator v3.

Loads evaluation tasks from YAML/JSON files, runs each task through the real
MAOS ``AgentLoop``, verifies the outcome against the task's declared
expectations, and produces a structured summary report.

The harness is model-agnostic: it accepts any ``Model`` instance and wires up
the full tool/safety/trajectory stack so the operator runs exactly as it would
in production.

Usage::

    from app.models.registry import build_model
    from app.config.settings import load_settings
    from eval.operator_v3.eval_harness import EvalHarness, HarnessConfig

    settings = load_settings()
    model = build_model("planner", settings)
    harness = EvalHarness(model=model, config=HarnessConfig())
    report = harness.run()
    print(report.summary())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.agent_loop import AgentLoop, LoopConfig, LoopResult
from app.core.safety import SafetyGate
from app.core.tool_executor import ToolExecutor
from app.core.trajectory import StepType, Trajectory
from app.log import get_logger
from app.models.base import Model
from app.tools.builtin import builtin_tools
from app.tools.registry import ToolRouter

log = get_logger(__name__)

# Optional YAML support — fall back to JSON if PyYAML is unavailable.
try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# The four evaluation categories.
VALID_CATEGORIES: frozenset[str] = frozenset(
    {"tool_competence", "recovery", "planning", "compute_allocation"}
)


@dataclass
class EvalTask:
    """A single evaluation task loaded from a task file."""

    name: str
    category: str
    prompt: str
    expected_behavior: str
    # Verification fields (all optional)
    verify_tool_used: list[str] = field(default_factory=list)
    verify_min_tool_calls: int = 0
    verify_no_tools: bool = False
    verify_recovery: bool = False
    # Free-form verification block (optional, currently informational)
    verification: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], category: str) -> EvalTask:
        """Build an EvalTask from a parsed task-file entry."""
        return cls(
            name=data["name"],
            category=category,
            prompt=data["prompt"],
            expected_behavior=data.get("expected_behavior", ""),
            verify_tool_used=list(data.get("verify_tool_used", []) or []),
            verify_min_tool_calls=int(data.get("verify_min_tool_calls", 0) or 0),
            verify_no_tools=bool(data.get("verify_no_tools", False)),
            verify_recovery=bool(data.get("verify_recovery", False)),
            verification=data.get("verification"),
        )


@dataclass
class TaskResult:
    """The outcome of running a single EvalTask through the agent loop."""

    name: str
    category: str
    prompt: str
    expected_behavior: str
    # Raw loop metrics
    success: bool
    tool_calls: int
    steps: int
    duration_s: float
    answer: str
    failure_category: str
    failure_detail: str
    # Verification
    passed: bool
    verification_checks: list[dict[str, Any]] = field(default_factory=list)
    # Tools actually used (names, in order)
    tools_used: list[str] = field(default_factory=list)
    # Whether an error occurred and the loop continued
    had_error_then_continued: bool = False
    # Full trajectory (serialized) for replay/analysis
    trajectory: dict[str, Any] | None = None
    # Error running the harness itself (not the agent loop)
    harness_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "success": self.success,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "duration_s": round(self.duration_s, 3),
            "answer": self.answer[:2000],
            "failure_category": self.failure_category,
            "failure_detail": self.failure_detail,
            "passed": self.passed,
            "verification_checks": self.verification_checks,
            "tools_used": self.tools_used,
            "had_error_then_continued": self.had_error_then_continued,
            "trajectory": self.trajectory,
            "harness_error": self.harness_error,
        }


@dataclass
class EvalReport:
    """Aggregate report across all tasks."""

    model_name: str
    model_provider: str
    total_tasks: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    duration_s: float
    by_category: dict[str, dict[str, Any]]
    results: list[TaskResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "total_tasks": self.total_tasks,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": round(self.pass_rate, 4),
            "duration_s": round(self.duration_s, 3),
            "by_category": self.by_category,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable one-line-per-task summary."""
        lines = [
            f"Eval Report — {self.model_name} ({self.model_provider})",
            f"  {self.passed}/{self.total_tasks} passed "
            f"({self.pass_rate:.1%}) in {self.duration_s:.1f}s",
            "",
        ]
        for cat, stats in sorted(self.by_category.items()):
            lines.append(
                f"  [{cat}] {stats['passed']}/{stats['total']} passed "
                f"({stats['pass_rate']:.1%})"
            )
        lines.append("")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            if r.harness_error:
                status = "ERROR"
            lines.append(
                f"  {status:5s} {r.category:20s} {r.name:28s} "
                f"tools={r.tool_calls} steps={r.steps} {r.duration_s:.1f}s"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness configuration
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """Configuration for the evaluation harness."""

    tasks_dir: Path = Path("eval/operator_v3/tasks")
    results_dir: Path = Path("eval/operator_v3/results")
    # Filter to a single category (None = run all)
    category: str | None = None
    # Agent loop config
    max_steps: int = 15
    max_errors: int = 8
    timeout_s: float = 120.0
    temperature: float = 0.3
    max_tokens: int = 2048
    workspace: str = "."
    # Persist trajectories to disk
    persist_trajectories: bool = True
    trajectory_dir: Path | None = None
    # Auto-approve all tool calls (eval runs autonomously)
    auto_approve: bool = True


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(tasks_dir: Path, category: str | None = None) -> list[EvalTask]:
    """Load all evaluation tasks from a directory.

    Supports ``.yaml``/``.yml`` (if PyYAML is installed) and ``.json`` files.
    Each file defines a top-level ``category`` and a list of ``tasks``.

    Args:
        tasks_dir: Directory containing task definition files.
        category: If set, only load tasks from this category.

    Returns:
        A list of ``EvalTask`` objects.
    """
    tasks: list[EvalTask] = []
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks directory not found: {tasks_dir}")

    files = sorted(
        p for p in tasks_dir.iterdir()
        if p.suffix in (".yaml", ".yml", ".json")
    )

    for path in files:
        data = _parse_task_file(path)
        file_category = data.get("category", "")
        if not file_category:
            log.warning("eval.task_file.no_category", path=str(path))
            continue
        if file_category not in VALID_CATEGORIES:
            log.warning(
                "eval.task_file.unknown_category",
                path=str(path), category=file_category,
            )
            continue
        if category and file_category != category:
            continue

        for entry in data.get("tasks", []):
            try:
                tasks.append(EvalTask.from_dict(entry, category=file_category))
            except KeyError as e:
                log.error(
                    "eval.task_file.malformed",
                    path=str(path), error=str(e), entry=entry,
                )
    return tasks


def _parse_task_file(path: Path) -> dict[str, Any]:
    """Parse a single task file (YAML or JSON)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError(
                f"PyYAML is required to parse {path} but is not installed. "
                "Install pyyaml or use .json task files."
            )
        return yaml.safe_load(text) or {}
    return json.loads(text)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _extract_tools_used(trajectory: Trajectory) -> list[str]:
    """Extract the ordered list of tool names called during a trajectory."""
    return [
        s.tool_name
        for s in trajectory.steps
        if s.step_type == StepType.TOOL_CALL and s.tool_name
    ]


def _had_error_then_continued(trajectory: Trajectory) -> bool:
    """Check whether an error occurred and the loop continued afterward.

    This is the recovery signal: a tool or model error happened, but the
    trajectory did not terminate immediately after.
    """
    steps = trajectory.steps
    for i, s in enumerate(steps):
        is_error = (
            s.step_type == StepType.ERROR
            or (s.step_type == StepType.TOOL_RESULT and s.tool_ok is False)
        )
        if is_error and i + 1 < len(steps):
            return True
    return False


def verify_task(task: EvalTask, result: LoopResult) -> tuple[bool, list[dict[str, Any]]]:
    """Verify a loop result against the task's declared expectations.

    Returns:
        (passed, checks) where checks is a list of per-check dicts with
        ``check``, ``expected``, ``actual``, and ``passed`` fields.
    """
    checks: list[dict[str, Any]] = []
    trajectory = result.trajectory
    tools_used = _extract_tools_used(trajectory)

    # 1. verify_tool_used — at least one of the listed tools must be called
    if task.verify_tool_used:
        matched = [t for t in tools_used if t in task.verify_tool_used]
        check = {
            "check": "verify_tool_used",
            "expected": task.verify_tool_used,
            "actual": tools_used,
            "passed": len(matched) > 0,
        }
        checks.append(check)

    # 2. verify_min_tool_calls — total tool calls must meet the minimum
    if task.verify_min_tool_calls:
        actual = result.tool_calls
        check = {
            "check": "verify_min_tool_calls",
            "expected": f">= {task.verify_min_tool_calls}",
            "actual": actual,
            "passed": actual >= task.verify_min_tool_calls,
        }
        checks.append(check)

    # 3. verify_no_tools — no tools should have been called
    if task.verify_no_tools:
        check = {
            "check": "verify_no_tools",
            "expected": 0,
            "actual": result.tool_calls,
            "passed": result.tool_calls == 0,
        }
        checks.append(check)

    # 4. verify_recovery — an error must occur and the loop must continue
    if task.verify_recovery:
        recovered = _had_error_then_continued(trajectory)
        check = {
            "check": "verify_recovery",
            "expected": "error then continuation",
            "actual": recovered,
            "passed": recovered,
        }
        checks.append(check)

    # Overall pass: all declared checks pass AND the loop itself succeeded.
    # For recovery tasks, we relax the success requirement — the point is
    # that the operator handled the error gracefully, not that the original
    # (impossible) task "succeeded".
    all_checks_pass = all(c["passed"] for c in checks)
    if task.verify_recovery:
        passed = all_checks_pass
    else:
        passed = all_checks_pass and result.success

    return passed, checks


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Runs evaluation tasks through the MAOS AgentLoop and verifies results.

    The harness wires up a fresh ``ToolRouter`` + ``ToolExecutor`` + ``SafetyGate``
    + ``AgentLoop`` for each task so there is no state leakage between tasks.
    The model instance is reused across tasks (model loading is expensive).
    """

    def __init__(self, *, model: Model, config: HarnessConfig | None = None):
        self.model = model
        self.config = config or HarnessConfig()

    def run(self) -> EvalReport:
        """Load tasks, run each through the agent loop, return an EvalReport."""
        tasks = load_tasks(self.config.tasks_dir, category=self.config.category)
        if not tasks:
            log.warning("eval.no_tasks", tasks_dir=str(self.config.tasks_dir))

        results: list[TaskResult] = []
        start = time.monotonic()

        for task in tasks:
            log.info("eval.task.start", name=task.name, category=task.category)
            result = self._run_task(task)
            results.append(result)
            log.info(
                "eval.task.done",
                name=task.name,
                passed=result.passed,
                success=result.success,
                tool_calls=result.tool_calls,
                duration_s=round(result.duration_s, 2),
            )

        total_duration = time.monotonic() - start
        return self._build_report(results, total_duration)

    # -- internals ---------------------------------------------------------

    def _run_task(self, task: EvalTask) -> TaskResult:
        """Run a single task through a fresh agent loop."""
        # Build a fresh tool stack per task for isolation.
        router = ToolRouter(
            builtin_tools(),
            granted_permissions={"shell.exec", "file.read", "file.write", "git.exec", "web.search", "web.fetch", "todo.write", "user.interact"},
        )
        safety = SafetyGate(
            auto_approve_low=True,
            auto_approve_medium=True,
            approval_callback=self._auto_approve if self.config.auto_approve else None,
        )
        executor = ToolExecutor(router=router, safety=safety)

        loop_config = LoopConfig(
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
            timeout_s=self.config.timeout_s,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            workspace=self.config.workspace,
            persist_trajectories=self.config.persist_trajectories,
            trajectory_dir=self.config.trajectory_dir,
        )
        loop = AgentLoop(model=self.model, executor=executor, config=loop_config)

        try:
            loop_result = loop.run(goal=task.prompt)
        except Exception as e:
            log.exception("eval.task.exception", name=task.name)
            return TaskResult(
                name=task.name,
                category=task.category,
                prompt=task.prompt,
                expected_behavior=task.expected_behavior,
                success=False,
                tool_calls=0,
                steps=0,
                duration_s=0.0,
                answer="",
                failure_category="harness_error",
                failure_detail=str(e),
                passed=False,
                harness_error=str(e),
            )

        # Verify
        passed, checks = verify_task(task, loop_result)
        tools_used = _extract_tools_used(loop_result.trajectory)
        recovered = _had_error_then_continued(loop_result.trajectory)

        return TaskResult(
            name=task.name,
            category=task.category,
            prompt=task.prompt,
            expected_behavior=task.expected_behavior,
            success=loop_result.success,
            tool_calls=loop_result.tool_calls,
            steps=loop_result.steps,
            duration_s=loop_result.duration_s,
            answer=loop_result.answer,
            failure_category=loop_result.failure_category.value,
            failure_detail=loop_result.failure_detail,
            passed=passed,
            verification_checks=checks,
            tools_used=tools_used,
            had_error_then_continued=recovered,
            trajectory=loop_result.trajectory.to_dict(),
        )

    def _build_report(self, results: list[TaskResult], duration_s: float) -> EvalReport:
        """Aggregate task results into an EvalReport."""
        passed = sum(1 for r in results if r.passed)
        errors = sum(1 for r in results if r.harness_error)
        failed = len(results) - passed
        total = len(results)
        pass_rate = passed / total if total else 0.0

        # Per-category breakdown
        by_category: dict[str, dict[str, Any]] = {}
        for r in results:
            cat = by_category.setdefault(
                r.category,
                {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
            )
            cat["total"] += 1
            if r.passed:
                cat["passed"] += 1
            else:
                cat["failed"] += 1
        for cat in by_category.values():
            cat["pass_rate"] = cat["passed"] / cat["total"] if cat["total"] else 0.0

        return EvalReport(
            model_name=self.model.name,
            model_provider=self.model.provider,
            total_tasks=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=pass_rate,
            duration_s=duration_s,
            by_category=by_category,
            results=results,
        )

    @staticmethod
    def _auto_approve(risk: Any, tool_name: str, args: dict[str, Any]) -> bool:
        """Auto-approve all high-risk operations during evaluation."""
        log.info("eval.auto_approve", tool=tool_name, risk=str(risk))
        return True


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


def save_report(report: EvalReport, output: Path) -> Path:
    """Write an EvalReport to a JSON file, creating parent dirs as needed."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json(), encoding="utf-8")
    log.info("eval.report.saved", path=str(output))
    return output
