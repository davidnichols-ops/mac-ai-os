"""Tests for eval/operator_v3/eval_harness.py — the evaluation harness.

Covers:
- EvalTask.from_dict parsing (required fields, optional defaults)
- load_tasks (YAML/JSON, category filter, missing dir, malformed entries)
- _parse_task_file (YAML and JSON)
- _extract_tools_used (ordered tool names from trajectory)
- _had_error_then_continued (recovery signal detection)
- verify_task (all 4 checks: tool_used, min_tool_calls, no_tools, recovery)
- EvalReport (summary, to_dict, to_json, round-trip)
- EvalHarness._run_task (integration with ScriptedModel + real AgentLoop)
- save_report (file persistence)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.agent_loop import LoopResult
from app.core.trajectory import Step, StepType, Trajectory
from app.models.base import Model
from app.models.types import Message, ModelResponse, ToolCall
from eval.operator_v3.eval_harness import (
    EvalHarness,
    EvalReport,
    EvalTask,
    HarnessConfig,
    TaskResult,
    _extract_tools_used,
    _had_error_then_continued,
    load_tasks,
    save_report,
    verify_task,
)

# --- Helpers -----------------------------------------------------------------


class ScriptedModel(Model):
    name = "scripted"
    provider = "test"

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._idx = 0

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
        raise NotImplementedError

    def chat(self, messages: list[Message], *, tools=None, **kwargs) -> ModelResponse:
        if self._idx >= len(self._responses):
            raise RuntimeError("scripted model exhausted")
        r = self._responses[self._idx]
        self._idx += 1
        return r


def _make_traj(
    steps: list[Step],
    *,
    success: bool = True,
    tool_calls: int = 0,
) -> Trajectory:
    """Build a Trajectory with pre-populated steps."""
    t = Trajectory()
    for s in steps:
        t.add_step(s)
    t.success = success
    t.total_tool_calls = tool_calls or sum(1 for s in steps if s.step_type == StepType.TOOL_CALL)
    return t


def _make_loop_result(
    steps: list[Step],
    *,
    success: bool = True,
    tool_calls: int = 0,
) -> LoopResult:
    """Build a LoopResult with a trajectory containing the given steps."""
    traj = _make_traj(steps, success=success, tool_calls=tool_calls)
    return LoopResult(
        success=success,
        answer="test answer",
        trajectory=traj,
        tool_calls=tool_calls or sum(1 for s in steps if s.step_type == StepType.TOOL_CALL),
    )


def _tool_call_step(name: str) -> Step:
    return Step(step_type=StepType.TOOL_CALL, tool_name=name)


def _tool_result_step(ok: bool = True, error: str | None = None) -> Step:
    return Step(step_type=StepType.TOOL_RESULT, tool_ok=ok, error=error)


def _error_step(error: str = "something broke") -> Step:
    return Step(step_type=StepType.ERROR, error=error)


# --- EvalTask.from_dict ------------------------------------------------------


class TestEvalTaskFromDict:
    def test_required_fields(self):
        t = EvalTask.from_dict(
            {"name": "t1", "prompt": "do something", "expected_behavior": "x"},
            category="tool_competence",
        )
        assert t.name == "t1"
        assert t.category == "tool_competence"
        assert t.prompt == "do something"
        assert t.expected_behavior == "x"

    def test_optional_defaults(self):
        t = EvalTask.from_dict(
            {"name": "t1", "prompt": "p"},
            category="recovery",
        )
        assert t.expected_behavior == ""
        assert t.verify_tool_used == []
        assert t.verify_min_tool_calls == 0
        assert t.verify_no_tools is False
        assert t.verify_recovery is False
        assert t.verification is None

    def test_all_verification_fields(self):
        t = EvalTask.from_dict(
            {
                "name": "t1",
                "prompt": "p",
                "verify_tool_used": ["shell", "file_read"],
                "verify_min_tool_calls": 3,
                "verify_no_tools": True,
                "verify_recovery": True,
                "verification": {"custom": "check"},
            },
            category="planning",
        )
        assert t.verify_tool_used == ["shell", "file_read"]
        assert t.verify_min_tool_calls == 3
        assert t.verify_no_tools is True
        assert t.verify_recovery is True
        assert t.verification == {"custom": "check"}

    def test_missing_name_raises(self):
        with pytest.raises(KeyError):
            EvalTask.from_dict({"prompt": "p"}, category="tool_competence")

    def test_missing_prompt_raises(self):
        with pytest.raises(KeyError):
            EvalTask.from_dict({"name": "t1"}, category="tool_competence")


# --- load_tasks / _parse_task_file -------------------------------------------


class TestLoadTasks:
    def test_load_json_task_file(self, tmp_path: Path):
        f = tmp_path / "tasks.json"
        f.write_text(json.dumps({
            "category": "tool_competence",
            "tasks": [
                {"name": "t1", "prompt": "p1"},
                {"name": "t2", "prompt": "p2", "verify_tool_used": ["shell"]},
            ],
        }))
        tasks = load_tasks(tmp_path)
        assert len(tasks) == 2
        assert tasks[0].name == "t1"
        assert tasks[1].verify_tool_used == ["shell"]

    def test_load_yaml_task_file(self, tmp_path: Path):
        f = tmp_path / "tasks.yaml"
        f.write_text(
            "category: recovery\n"
            "tasks:\n"
            "  - name: r1\n"
            "    prompt: \"read bad path\"\n"
            "    verify_recovery: true\n"
        )
        tasks = load_tasks(tmp_path)
        assert len(tasks) == 1
        assert tasks[0].category == "recovery"
        assert tasks[0].verify_recovery is True

    def test_category_filter(self, tmp_path: Path):
        (tmp_path / "a.json").write_text(json.dumps({
            "category": "tool_competence", "tasks": [{"name": "t1", "prompt": "p"}],
        }))
        (tmp_path / "b.json").write_text(json.dumps({
            "category": "recovery", "tasks": [{"name": "r1", "prompt": "p"}],
        }))
        tasks = load_tasks(tmp_path, category="recovery")
        assert len(tasks) == 1
        assert tasks[0].name == "r1"

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            load_tasks(Path("/nonexistent/dir"))

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        assert load_tasks(tmp_path) == []

    def test_skips_file_without_category(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text(json.dumps({
            "tasks": [{"name": "t1", "prompt": "p"}],
        }))
        assert load_tasks(tmp_path) == []

    def test_skips_unknown_category(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text(json.dumps({
            "category": "unknown_cat", "tasks": [{"name": "t1", "prompt": "p"}],
        }))
        assert load_tasks(tmp_path) == []

    def test_skips_malformed_task_entry(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text(json.dumps({
            "category": "tool_competence",
            "tasks": [
                {"prompt": "missing name"},
                {"name": "good", "prompt": "p"},
            ],
        }))
        tasks = load_tasks(tmp_path)
        assert len(tasks) == 1
        assert tasks[0].name == "good"

    def test_loads_real_task_files(self):
        """Load the actual eval task files that ship with the repo."""
        tasks_dir = Path("eval/operator_v3/tasks")
        if not tasks_dir.is_dir():
            pytest.skip("eval task files not found")
        tasks = load_tasks(tasks_dir)
        assert len(tasks) > 0
        categories = {t.category for t in tasks}
        assert categories <= {"tool_competence", "recovery", "planning", "compute_allocation"}


# --- _extract_tools_used -----------------------------------------------------


class TestExtractToolsUsed:
    def test_returns_ordered_tool_names(self):
        traj = _make_traj([
            _tool_call_step("file_list"),
            _tool_result_step(),
            _tool_call_step("file_read"),
            _tool_result_step(),
        ])
        assert _extract_tools_used(traj) == ["file_list", "file_read"]

    def test_skips_non_tool_call_steps(self):
        traj = _make_traj([
            Step(step_type=StepType.MODEL_OUTPUT, content="thinking"),
            _tool_call_step("shell"),
            _tool_result_step(),
        ])
        assert _extract_tools_used(traj) == ["shell"]

    def test_empty_trajectory(self):
        traj = _make_traj([])
        assert _extract_tools_used(traj) == []

    def test_skips_tool_call_without_name(self):
        traj = _make_traj([
            Step(step_type=StepType.TOOL_CALL, tool_name=None),
            _tool_call_step("shell"),
        ])
        assert _extract_tools_used(traj) == ["shell"]


# --- _had_error_then_continued -----------------------------------------------


class TestHadErrorThenContinued:
    def test_error_then_continuation(self):
        traj = _make_traj([
            _tool_call_step("file_read"),
            _tool_result_step(ok=False, error="not found"),
            _tool_call_step("file_list"),
            _tool_result_step(ok=True),
        ])
        assert _had_error_then_continued(traj) is True

    def test_error_step_then_continuation(self):
        traj = _make_traj([
            _error_step("model crashed"),
            Step(step_type=StepType.MODEL_OUTPUT, content="retrying"),
        ])
        assert _had_error_then_continued(traj) is True

    def test_no_error(self):
        traj = _make_traj([
            _tool_call_step("file_list"),
            _tool_result_step(ok=True),
        ])
        assert _had_error_then_continued(traj) is False

    def test_error_at_end_no_continuation(self):
        traj = _make_traj([
            _tool_call_step("file_read"),
            _tool_result_step(ok=False, error="not found"),
        ])
        assert _had_error_then_continued(traj) is False

    def test_empty_trajectory(self):
        traj = _make_traj([])
        assert _had_error_then_continued(traj) is False


# --- verify_task -------------------------------------------------------------


class TestVerifyTask:
    def test_verify_tool_used_pass(self):
        task = EvalTask(name="t", category="tool_competence", prompt="p",
                        expected_behavior="", verify_tool_used=["shell", "file_list"])
        result = _make_loop_result([_tool_call_step("shell")], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is True
        assert len(checks) == 1
        assert checks[0]["check"] == "verify_tool_used"
        assert checks[0]["passed"] is True

    def test_verify_tool_used_fail(self):
        task = EvalTask(name="t", category="tool_competence", prompt="p",
                        expected_behavior="", verify_tool_used=["shell"])
        result = _make_loop_result([_tool_call_step("file_read")], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is False
        assert checks[0]["passed"] is False

    def test_verify_min_tool_calls_pass(self):
        task = EvalTask(name="t", category="planning", prompt="p",
                        expected_behavior="", verify_min_tool_calls=2)
        result = _make_loop_result([
            _tool_call_step("a"), _tool_call_step("b"),
        ], tool_calls=2)
        passed, checks = verify_task(task, result)
        assert passed is True
        assert checks[0]["actual"] == 2

    def test_verify_min_tool_calls_fail(self):
        task = EvalTask(name="t", category="planning", prompt="p",
                        expected_behavior="", verify_min_tool_calls=3)
        result = _make_loop_result([_tool_call_step("a")], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is False

    def test_verify_no_tools_pass(self):
        task = EvalTask(name="t", category="compute_allocation", prompt="p",
                        expected_behavior="", verify_no_tools=True)
        result = _make_loop_result([], tool_calls=0)
        passed, checks = verify_task(task, result)
        assert passed is True
        assert checks[0]["actual"] == 0

    def test_verify_no_tools_fail(self):
        task = EvalTask(name="t", category="compute_allocation", prompt="p",
                        expected_behavior="", verify_no_tools=True)
        result = _make_loop_result([_tool_call_step("shell")], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is False

    def test_verify_recovery_pass(self):
        task = EvalTask(name="t", category="recovery", prompt="p",
                        expected_behavior="", verify_recovery=True)
        result = _make_loop_result([
            _tool_call_step("file_read"),
            _tool_result_step(ok=False, error="not found"),
            _tool_call_step("file_list"),
            _tool_result_step(ok=True),
        ], success=False, tool_calls=2)
        passed, checks = verify_task(task, result)
        # recovery tasks relax the success requirement
        assert passed is True
        assert checks[0]["passed"] is True

    def test_verify_recovery_fail_no_error(self):
        task = EvalTask(name="t", category="recovery", prompt="p",
                        expected_behavior="", verify_recovery=True)
        result = _make_loop_result([
            _tool_call_step("file_list"),
            _tool_result_step(ok=True),
        ], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is False

    def test_no_checks_passes_if_success(self):
        task = EvalTask(name="t", category="tool_competence", prompt="p",
                        expected_behavior="")
        result = _make_loop_result([], success=True)
        passed, checks = verify_task(task, result)
        assert passed is True
        assert checks == []

    def test_no_checks_fails_if_unsuccessful(self):
        task = EvalTask(name="t", category="tool_competence", prompt="p",
                        expected_behavior="")
        result = _make_loop_result([], success=False)
        passed, checks = verify_task(task, result)
        assert passed is False

    def test_multiple_checks_all_pass(self):
        task = EvalTask(name="t", category="tool_competence", prompt="p",
                        expected_behavior="",
                        verify_tool_used=["shell"], verify_min_tool_calls=1)
        result = _make_loop_result([_tool_call_step("shell")], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is True
        assert len(checks) == 2
        assert all(c["passed"] for c in checks)

    def test_multiple_checks_one_fails(self):
        task = EvalTask(name="t", category="tool_competence", prompt="p",
                        expected_behavior="",
                        verify_tool_used=["shell"], verify_min_tool_calls=5)
        result = _make_loop_result([_tool_call_step("shell")], tool_calls=1)
        passed, checks = verify_task(task, result)
        assert passed is False
        assert checks[0]["passed"] is True
        assert checks[1]["passed"] is False


# --- EvalReport --------------------------------------------------------------


class TestEvalReport:
    def _make_report(self) -> EvalReport:
        results = [
            TaskResult(name="t1", category="tool_competence", prompt="p",
                       expected_behavior="", success=True, tool_calls=1,
                       steps=2, duration_s=1.5, answer="a",
                       failure_category="none", failure_detail="",
                       passed=True, tools_used=["shell"]),
            TaskResult(name="t2", category="recovery", prompt="p",
                       expected_behavior="", success=False, tool_calls=0,
                       steps=1, duration_s=0.5, answer="",
                       failure_category="env_error", failure_detail="broke",
                       passed=False, tools_used=[]),
        ]
        return EvalReport(
            model_name="test-model", model_provider="test",
            total_tasks=2, passed=1, failed=1, errors=0,
            pass_rate=0.5, duration_s=2.0,
            by_category={
                "tool_competence": {"total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0},
                "recovery": {"total": 1, "passed": 0, "failed": 1, "pass_rate": 0.0},
            },
            results=results,
        )

    def test_summary_contains_pass_rate(self):
        r = self._make_report()
        s = r.summary()
        assert "1/2 passed" in s
        assert "50.0%" in s

    def test_summary_contains_per_category(self):
        r = self._make_report()
        s = r.summary()
        assert "[tool_competence]" in s
        assert "[recovery]" in s

    def test_summary_contains_per_task_lines(self):
        r = self._make_report()
        s = r.summary()
        assert "PASS" in s
        assert "FAIL" in s
        assert "t1" in s
        assert "t2" in s

    def test_to_dict_round_trip(self):
        r = self._make_report()
        d = r.to_dict()
        assert d["total_tasks"] == 2
        assert d["passed"] == 1
        assert len(d["results"]) == 2
        assert d["results"][0]["name"] == "t1"

    def test_to_json_parses(self):
        r = self._make_report()
        j = r.to_json()
        d = json.loads(j)
        assert d["model_name"] == "test-model"
        assert d["pass_rate"] == 0.5


# --- save_report -------------------------------------------------------------


class TestSaveReport:
    def test_writes_json_file(self, tmp_path: Path):
        r = EvalReport(
            model_name="m", model_provider="p",
            total_tasks=0, passed=0, failed=0, errors=0,
            pass_rate=0.0, duration_s=0.0,
            by_category={}, results=[],
        )
        out = tmp_path / "sub" / "report.json"
        save_report(r, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["model_name"] == "m"

    def test_creates_parent_dirs(self, tmp_path: Path):
        r = EvalReport(
            model_name="m", model_provider="p",
            total_tasks=0, passed=0, failed=0, errors=0,
            pass_rate=0.0, duration_s=0.0,
            by_category={}, results=[],
        )
        out = tmp_path / "a" / "b" / "c" / "report.json"
        save_report(r, out)
        assert out.exists()


# --- EvalHarness integration -------------------------------------------------


class TestEvalHarnessIntegration:
    def _make_harness(self, model: Model, tmp_path: Path, tasks_dir: Path) -> EvalHarness:
        config = HarnessConfig(
            tasks_dir=tasks_dir,
            results_dir=tmp_path / "results",
            max_steps=5,
            max_errors=3,
            timeout_s=10.0,
            persist_trajectories=False,
            workspace=str(tmp_path),
        )
        return EvalHarness(model=model, config=config)

    def test_run_single_task_success(self, tmp_path: Path):
        """Run a real task through the harness with a ScriptedModel."""
        # Create a task file
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "t.json").write_text(json.dumps({
            "category": "tool_competence",
            "tasks": [{
                "name": "list_files",
                "prompt": "list files",
                "verify_tool_used": ["file_list"],
            }],
        }))
        # Create a workspace with a file
        (tmp_path / "hello.py").write_text("print('hi')\n")
        # Scripted model: call file_list then give answer
        model = ScriptedModel([
            ModelResponse(
                text="listing files",
                model="scripted", provider="test",
                tool_calls=[ToolCall(id="c1", name="file_list", arguments={"path": str(tmp_path)})],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                text="Found hello.py",
                model="scripted", provider="test",
                tool_calls=[],
                finish_reason="stop",
            ),
        ])
        harness = self._make_harness(model, tmp_path, tasks_dir)
        report = harness.run()
        assert report.total_tasks == 1
        assert report.passed == 1
        assert report.results[0].passed is True
        assert report.results[0].success is True
        assert "file_list" in report.results[0].tools_used

    def test_run_task_model_errors_caught_by_loop(self, tmp_path: Path):
        """Model exceptions are caught by the agent loop, not the harness.

        The loop has its own try/except around model.chat — it increments
        the error budget and eventually returns a failed LoopResult. The
        harness sees this as a normal failed task, not a harness_error.
        """
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "t.json").write_text(json.dumps({
            "category": "tool_competence",
            "tasks": [{"name": "boom", "prompt": "explode"}],
        }))
        class ExplodingModel(Model):
            name = "boom"
            provider = "test"
            def generate(self, prompt, **kw): raise NotImplementedError
            def chat(self, messages, **kw): raise RuntimeError("model exploded")

        harness = self._make_harness(ExplodingModel(), tmp_path, tasks_dir)
        report = harness.run()
        assert report.total_tasks == 1
        assert report.errors == 0  # no harness-level errors
        assert report.failed == 1
        assert report.results[0].harness_error is None
        assert report.results[0].success is False
        # The loop exhausted its error budget (3) and returned ENVIRONMENT_ERROR
        assert report.results[0].failure_category in ("env_error", "model_error")

    def test_run_empty_tasks(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        model = ScriptedModel([])
        harness = self._make_harness(model, tmp_path, tasks_dir)
        report = harness.run()
        assert report.total_tasks == 0
        assert report.passed == 0
        assert report.pass_rate == 0.0

    def test_category_filter(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "a.json").write_text(json.dumps({
            "category": "tool_competence",
            "tasks": [{"name": "t1", "prompt": "p"}],
        }))
        (tasks_dir / "b.json").write_text(json.dumps({
            "category": "recovery",
            "tasks": [{"name": "r1", "prompt": "p", "verify_recovery": True}],
        }))
        (tmp_path / "hello.py").write_text("print('hi')")
        model = ScriptedModel([
            ModelResponse(text="done", model="scripted", provider="test",
                          tool_calls=[], finish_reason="stop"),
        ])
        config = HarnessConfig(
            tasks_dir=tasks_dir, results_dir=tmp_path / "results",
            category="tool_competence",
            max_steps=3, timeout_s=10.0, persist_trajectories=False,
            workspace=str(tmp_path),
        )
        harness = EvalHarness(model=model, config=config)
        report = harness.run()
        assert report.total_tasks == 1
        assert report.results[0].name == "t1"
