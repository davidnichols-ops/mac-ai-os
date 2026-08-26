"""Tests for app/core/trajectory.py - trajectory recording and serialization.

Covers:
- Trajectory dataclass: defaults, add_step, finalize metrics computation
- Compute mode tracking (fast/think/recover/verify + switches)
- Error and recovery action collection in finalize
- to_dict / to_json / from_dict round-trip
- from_dict v2.0 backward compatibility (missing v2.1 fields)
- TrajectoryRecorder: all record_* methods
- TrajectoryRecorder.finalize returns Trajectory + computes time
- TrajectoryRecorder persistence to disk
- SCHEMA_VERSION, StepType, ComputeMode, FailureCategory enums
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.trajectory import (
    SCHEMA_VERSION,
    ComputeMode,
    FailureCategory,
    Step,
    StepType,
    Trajectory,
    TrajectoryClass,
    TrajectoryRecorder,
)

# --- Enums / constants -------------------------------------------------------


class TestEnums:
    def test_schema_version_is_2_1(self):
        assert SCHEMA_VERSION == "2.1.0"

    def test_step_type_values(self):
        assert StepType.OBSERVATION.value == "observation"
        assert StepType.MODEL_OUTPUT.value == "model_output"
        assert StepType.TOOL_CALL.value == "tool_call"
        assert StepType.TOOL_RESULT.value == "tool_result"
        assert StepType.ERROR.value == "error"
        assert StepType.VERIFICATION.value == "verification"
        assert StepType.RECOVERY.value == "recovery"
        assert StepType.TERMINATION.value == "termination"

    def test_compute_mode_values(self):
        assert ComputeMode.FAST.value == "fast"
        assert ComputeMode.THINK.value == "think"
        assert ComputeMode.RECOVER.value == "recover"
        assert ComputeMode.VERIFY.value == "verify"

    def test_failure_category_values(self):
        assert FailureCategory.NONE.value == "none"
        assert FailureCategory.WRONG_TOOL.value == "wrong_tool"
        assert FailureCategory.INFINITE_LOOP.value == "infinite_loop"
        assert FailureCategory.STEP_BUDGET_EXCEEDED.value == "step_budget"
        assert FailureCategory.SAFETY_VIOLATION.value == "safety"


# --- Trajectory defaults -----------------------------------------------------


class TestTrajectoryDefaults:
    def test_defaults(self):
        t = Trajectory()
        assert t.trajectory_id.startswith("traj_")
        assert t.schema_version == SCHEMA_VERSION
        assert t.task == ""
        assert t.steps == []
        assert t.success is False
        assert t.failure_category == FailureCategory.NONE
        assert t.total_steps == 0
        assert t.total_tool_calls == 0
        assert t.fast_steps == 0
        assert t.think_steps == 0
        assert t.recover_steps == 0
        assert t.verify_steps == 0
        assert t.compute_mode_switches == 0
        assert t.safety_violations == 0
        assert t.approvals_requested == 0
        assert t.created_at != ""
        assert t.completed_at is None

    def test_trajectory_id_is_unique(self):
        ids = {Trajectory().trajectory_id for _ in range(100)}
        assert len(ids) == 100


# --- add_step ----------------------------------------------------------------


class TestAddStep:
    def test_add_step_increments_step_id(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content="a"))
        t.add_step(Step(step_type=StepType.TOOL_CALL, tool_name="shell"))
        t.add_step(Step(step_type=StepType.TOOL_RESULT, tool_ok=True))
        assert t.steps[0].step_id == 0
        assert t.steps[1].step_id == 1
        assert t.steps[2].step_id == 2
        assert len(t.steps) == 3

    def test_add_step_appends(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content="first"))
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content="second"))
        assert t.steps[0].content == "first"
        assert t.steps[1].content == "second"


# --- finalize ----------------------------------------------------------------


class TestFinalize:
    def test_sets_success_and_answer(self):
        t = Trajectory()
        t.finalize(success=True, final_answer="done")
        assert t.success is True
        assert t.final_answer == "done"

    def test_total_steps_computed(self):
        t = Trajectory()
        for i in range(5):
            t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content=str(i)))
        t.finalize(success=True)
        assert t.total_steps == 5

    def test_total_tool_calls_computed(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT))
        t.add_step(Step(step_type=StepType.TOOL_CALL, tool_name="a"))
        t.add_step(Step(step_type=StepType.TOOL_RESULT))
        t.add_step(Step(step_type=StepType.TOOL_CALL, tool_name="b"))
        t.add_step(Step(step_type=StepType.TOOL_RESULT))
        t.finalize(success=True)
        assert t.total_tool_calls == 2

    def test_total_tokens_computed(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, completion_tokens=50))
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, completion_tokens=30))
        t.finalize(success=True)
        assert t.total_tokens == 80

    def test_total_generation_time_computed(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, generation_time_s=1.5))
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, generation_time_s=2.5))
        t.finalize(success=True)
        assert t.total_generation_time_s == 4.0

    def test_compute_mode_fast_steps(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode="fast"))
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode="fast"))
        t.finalize(success=True)
        assert t.fast_steps == 2
        assert t.think_steps == 0

    def test_compute_mode_think_steps(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode="think"))
        t.finalize(success=True)
        assert t.think_steps == 1
        assert t.fast_steps == 0

    def test_compute_mode_recover_steps(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.RECOVERY, compute_mode="recover"))
        t.finalize(success=True)
        assert t.recover_steps == 1

    def test_compute_mode_verify_steps(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.VERIFICATION, compute_mode="verify"))
        t.finalize(success=True)
        assert t.verify_steps == 1

    def test_compute_mode_switches(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode="fast"))
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode="think"))
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode="fast"))
        t.finalize(success=True)
        assert t.compute_mode_switches == 2  # fast->think, think->fast

    def test_compute_mode_none_not_counted(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, compute_mode=None))
        t.finalize(success=True)
        assert t.fast_steps == 0
        assert t.think_steps == 0
        assert t.compute_mode_switches == 0

    def test_errors_collected(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.ERROR, error="first error"))
        t.add_step(Step(step_type=StepType.ERROR, error="second error"))
        t.finalize(success=False)
        assert len(t.errors) == 2
        assert t.errors[0]["error"] == "first error"
        assert t.errors[1]["error"] == "second error"

    def test_recovery_actions_collected(self):
        t = Trajectory()
        t.add_step(Step(step_type=StepType.RECOVERY, content="trying different approach"))
        t.finalize(success=True)
        assert len(t.recovery_actions) == 1
        assert t.recovery_actions[0]["content"] == "trying different approach"

    def test_completed_at_set(self):
        t = Trajectory()
        assert t.completed_at is None
        t.finalize(success=True)
        assert t.completed_at is not None

    def test_completed_at_not_overwritten(self):
        t = Trajectory()
        t.completed_at = "2020-01-01T00:00:00Z"
        t.finalize(success=True)
        assert t.completed_at == "2020-01-01T00:00:00Z"


# --- to_dict / to_json / from_dict -------------------------------------------


class TestSerialization:
    def test_to_dict_has_required_fields(self):
        t = Trajectory(task="test task", model_name="m", model_provider="p")
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content="hello"))
        t.finalize(success=True, final_answer="done")
        d = t.to_dict()
        assert d["task"] == "test task"
        assert d["model_name"] == "m"
        assert d["schema_version"] == SCHEMA_VERSION
        assert d["success"] is True
        assert d["final_answer"] == "done"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["content"] == "hello"
        assert d["failure_category"] == "none"

    def test_to_json_parses(self):
        t = Trajectory(task="t")
        t.finalize(success=True)
        j = t.to_json()
        d = json.loads(j)
        assert d["task"] == "t"

    def test_round_trip(self):
        t = Trajectory(
            task="round trip",
            model_name="test-model",
            model_provider="test",
            task_context="ctx",
        )
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content="thinking", completion_tokens=10))
        t.add_step(Step(step_type=StepType.TOOL_CALL, tool_name="shell", tool_args={"cmd": "ls"}))
        t.add_step(Step(step_type=StepType.TOOL_RESULT, tool_result="output", tool_ok=True))
        t.add_step(Step(step_type=StepType.ERROR, error="broke"))
        t.add_step(Step(step_type=StepType.RECOVERY, content="retrying"))
        t.finalize(success=False, final_answer="failed")
        t.failure_category = FailureCategory.INFINITE_LOOP
        t.failure_detail = "repeated same call"

        d = t.to_dict()
        restored = Trajectory.from_dict(d)

        assert restored.task == "round trip"
        assert restored.model_name == "test-model"
        assert restored.success is False
        assert restored.failure_category == FailureCategory.INFINITE_LOOP
        assert restored.failure_detail == "repeated same call"
        assert len(restored.steps) == 5
        assert restored.steps[0].content == "thinking"
        assert restored.steps[1].tool_name == "shell"
        assert restored.steps[2].tool_result == "output"
        assert restored.steps[3].error == "broke"
        assert restored.steps[4].content == "retrying"
        assert restored.total_steps == 5
        assert restored.errors == t.errors
        assert restored.recovery_actions == t.recovery_actions

    def test_from_dict_v2_0_compatibility(self):
        """v2.0 trajectories lack v2.1 fields — from_dict should handle missing fields."""
        v20_data = {
            "trajectory_id": "traj_old",
            "schema_version": "2.0.0",
            "task": "old task",
            "model_name": "old-model",
            "model_provider": "mlxvlm",
            "steps": [
                {"step_type": "model_output", "content": "thinking", "step_id": 0},
                {"step_type": "tool_call", "tool_name": "file_list", "step_id": 1},
            ],
            "success": True,
            "failure_category": "none",
            "final_answer": "done",
            "total_steps": 2,
            "total_tool_calls": 1,
        }
        t = Trajectory.from_dict(v20_data)
        assert t.schema_version == "2.0.0"
        assert t.task == "old task"
        assert t.project == ""  # v2.1 field, missing in v2.0
        assert t.initial_state == {}
        assert t.errors == []
        assert t.recovery_actions == []
        assert len(t.steps) == 2
        assert t.steps[0].content == "thinking"
        assert t.steps[1].tool_name == "file_list"

    def test_from_dict_with_trajectory_class(self):
        d = Trajectory(task="t").to_dict()
        d["trajectory_class"] = "A"
        t = Trajectory.from_dict(d)
        assert t.trajectory_class == TrajectoryClass.A_GOLD

    def test_from_dict_empty(self):
        t = Trajectory.from_dict({})
        assert t.task == ""
        assert t.steps == []
        assert t.success is False

    def test_content_truncated_in_to_dict(self):
        t = Trajectory()
        long_content = "x" * 10000
        t.add_step(Step(step_type=StepType.MODEL_OUTPUT, content=long_content))
        d = t.to_dict()
        assert len(d["steps"][0]["content"]) <= 5000

    def test_tool_result_truncated_in_to_dict(self):
        t = Trajectory()
        long_result = "y" * 10000
        t.add_step(Step(step_type=StepType.TOOL_RESULT, tool_result=long_result))
        d = t.to_dict()
        assert len(d["steps"][0]["tool_result"]) <= 5000


# --- TrajectoryRecorder ------------------------------------------------------


class TestTrajectoryRecorder:
    def test_defaults(self):
        r = TrajectoryRecorder()
        assert r.trajectory.task == ""
        assert r.trajectory.steps == []

    def test_with_params(self):
        r = TrajectoryRecorder(
            task="my task", model_name="m", model_provider="p",
            task_context="ctx", project="proj",
            initial_state={"cwd": "/tmp"},
        )
        assert r.trajectory.task == "my task"
        assert r.trajectory.model_name == "m"
        assert r.trajectory.model_provider == "p"
        assert r.trajectory.task_context == "ctx"
        assert r.trajectory.project == "proj"
        assert r.trajectory.initial_state == {"cwd": "/tmp"}

    def test_record_model_output(self):
        r = TrajectoryRecorder()
        step = r.record_model_output("thinking", compute_mode="fast", completion_tokens=10)
        assert step.step_type == StepType.MODEL_OUTPUT
        assert step.content == "thinking"
        assert step.compute_mode == "fast"
        assert step.completion_tokens == 10
        assert len(r.trajectory.steps) == 1

    def test_record_tool_call(self):
        r = TrajectoryRecorder()
        step = r.record_tool_call("shell", {"command": "ls"}, risk_level="MEDIUM")
        assert step.step_type == StepType.TOOL_CALL
        assert step.tool_name == "shell"
        assert step.tool_args == {"command": "ls"}
        assert step.risk_level == "MEDIUM"

    def test_record_tool_result(self):
        r = TrajectoryRecorder()
        step = r.record_tool_result("output here", ok=True)
        assert step.step_type == StepType.TOOL_RESULT
        assert step.tool_result == "output here"
        assert step.tool_ok is True
        assert step.error is None

    def test_record_tool_result_with_error(self):
        r = TrajectoryRecorder()
        step = r.record_tool_result(None, ok=False, error="failed")
        assert step.tool_ok is False
        assert step.error == "failed"

    def test_record_error(self):
        r = TrajectoryRecorder()
        step = r.record_error("something broke")
        assert step.step_type == StepType.ERROR
        assert step.error == "something broke"

    def test_record_recovery(self):
        r = TrajectoryRecorder()
        step = r.record_recovery("trying different approach")
        assert step.step_type == StepType.RECOVERY
        assert step.content == "trying different approach"
        assert step.compute_mode == "recover"

    def test_record_recovery_custom_mode(self):
        r = TrajectoryRecorder()
        step = r.record_recovery("retry", compute_mode="fast")
        assert step.compute_mode == "fast"

    def test_record_verification(self):
        r = TrajectoryRecorder()
        step = r.record_verification(True, "all checks passed")
        assert step.step_type == StepType.VERIFICATION
        assert step.metadata["passed"] is True
        assert step.content == "all checks passed"

    def test_record_termination(self):
        r = TrajectoryRecorder()
        step = r.record_termination(True, "completed successfully")
        assert step.step_type == StepType.TERMINATION
        assert step.metadata["success"] is True
        assert step.content == "completed successfully"

    def test_finalize_returns_trajectory(self):
        r = TrajectoryRecorder(task="t")
        r.record_model_output("thinking")
        traj = r.finalize(success=True, final_answer="done")
        assert isinstance(traj, Trajectory)
        assert traj.success is True
        assert traj.final_answer == "done"
        assert traj.total_steps == 1

    def test_finalize_sets_time_to_completion(self):
        r = TrajectoryRecorder(task="t")
        traj = r.finalize(success=True)
        assert traj.time_to_completion_s > 0.0

    def test_finalize_increments_approvals(self):
        r = TrajectoryRecorder(task="t")
        r.trajectory.approvals_requested = 2
        r.finalize(success=True)
        # approvals_requested is set externally, finalize should preserve it
        assert r.trajectory.approvals_requested == 2


# --- TrajectoryRecorder persistence ------------------------------------------


class TestTrajectoryPersistence:
    def test_persist_writes_json_file(self, tmp_path: Path):
        r = TrajectoryRecorder(
            task="persisted task",
            model_name="test",
            persist_dir=tmp_path / "trajs",
        )
        r.record_model_output("thinking")
        r.record_tool_call("shell", {"cmd": "ls"})
        r.record_tool_result("output")
        r.finalize(success=True, final_answer="done")

        files = list((tmp_path / "trajs").glob("traj_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["task"] == "persisted task"
        assert data["success"] is True
        assert len(data["steps"]) == 3

    def test_no_persist_when_dir_is_none(self, tmp_path: Path):
        r = TrajectoryRecorder(task="t")
        r.record_model_output("x")
        r.finalize(success=True)
        # no files should be written anywhere
        assert not list(tmp_path.glob("traj_*.json"))

    def test_persist_creates_directory(self, tmp_path: Path):
        persist_dir = tmp_path / "a" / "b" / "c"
        r = TrajectoryRecorder(task="t", persist_dir=persist_dir)
        r.record_model_output("x")
        r.finalize(success=True)
        assert persist_dir.exists()
        assert list(persist_dir.glob("traj_*.json"))

    def test_persist_round_trip(self, tmp_path: Path):
        """A persisted trajectory should be loadable via from_dict."""
        r = TrajectoryRecorder(
            task="round trip",
            model_name="m",
            model_provider="p",
            persist_dir=tmp_path / "trajs",
        )
        r.record_model_output("thinking", compute_mode="fast", completion_tokens=5)
        r.record_tool_call("file_list", {"path": "."}, risk_level="LOW")
        r.record_tool_result("file1.py\nfile2.py", ok=True)
        r.finalize(success=True, final_answer="found 2 files")

        files = list((tmp_path / "trajs").glob("traj_*.json"))
        data = json.loads(files[0].read_text())
        restored = Trajectory.from_dict(data)
        assert restored.task == "round trip"
        assert restored.success is True
        assert restored.final_answer == "found 2 files"
        assert len(restored.steps) == 3
        assert restored.steps[1].tool_name == "file_list"
        assert restored.total_tool_calls == 1
