"""Compute policy format — FAST/THINK/VERIFY/RECOVER classification.

Trains the model to choose the right compute mode for each task:
  {"task": "...", "mode": "FAST|THINK|VERIFY|RECOVER", "justification": "..."}

This is the compute allocation policy that lets the model self-select
how much reasoning to allocate based on task difficulty and observed evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.trajectory import ComputeMode, Trajectory  # noqa: E402


def _determine_mode(traj: Trajectory) -> str | None:
    """Determine the dominant compute mode for a trajectory."""
    mode_counts: dict[str, int] = {}
    for step in traj.steps:
        if step.compute_mode:
            mode_counts[step.compute_mode] = mode_counts.get(step.compute_mode, 0) + 1

    if not mode_counts:
        # Infer from trajectory characteristics
        if traj.recovery_actions:
            return ComputeMode.RECOVER.value
        if traj.verification:
            return ComputeMode.VERIFY.value
        if traj.total_steps > 5 or traj.think_steps > 0:
            return ComputeMode.THINK.value
        return ComputeMode.FAST.value

    return max(mode_counts, key=mode_counts.get)


def _generate_justification(traj: Trajectory, mode: str) -> str:
    """Generate a human-readable justification for the compute mode."""
    if mode == ComputeMode.FAST.value:
        if traj.success:
            return "Task is straightforward. Direct tool use is sufficient."
        return "Task appeared simple but failed. May need more reasoning."
    if mode == ComputeMode.THINK.value:
        if traj.success:
            return "Task requires careful analysis. Extended reasoning before action is beneficial."
        return "Task is complex. Extended reasoning was applied but may need more steps."
    if mode == ComputeMode.RECOVER.value:
        return "Task encountered errors. Recovery mode to reassess and try a different approach."
    if mode == ComputeMode.VERIFY.value:
        return "Task requires verification of completed work before declaring success."
    return ""


def trajectory_to_policy_example(traj: Trajectory) -> dict[str, Any] | None:
    """Convert a trajectory to a compute policy training example.

    Returns:
        {"task": "...", "mode": "FAST|THINK|VERIFY|RECOVER",
         "justification": "...", "metadata": {...}}
        or None if trajectory is not suitable.
    """
    if not traj.task:
        return None

    mode = _determine_mode(traj)
    if mode is None:
        return None

    justification = _generate_justification(traj, mode)

    return {
        "task": traj.task,
        "mode": mode,
        "justification": justification,
        "metadata": {
            "trajectory_id": traj.trajectory_id,
            "success": traj.success,
            "total_steps": traj.total_steps,
            "total_tool_calls": traj.total_tool_calls,
            "fast_steps": traj.fast_steps,
            "think_steps": traj.think_steps,
            "recover_steps": traj.recover_steps,
            "verify_steps": traj.verify_steps,
            "errors": len(traj.errors),
            "recovery_actions": len(traj.recovery_actions),
        },
    }


def convert_trajectories_to_compute_policy(
    trajectory_dir: Path,
    output_path: Path,
) -> int:
    """Convert trajectories to compute policy JSONL format.

    Args:
        trajectory_dir: Directory containing traj_*.json files.
        output_path: Output JSONL file path.

    Returns:
        Number of policy examples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    for traj_file in sorted(trajectory_dir.glob("traj_*.json")):
        try:
            data = json.loads(traj_file.read_text())
            traj = Trajectory.from_dict(data)
        except Exception:
            continue

        example = trajectory_to_policy_example(traj)
        if example:
            with open(output_path, "a") as f:
                f.write(json.dumps(example) + "\n")
            count += 1

    return count
