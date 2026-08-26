"""Preference format — chosen/rejected pairs for proxy wars.

For training with preference optimization (DPO, KTO, etc.):
  {"prompt": "...", "chosen": "better trajectory", "rejected": "worse trajectory"}

Pairs are generated from:
  1. Successful vs failed trajectories on the same task
  2. Efficient (class A) vs inefficient (class B) trajectories
  3. FAST vs THINK proxy war results
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.trajectory import Trajectory, TrajectoryClass  # noqa: E402


def _trajectory_to_text(traj: Trajectory) -> str:
    """Convert a trajectory to a flat text representation for preference pairs."""
    parts: list[str] = []
    for step in traj.steps:
        if step.content:
            parts.append(step.content)
        if step.tool_name:
            parts.append(f"[tool: {step.tool_name}({step.tool_args})]")
        if step.tool_result:
            parts.append(f"[result: {str(step.tool_result)[:200]}]")
        if step.error:
            parts.append(f"[error: {step.error}]")
    return "\n".join(parts)[:4000]


def create_preference_pairs(
    trajectories: list[Trajectory],
) -> list[dict[str, Any]]:
    """Create preference pairs from a list of trajectories.

    Pairs are created by matching trajectories on the same task:
    - successful vs failed
    - class A (gold) vs class B (inefficient)
    - shorter vs longer (when both succeed)
    """
    pairs: list[dict[str, Any]] = []

    # Group by task
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        by_task[traj.task].append(traj)

    for task, trajs in by_task.items():
        if len(trajs) < 2:
            continue

        successful = [t for t in trajs if t.success]
        failed = [t for t in trajs if not t.success]

        # Pair: successful vs failed
        for s in successful:
            for f in failed:
                pairs.append({
                    "prompt": task,
                    "chosen": _trajectory_to_text(s),
                    "rejected": _trajectory_to_text(f),
                    "metadata": {
                        "chosen_id": s.trajectory_id,
                        "rejected_id": f.trajectory_id,
                        "pair_type": "success_vs_failure",
                    },
                })

        # Pair: gold vs inefficient
        gold = [t for t in successful if t.trajectory_class == TrajectoryClass.A_GOLD]
        inefficient = [t for t in successful if t.trajectory_class == TrajectoryClass.B_CORRECT_INEFFICIENT]
        for g in gold:
            for i in inefficient:
                pairs.append({
                    "prompt": task,
                    "chosen": _trajectory_to_text(g),
                    "rejected": _trajectory_to_text(i),
                    "metadata": {
                        "chosen_id": g.trajectory_id,
                        "rejected_id": i.trajectory_id,
                        "pair_type": "gold_vs_inefficient",
                    },
                })

        # Pair: shorter vs longer (both successful)
        if len(successful) >= 2:
            sorted_by_steps = sorted(successful, key=lambda t: t.total_steps)
            shortest = sorted_by_steps[0]
            longest = sorted_by_steps[-1]
            if shortest.trajectory_id != longest.trajectory_id:
                pairs.append({
                    "prompt": task,
                    "chosen": _trajectory_to_text(shortest),
                    "rejected": _trajectory_to_text(longest),
                    "metadata": {
                        "chosen_id": shortest.trajectory_id,
                        "rejected_id": longest.trajectory_id,
                        "pair_type": "efficient_vs_inefficient",
                        "chosen_steps": shortest.total_steps,
                        "rejected_steps": longest.total_steps,
                    },
                })

    return pairs


def convert_trajectories_to_preference(
    trajectory_dir: Path,
    output_path: Path,
) -> int:
    """Convert trajectories to preference JSONL format.

    Args:
        trajectory_dir: Directory containing traj_*.json files.
        output_path: Output JSONL file path.

    Returns:
        Number of preference pairs written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories: list[Trajectory] = []
    for traj_file in sorted(trajectory_dir.glob("traj_*.json")):
        try:
            data = json.loads(traj_file.read_text())
            traj = Trajectory.from_dict(data)
            trajectories.append(traj)
        except Exception:
            continue

    pairs = create_preference_pairs(trajectories)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return len(pairs)
