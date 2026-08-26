"""SFT format — supervised fine-tuning for behavioral imitation.

Converts trajectories to OpenAI-compatible conversation format:
  {"messages": [
    {"role": "system", "content": "You are Adaptive Operator."},
    {"role": "user", "content": "problem"},
    {"role": "assistant", "content": "trajectory (thoughts + tool calls + results)"},
    ...
  ]}

The model learns to imitate successful trajectories — selecting the right
tools, making correct arguments, recovering from errors, and verifying
results.

Training philosophy: do NOT maximize memorization. Optimize:
  - tool selection
  - recovery behavior
  - verification
  - compute allocation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.trajectory import StepType, Trajectory  # noqa: E402


def trajectory_to_sft(
    traj: Trajectory,
    *,
    include_thinking: bool = False,
    system_prompt: str = "You are Adaptive Operator.",
) -> dict[str, Any] | None:
    """Convert a single trajectory to SFT conversation format.

    Args:
        traj: The trajectory to convert.
        include_thinking: Whether to include thinking/reasoning blocks.
        system_prompt: The system prompt to use.

    Returns:
        {"messages": [...]} dict, or None if trajectory is not suitable.
    """
    if not traj.steps or not traj.success:
        return None

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Find the initial observation (task) for the user message
    user_content = traj.task
    if traj.initial_state:
        state_lines = ["## Environment State"]
        for k, v in traj.initial_state.items():
            state_lines.append(f"  {k}: {v}")
        user_content = "\n".join(state_lines) + f"\n## Task\n{traj.task}"
    messages.append({"role": "user", "content": user_content})

    # Build assistant messages from steps
    assistant_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for step in traj.steps:
        if step.step_type == StepType.MODEL_OUTPUT:
            content = step.content
            if not include_thinking:
                # Strip thinking blocks
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                content = re.sub(r"</think>", "", content).strip()
            if content:
                assistant_parts.append(content)

        elif step.step_type == StepType.TOOL_CALL and step.tool_name:
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": step.tool_name,
                    "arguments": json.dumps(step.tool_args),
                },
            })

        elif step.step_type == StepType.TOOL_RESULT:
            # Tool results become tool messages
            if assistant_parts or tool_calls:
                # Flush pending assistant message
                msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(assistant_parts)}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
                assistant_parts = []
                tool_calls = []

            result_content = str(step.tool_result) if step.tool_result else ""
            if step.error:
                result_content = f"Error: {step.error}\n{result_content}"
            messages.append({"role": "tool", "content": result_content[:4000]})

        elif step.step_type == StepType.RECOVERY:
            if step.content:
                assistant_parts.append(f"[Recovery] {step.content}")

        elif step.step_type == StepType.VERIFICATION:
            if step.content:
                assistant_parts.append(f"[Verification] {step.content}")

    # Flush remaining assistant content
    if assistant_parts or tool_calls:
        msg = {"role": "assistant", "content": "\n".join(assistant_parts)}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)

    # Final answer
    if traj.final_answer and traj.final_answer not in "\n".join(assistant_parts):
        messages.append({"role": "assistant", "content": traj.final_answer})

    # Need at least system + user + one assistant
    if len(messages) < 3:
        return None

    return {"messages": messages, "metadata": {
        "trajectory_id": traj.trajectory_id,
        "quality_score": traj.quality_score,
        "total_steps": traj.total_steps,
        "total_tool_calls": traj.total_tool_calls,
    }}


def convert_trajectories_to_sft(
    trajectory_dir: Path,
    output_path: Path,
    *,
    include_thinking: bool = False,
    winners_only: bool = True,
    min_quality: float = 0.0,
) -> int:
    """Convert all trajectories in a directory to SFT JSONL.

    Args:
        trajectory_dir: Directory containing traj_*.json files.
        output_path: Output JSONL file path.
        include_thinking: Include thinking blocks in output.
        winners_only: Only include successful trajectories.
        min_quality: Minimum quality_score to include (0.0 = all).

    Returns:
        Number of samples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    for traj_file in sorted(trajectory_dir.glob("traj_*.json")):
        try:
            data = json.loads(traj_file.read_text())
            traj = Trajectory.from_dict(data)
        except Exception:
            skipped += 1
            continue

        if winners_only and not traj.success:
            skipped += 1
            continue
        if traj.quality_score < min_quality:
            skipped += 1
            continue

        sample = trajectory_to_sft(traj, include_thinking=include_thinking)
        if sample:
            with open(output_path, "a") as f:
                f.write(json.dumps(sample) + "\n")
            count += 1

    return count
