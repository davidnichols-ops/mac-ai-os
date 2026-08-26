"""Agent loop — the think-act-observe-verify execution cycle.

This is the core of the Adaptive Operator Runtime Substrate. The loop:

    Think  →  model generates a response (text and/or tool calls)
    Act    →  tool executor runs the requested tools
    Observe →  tool results are fed back to the model
    Think  →  model decides next action based on results
    Verify →  model declares completion or loop continues

The loop is model-agnostic. Any model that can produce text and parse
tool calls can operate through this loop. The model is the brain; this
loop is the body.

Safety:
    - Step budget (default 30, configurable)
    - Infinite loop detection (same tool+args 3x = stop)
    - Error budget (default 10 errors = stop)
    - Safety gate on every tool call
    - Timeout (configurable)

Telemetry:
    - Every step recorded in Trajectory
    - Full inference metrics (tokens, time, TPS)
    - Tool call results and errors
    - Compute mode (FAST/THINK) per step
    - Final outcome and failure category
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.safety import SafetyGate
from app.core.state import ExecutionState
from app.core.system_prompt import build_system_prompt
from app.core.tool_executor import ToolExecutor
from app.core.trajectory import FailureCategory, Trajectory, TrajectoryRecorder
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message

log = get_logger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the agent loop."""
    max_steps: int = 30
    max_errors: int = 10
    max_history_messages: int = 20
    timeout_s: float = 300.0  # 5 minutes
    temperature: float = 0.7
    max_tokens: int = 2048
    enable_thinking: bool | None = None  # None = let model self-select
    nudge_after_steps: int = 3  # nudge if no tool calls after N steps
    persist_trajectories: bool = True
    trajectory_dir: Path | None = None
    workspace: str = "."


@dataclass
class LoopResult:
    """The result of an agent loop execution."""
    success: bool
    answer: str
    trajectory: Trajectory
    steps: int = 0
    tool_calls: int = 0
    duration_s: float = 0.0
    failure_category: FailureCategory = FailureCategory.NONE
    failure_detail: str = ""


class AgentLoop:
    """The think-act-observe-verify execution loop.

    This is the body that any sufficiently capable model can inhabit.
    The model is the brain — it decides what to do. The loop provides:
    - Tool discovery and execution
    - Safety enforcement
    - State management
    - Trajectory recording (training data generation)
    - Infinite loop detection
    - Step/error budgets
    """

    def __init__(
        self,
        model: Model,
        executor: ToolExecutor,
        *,
        config: LoopConfig | None = None,
    ):
        self.model = model
        self.executor = executor
        self.config = config or LoopConfig()

    def run(
        self,
        goal: str,
        *,
        context: str | None = None,
        lessons: list[str] | None = None,
        approval_callback: Callable | None = None,
    ) -> LoopResult:
        """Execute the agent loop for a single goal.

        Args:
            goal: The user's task/goal.
            context: Additional context (onboarding, blackboard state).
            lessons: Hard-won lessons to inject.
            approval_callback: Callback for high-risk operation approval.

        Returns:
            LoopResult with the final answer, trajectory, and metrics.
        """
        start_time = time.monotonic()

        # Set up trajectory recorder
        persist_dir = self.config.trajectory_dir if self.config.persist_trajectories else None
        trajectory = TrajectoryRecorder(
            task=goal,
            model_name=self.model.name,
            model_provider=self.model.provider,
            task_context=context or "",
            persist_dir=persist_dir,
        )
        # Update executor's trajectory reference
        self.executor.trajectory = trajectory

        # Set up safety gate with approval callback
        if approval_callback:
            self.executor.safety = SafetyGate(approval_callback=approval_callback)

        # Set up execution state
        state = ExecutionState(
            goal=goal,
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
            max_history_messages=self.config.max_history_messages,
            workspace=self.config.workspace,
        )

        # Build system prompt
        tools = self.executor.available_tools_schema()
        system_prompt = build_system_prompt(
            tools=tools,
            workspace=self.config.workspace,
            context=context,
            lessons=lessons,
        )

        # Initialize messages
        state.add_message(Message(role="system", content=system_prompt))
        state.add_message(Message(role="user", content=goal))

        log.info(
            "agent_loop.start",
            goal=goal[:200],
            tools=len(tools),
            max_steps=self.config.max_steps,
            workspace=self.config.workspace,
        )

        # Main loop
        while not state.finished and not state.is_step_budget_exceeded:
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > self.config.timeout_s:
                trajectory.record_termination(False, f"timeout after {elapsed:.1f}s")
                return self._finalize(
                    trajectory, state, start_time,
                    success=False,
                    answer="",
                    failure_category=FailureCategory.STEP_BUDGET_EXCEEDED,
                    failure_detail=f"timeout after {elapsed:.1f}s",
                )

            state.step += 1
            state.truncate_history()

            # Generate model response
            try:
                response = self.model.chat(
                    state.messages,
                    tools=tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    enable_thinking=self.config.enable_thinking,
                )
            except Exception as e:
                state.errors += 1
                trajectory.record_error(f"model error: {e}")
                log.error("agent_loop.model_error", error=str(e), step=state.step)

                if state.is_error_budget_exceeded:
                    trajectory.record_termination(False, "error budget exceeded")
                    return self._finalize(
                        trajectory, state, start_time,
                        success=False,
                        answer="",
                        failure_category=FailureCategory.MODEL_ERROR,
                        failure_detail=f"error budget exceeded: {e}",
                    )
                continue

            # Record model output
            trajectory.record_model_output(
                response.text,
                compute_mode=response.mode,
                generation_time_s=response.generation_time_s,
                completion_tokens=response.completion_tokens,
                tokens_per_second=response.tokens_per_second,
            )

            log.info(
                "agent_loop.step",
                step=state.step,
                mode=response.mode,
                tool_calls=len(response.tool_calls),
                finish_reason=response.finish_reason,
                tokens=response.completion_tokens,
                gen_time_s=round(response.generation_time_s, 2),
            )

            # Handle tool calls
            if response.has_tool_calls:
                state.add_assistant_message(response.text, response.tool_calls)

                for tc in response.tool_calls:
                    # Infinite loop detection
                    if state.check_infinite_loop(tc.name, tc.arguments):
                        trajectory.record_termination(
                            False,
                            f"infinite loop: {tc.name}({tc.arguments})",
                        )
                        log.warning("agent_loop.infinite_loop", tool=tc.name, step=state.step)
                        return self._finalize(
                            trajectory, state, start_time,
                            success=False,
                            answer="",
                            failure_category=FailureCategory.INFINITE_LOOP,
                            failure_detail=f"repeated action: {tc.name}({tc.arguments})",
                        )

                    # Execute the tool
                    result_text, ok, error = self.executor.execute(tc.name, tc.arguments)
                    state.tool_calls += 1

                    if not ok:
                        state.errors += 1
                        if state.is_error_budget_exceeded:
                            trajectory.record_termination(False, "error budget exceeded")
                            return self._finalize(
                                trajectory, state, start_time,
                                success=False,
                                answer="",
                                failure_category=FailureCategory.ENVIRONMENT_ERROR,
                                failure_detail="error budget exceeded",
                            )

                    # Feed result back to model
                    state.add_tool_result(tc.id, tc.name, result_text)

            else:
                # No tool calls — model gave a final answer
                state.add_assistant_message(response.text)
                state.finished = True
                state.final_answer = response.text

                log.info(
                    "agent_loop.complete",
                    step=state.step,
                    tool_calls=state.tool_calls,
                    duration_s=round(time.monotonic() - start_time, 2),
                )

                return self._finalize(
                    trajectory, state, start_time,
                    success=True,
                    answer=response.text,
                )

            # Nudge if model isn't making progress
            if state.step >= self.config.nudge_after_steps:
                recent = state.messages[-4:]
                has_tool_results = any(m.role == "tool" for m in recent)
                if not has_tool_results and not response.has_tool_calls:
                    state.add_user_message(
                        "You haven't used any tools recently. Use the available "
                        "tools to make progress on the task, or provide your "
                        "final answer if you believe the task is done."
                    )

        # Step budget exceeded
        if not state.finished:
            trajectory.record_termination(False, f"exceeded {self.config.max_steps} steps")
            log.warning("agent_loop.step_budget", steps=state.step, max=self.config.max_steps)
            return self._finalize(
                trajectory, state, start_time,
                success=False,
                answer="",
                failure_category=FailureCategory.STEP_BUDGET_EXCEEDED,
                failure_detail=f"exceeded {self.config.max_steps} steps",
            )

        return self._finalize(
            trajectory, state, start_time,
            success=True,
            answer=state.final_answer,
        )

    def _finalize(
        self,
        trajectory: TrajectoryRecorder,
        state: ExecutionState,
        start_time: float,
        *,
        success: bool,
        answer: str,
        failure_category: FailureCategory = FailureCategory.NONE,
        failure_detail: str = "",
    ) -> LoopResult:
        """Finalize the loop and return the result."""
        duration = time.monotonic() - start_time
        traj = trajectory.finalize(success=success, final_answer=answer)
        traj.failure_category = failure_category
        traj.failure_detail = failure_detail

        return LoopResult(
            success=success,
            answer=answer,
            trajectory=traj,
            steps=state.step,
            tool_calls=state.tool_calls,
            duration_s=duration,
            failure_category=failure_category,
            failure_detail=failure_detail,
        )
