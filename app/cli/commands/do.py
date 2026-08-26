"""`maos do "..."` — run the agent loop with tools to accomplish a task.

This is the primary interface for the Adaptive Operator Runtime Substrate.
The model runs in a think-act-observe-verify loop with access to tools.

Examples:
    maos do "list all Python files in the current directory"
    maos do "read the file src/main.py and explain what it does"
    maos do "run the test suite and report any failures"
    maos do "show the git status and recent commits"
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from app.core.agent_loop import AgentLoop, LoopConfig
from app.core.safety import RiskLevel, SafetyGate
from app.core.tool_executor import ToolExecutor
from app.log import get_logger
from app.runtime import build_runtime

log = get_logger(__name__)
console = Console()


def _print_approval_request(risk: RiskLevel, tool_name: str, args: dict) -> bool:
    """Ask the user to approve a high-risk operation."""
    console.print()
    console.print(Panel(
        f"[bold yellow]APPROVAL REQUIRED[/bold yellow]\n\n"
        f"Tool: {tool_name}\n"
        f"Risk: [bold red]{risk.name}[/bold red]\n"
        f"Args: {args}",
        title="Safety Gate",
        border_style="yellow",
    ))
    try:
        return typer.confirm("Allow this operation?", default=False)
    except (KeyboardInterrupt, EOFError):
        return False


def do(
    task: str = typer.Argument(..., help="The task to accomplish"),
    max_steps: int = typer.Option(30, "--max-steps", help="Maximum tool call steps"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Max tokens per generation"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Working directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full tool results"),
    json_output: bool = typer.Option(False, "--json", help="Output trajectory as JSON"),
    model: str = typer.Option("default", "--model", "-m", help="Which planner model to use: 'default' (base Qwen3.5-9B) or 'operator' (Adaptive Operator v4)"),
) -> None:
    """Run the agent loop to accomplish a task using tools."""
    rt = build_runtime()

    # Select which planner model to use
    if model == "operator" and rt.planner_alt is not None:
        planner = rt.planner_alt
        console.print("[dim]Using: Adaptive Operator v4 (control tokens + tool calls)[/dim]")
    elif model == "operator" and rt.planner_alt is None:
        console.print("[yellow]Warning: planner_alt not configured, falling back to default planner[/yellow]")
        planner = rt.planner
    else:
        planner = rt.planner
        console.print(f"[dim]Using: {planner.name} ({planner.provider})[/dim]")

    # Build the agent loop
    config = LoopConfig(
        max_steps=max_steps,
        max_tokens=max_tokens,
        temperature=temperature,
        workspace=workspace,
        persist_trajectories=True,
        trajectory_dir=rt.settings.resolved_data_dir() / "trajectories",
    )

    safety = SafetyGate(approval_callback=_print_approval_request)
    executor = ToolExecutor(rt.tools, safety=safety)
    loop = AgentLoop(planner, executor, config=config)

    # Gather context (onboarding briefing if available)
    context = None
    try:
        from app.onboarding import build_onboarding_packet, format_onboarding_briefing
        packet = build_onboarding_packet(rt.db, rt.blackboard)
        context = format_onboarding_briefing(packet)
    except Exception:
        pass  # context is optional

    # Trajectory OS is an optional, provider-neutral personal context source.
    # Keep the local MAOS briefing first; append the durable life context when
    # the separate CLI/database is available.
    try:
        from app.integrations.trajectory import TrajectoryContextProvider
        trajectory_context = TrajectoryContextProvider.from_environment().get("daily")
        if trajectory_context:
            context = f"{context}\n\n{trajectory_context}" if context else trajectory_context
    except Exception:
        pass  # Trajectory OS is optional and must never break MAOS execution.

    # Gather lessons
    lessons = None
    try:
        from app.onboarding import list_lessons
        lesson_items = list_lessons(rt.db)
        lessons = [item.get("text", "") for item in lesson_items[:5] if item.get("text")]
    except Exception:
        pass

    console.print("\n[bold cyan]MAOS Agent Loop[/bold cyan]")
    console.print(f"Task: {task}")
    console.print(f"Workspace: {workspace}")
    console.print(f"Max steps: {max_steps}")
    console.print()

    # Run the loop
    result = loop.run(
        task,
        context=context,
        lessons=lessons,
        approval_callback=_print_approval_request,
    )

    if json_output:
        import json
        console.print_json(json.dumps(result.trajectory.to_dict(), default=str))
        return

    # Print results
    console.print()
    if result.success:
        console.print(Panel(
            result.answer,
            title="[bold green]Result[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]FAILED[/bold red]: {result.failure_category.value}\n"
            f"{result.failure_detail}",
            title="Agent Loop Failed",
            border_style="red",
        ))
        if result.answer:
            console.print()
            console.print("[dim]Last output:[/dim]")
            console.print(result.answer)

    # Print metrics
    console.print()
    console.print(
        f"[dim]Steps: {result.steps} | "
        f"Tool calls: {result.tool_calls} | "
        f"Duration: {result.duration_s:.1f}s | "
        f"Trajectory: {result.trajectory.trajectory_id}[/dim]"
    )
