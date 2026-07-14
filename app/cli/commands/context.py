"""maos context — context quality evaluation commands."""

from __future__ import annotations

import typer
from rich.console import Console

from app.runtime import build_runtime
from app.ui.console import print_panel

console = Console()

context_app = typer.Typer(
    name="context",
    help="Context quality evaluation commands.",
    no_args_is_help=True,
)


@context_app.command("evaluate")
def evaluate() -> None:
    """Evaluate context quality across all dimensions.

    Scores (0-100):
      - Workspace awareness
      - Session continuity
      - Task awareness
      - Knowledge retrieval
      - Preference awareness
      - Activity timeline coverage

    Results are stored in the database for tracking over time.
    """
    rt = build_runtime()
    from app.evaluation import evaluate_context

    result = evaluate_context(rt.db)

    lines = [
        f"Workspace:          {result['workspace']}%",
        f"Session continuity: {result['session']}%",
        f"Task awareness:     {result['tasks']}%",
        f"Knowledge retrieval:{result['knowledge']}%",
        f"Preference awareness:{result['preferences']}%",
        f"Activity coverage:  {result['activity']}%",
        "",
        f"Overall:            {result['overall']}%",
    ]

    # Add details for low-scoring dimensions.
    details = result["details"]
    lines.append("")
    lines.append("Details:")
    for dim, info in details.items():
        if info.get("reason"):
            lines.append(f"  {dim}: {info['reason']}")
        else:
            key_parts = [f"{k}={v}" for k, v in info.items() if k != "types"]
            lines.append(f"  {dim}: {', '.join(key_parts)}")

    # Show trend if previous evaluations exist.
    evals = rt.db.recent_context_evaluations(limit=5)
    if len(evals) > 1:
        prev = evals[1]  # second most recent
        delta = result["overall"] - prev["overall"]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        lines.append(f"\nTrend: {arrow} {delta:+.0f}% from last evaluation")

    print_panel("\n".join(lines), title="Context Quality Report")


@context_app.command("history")
def history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of past evaluations"),
) -> None:
    """Show past context quality evaluations."""
    rt = build_runtime()
    evals = rt.db.recent_context_evaluations(limit=limit)

    if not evals:
        typer.echo("No evaluations recorded. Run `maos context evaluate` first.")
        return

    lines = [
        f"{'Time':>19s}  {'Overall':>7s}  {'WS':>4s}  {'Sess':>4s}  {'Task':>4s}  "
        f"{'Know':>4s}  {'Pref':>4s}  {'Act':>4s}"
    ]
    for e in evals:
        lines.append(
            f"{e['timestamp'][:19]:>19s}  {e['overall']:6d}%  "
            f"{e['workspace']:3d}%  {e['session']:3d}%  {e['tasks']:3d}%  "
            f"{e['knowledge']:3d}%  {e['preferences']:3d}%  {e['activity']:3d}%"
        )

    print_panel("\n".join(lines), title=f"Context Evaluation History ({len(evals)})")
