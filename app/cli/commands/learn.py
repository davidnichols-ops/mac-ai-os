"""`maos learn` — learning and procedural memory commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

learn_app = typer.Typer(name="learn", help="Learning / procedural memory commands.", no_args_is_help=True)


@learn_app.command("summary")
def summary() -> None:
    """Show what the system has learned."""
    rt = build_runtime()
    s = rt.learner.learned_summary()

    lines: list[str] = []

    if s["actions"]:
        lines.append("Action success rates:")
        for a in s["actions"]:
            lines.append(f"  {a['action']:>20s}  {a['successes']}/{a['total']}  "
                         f"({a['rate']:.0%})  conf={a['confidence']:.2f}")
        lines.append("")

    if s["workflows"]:
        lines.append("Workflow success rates:")
        for w in s["workflows"]:
            lines.append(f"  {w['workflow']:>25s}  {w['successes']}/{w['total']}  "
                         f"conf={w['confidence']:.2f}")
        lines.append("")

    if s["preferences"]:
        lines.append("Learned preferences:")
        for p in s["preferences"]:
            lines.append(f"  {p['key']:>20s} = {p['value'][:40]}  "
                         f"(conf={p['confidence']:.2f}, {p['source']})")
        lines.append("")

    lines.append(f"Total user_model entries: {s['total_entries']}")
    print_panel("\n".join(lines), title="Learned")


@learn_app.command("action")
def record_action(
    action: str = typer.Argument(..., help="Action name"),
    success: bool = typer.Option(True, "--success/--fail", help="Whether it succeeded"),
    error: str = typer.Option("", "--error", "-e", help="Error message if failed"),
) -> None:
    """Manually record an action outcome for learning."""
    rt = build_runtime()
    rt.learner.record_action_outcome(action, target="", success=success, error=error)
    stats = rt.learner.action_success_rate(action)
    if stats:
        typer.echo(f"Recorded: {action} success={success}")
        typer.echo(f"  Rate: {stats['successes']}/{stats['total']} ({stats['rate']:.0%})")


@learn_app.command("candidate")
def candidate(
    name: str = typer.Argument(..., help="Lesson name (lowercase-hyphenated)"),
    description: str = typer.Argument(..., help="One-sentence description (<=60 chars)"),
    body: str = typer.Argument(..., help="Lesson body (markdown)"),
    category: str = typer.Option("workflow", "--category", "-c",
        help="tooling | workflow | environment | api | architecture"),
    tags: str = typer.Option("", "--tags", "-t", help="Comma-separated tags"),
    confidence: float = typer.Option(0.5, "--confidence", help="Confidence 0-1"),
    source: str = typer.Option("observed", "--source", help="observed | incident | manual"),
    reminder: str = typer.Option("", "--reminder", help="One-line reminder"),
    record: bool = typer.Option(False, "--record", help="Record directly to user_model"),
) -> None:
    """Build and validate a structured lesson candidate.

    Applies the Hermes-derived authoring standards (description <=60 chars,
    frontmatter, source hygiene). By default this only validates and prints
    the candidate; pass --record to write it through the approved
    record_lesson pathway.
    """
    from app.learn import LessonCandidate, validate_lesson

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    cand = LessonCandidate(
        name=name,
        description=description,
        body=body,
        category=category,
        tags=tag_list,
        source=source,
        confidence=confidence,
        reminder=reminder,
        keywords=tag_list,
    )
    validate_lesson(cand)

    lines = [f"Name:        {cand.name}",
             f"Description: {cand.description} ({len(cand.description)} chars)",
             f"Category:    {cand.category}",
             f"Tags:        {', '.join(cand.tags) or '-'}",
             f"Confidence:  {cand.confidence}",
             f"Source:      {cand.source}",
             "", "---", "", cand.body.strip()]
    if cand.warnings:
        lines += ["", "Warnings:"] + [f"  ! {w}" for w in cand.warnings]
    print_panel("\n".join(lines), title="Lesson Candidate")

    if record:
        rt = build_runtime()
        from app.onboarding import record_lesson

        stored = record_lesson(rt.db, **cand.to_record_kwargs())
        typer.echo(f"Recorded lesson '{stored['slug']}' (conf={stored['confidence']})")
    elif cand.warnings:
        typer.echo("\nFix the warnings above, or pass --record to record anyway.")
