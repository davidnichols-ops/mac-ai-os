"""`maos office` — CEO portfolio vision and delegation."""

from __future__ import annotations

from pathlib import Path

import typer

from app.ceo_office import delegate, generate_vision, render_vision
from app.runtime import build_runtime
from app.ui.console import print_panel

office_app = typer.Typer(
    name="office",
    help="CEO portfolio vision and delegation.",
    no_args_is_help=True,
)


def _default_output() -> Path:
    return Path("docs") / "ceo" / "portfolio-vision.md"


@office_app.command("vision")
def vision(
    output: str = typer.Option(str(_default_output()), "--output", "-o"),
) -> None:
    """Generate the evidence-backed portfolio vision and agent handoff."""
    rt = build_runtime()
    result = generate_vision(rt.db, output)
    print_panel(render_vision(result), title="MAOS CEO Office")
    typer.echo(f"Saved: {Path(output).expanduser()}")


@office_app.command("delegate")
def delegate_work(
    output: str = typer.Option(str(_default_output()), "--output", "-o"),
) -> None:
    """Generate the vision and materialize its delegation queue as tasks."""
    rt = build_runtime()
    result = generate_vision(rt.db, output)
    task_ids = delegate(rt.db, result)
    typer.echo(f"Created {len(task_ids)} delegation task(s): {', '.join(map(str, task_ids))}")
    typer.echo(f"Brief: {Path(output).expanduser()}")
