"""`maos memory` — capture and inspect personal memories."""

from __future__ import annotations

import json

import typer

from app.runtime import build_runtime

memory_app = typer.Typer(name="memory", help="Capture and inspect personal memories.")


@memory_app.command("add")
def add_memory(
    content: str = typer.Argument(..., help="Thought or fact to remember"),
    title: str = typer.Option("", "--title"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    no_archive: bool = typer.Option(False, "--no-archive"),
) -> None:
    rt = build_runtime()
    result = rt.memory.capture(
        content,
        title=title,
        tags=tuple(value.strip() for value in tags.split(",") if value.strip()),
        source="cli",
        archive=not no_archive,
    )
    if not result.ok:
        typer.echo(result.error, err=True)
        raise typer.Exit(1)
    payload = {
        "memory_id": result.memory_id,
        "content_hash": result.content_hash,
        "indexed": result.indexed,
        "duplicate": result.duplicate,
        "archive": result.archive.__dict__ if result.archive else None,
    }
    typer.echo(json.dumps(payload, indent=2))


@memory_app.command("list")
def list_memories(limit: int = typer.Option(20, "--limit", min=1, max=200)) -> None:
    rt = build_runtime()
    typer.echo(json.dumps(rt.db.list_personal_memories(limit), indent=2, default=str))
