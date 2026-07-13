"""`maos watch <path...>` — continuously index directories as files change."""

from __future__ import annotations

import signal
import time

import typer

from app.db import get_db
from app.runtime import build_runtime
from app.ui.console import print_panel


def watch(
    paths: list[str] = typer.Argument(..., help="Directories to watch"),
) -> None:
    """Start a continuous filesystem watcher that incrementally indexes changes."""
    if not paths:
        typer.echo("specify at least one directory to watch", err=True)
        raise typer.Exit(1)

    rt = build_runtime()
    from app.retrieval.pipeline import IndexingPipeline
    from app.retrieval.watcher import ContinuousIndexer

    db = get_db(rt.settings.paths.data_dir + "/maos.db")
    pipeline = IndexingPipeline(
        rt.retriever,
        db,
        chunk_size=rt.settings.retrieval.chunk_size,
        chunk_overlap=rt.settings.retrieval.chunk_overlap,
    )

    watcher = ContinuousIndexer(pipeline, paths)
    watcher.start()

    print_panel(
        f"Watching {len(paths)} director{'y' if len(paths) == 1 else 'ies'}:\n"
        + "\n".join(f"  - {p}" for p in paths)
        + "\n\nPress Ctrl-C to stop.",
        title="Watch",
    )

    def _shutdown(*_):
        watcher.stop()
        raise typer.Exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
