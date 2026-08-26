"""Device-wide RAG lifecycle commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel

device_rag_app = typer.Typer(
    name="device-rag",
    help="Safely index and search the user-controlled knowledge on this Mac.",
    no_args_is_help=True,
)


@device_rag_app.command("sync")
def sync(
    roots: list[str] = typer.Option(
        None,
        "--root",
        "-r",
        help="Root to index (repeatable). Defaults to the home folder with safety exclusions.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-index unchanged files."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan without changing either index."),
    max_files: int | None = typer.Option(
        None, "--max-files", min=1, help="Bound this run for a small initial slice."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Incrementally reconcile files, vectors, full-text chunks, and deletions."""
    rt = build_runtime()
    from app.retrieval.device import build_device_rag

    service = build_device_rag(rt, roots=roots or None)
    stats = service.sync(force=force, dry_run=dry_run, max_files=max_files)
    if json_output:
        typer.echo(json.dumps(stats.to_dict(), indent=2))
        return
    print_panel(
        f"Roots: {', '.join(stats.roots)}\n"
        f"Discovered: {stats.discovered}\n"
        f"Indexed: {stats.indexed}\n"
        f"Unchanged: {stats.unchanged}\n"
        f"Deleted: {stats.deleted}\n"
        f"Chunks added: {stats.chunks}\n"
        f"Errors: {stats.errors}\n"
        f"Safety skips: {sum(stats.skipped.values())}\n"
        f"Dry run: {stats.dry_run}\n"
        f"Limited: {stats.limited}",
        title="Device RAG Sync",
    )


@device_rag_app.command("status")
def status(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show device index coverage, freshness, and active safety policy."""
    rt = build_runtime()
    from app.retrieval.device import build_device_rag

    report = build_device_rag(rt).status()
    if json_output:
        typer.echo(json.dumps(report, indent=2, default=str))
        return
    last = report.get("last_sync") or {}
    print_panel(
        f"Files: {report['files']}\n"
        f"Hybrid chunks: {report['chunks']}\n"
        f"Semantic collection chunks: {report['semantic_chunks']}\n"
        f"Indexed bytes: {report['bytes']}\n"
        f"Last sync: {last.get('finished_at', 'never')}\n"
        f"Catalog: {report['catalog']}\n"
        "Sensitive files: excluded",
        title="Device RAG Status",
    )


@device_rag_app.command("search")
def search(
    query: str = typer.Argument(..., help="Exact or semantic query."),
    top_k: int = typer.Option(8, "--top-k", min=1, max=50),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Run hybrid semantic + full-text retrieval with source paths."""
    rt = build_runtime()
    from app.retrieval.device import build_device_rag

    results = build_device_rag(rt).search(query, top_k=top_k)
    payload = {
        "query": query,
        "count": len(results),
        "results": [
            {
                "text": result.text,
                "source": result.source,
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in results
        ],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if not results:
        print_panel("No matching indexed evidence.", title="Device RAG")
        raise typer.Exit(1)
    for index, result in enumerate(results, start=1):
        snippet = result.text[:700].replace("\n", " ")
        print_panel(
            f"{result.source}\nscore={result.score:.5f}\n{snippet}",
            title=f"Result {index}",
        )


@device_rag_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question grounded in indexed device evidence."),
    top_k: int = typer.Option(8, "--top-k", min=1, max=30),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Answer with inline numeric citations and explicit uncertainty."""
    rt = build_runtime()
    from app.retrieval.device import build_device_rag

    model = rt.qa_model or rt.planner
    answer = build_device_rag(rt).answer(question, model, top_k=top_k)
    if json_output:
        typer.echo(json.dumps(answer, indent=2, default=str))
        return
    sources = "\n".join(
        f"[{source['id']}] {source['path']}" for source in answer["sources"]
    )
    if sources:
        print_panel(sources, title="Sources")
    print_markdown(answer["answer"])


@device_rag_app.command("policy")
def policy() -> None:
    """Explain what broad device coverage includes and excludes."""
    from app.retrieval.device import DeviceIndexPolicy

    active = DeviceIndexPolicy()
    print_panel(
        "Default root: " + str(Path.home()) + "\n"
        f"Text/document types: {len(active.allowed_extensions)}\n"
        f"Maximum regular file: {active.max_file_bytes // (1024 * 1024)} MB\n"
        f"Maximum PDF: {active.max_pdf_bytes // (1024 * 1024)} MB\n"
        "Excluded: macOS Library/system data, hidden folders, credentials, "
        "databases, caches, dependencies, builds, model/vector data, and binaries.",
        title="Device RAG Safety Policy",
    )
