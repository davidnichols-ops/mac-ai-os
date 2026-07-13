"""`maos git <subcommand>` — git intelligence commands.

Subcommands:
  index <repo>     — index a repo's commit history
  log [repo]       — show recent commits from the DB
  changes <repo>   — what changed since yesterday
  files <repo>     — most modified files
  explain <query>  — RAG-based explanation of commits
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel

git_app = typer.Typer(name="git", help="Git intelligence commands.", no_args_is_help=True)


@git_app.command("index")
def git_index(
    repo: str = typer.Argument(..., help="Path to a git repository"),
) -> None:
    """Index a repository's commit history into the vector store and DB."""
    p = Path(repo).expanduser().resolve()
    if not (p / ".git").exists():
        typer.echo(f"not a git repo: {p}", err=True)
        raise typer.Exit(1)

    rt = build_runtime()
    from app.retrieval.indexers.git import GitIndexer

    indexer = GitIndexer(rt.db)
    chunks = indexer.index(str(p))

    # Also index the chunks into the vector store.
    n = 0
    if chunks:
        n = rt.retriever.index(str(p), chunks)

    print_panel(
        f"Repo: {p}\nCommits indexed: {len(chunks)}\nChunks in vector store: {n}\n"
        f"Total commits in DB: {rt.db.commit_count(str(p))}",
        title="Git Index",
    )


@git_app.command("log")
def git_log(
    repo: str = typer.Argument(None, help="Repository path (auto-detected if omitted)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of commits to show"),
) -> None:
    """Show recent commits from the indexed database."""
    rt = build_runtime()
    repo_path = _resolve_repo(rt, repo)
    if not repo_path:
        typer.echo("No repo specified and none indexed. Run `maos git index <repo>` first.", err=True)
        raise typer.Exit(1)

    commits = rt.db.recent_commits(repo_path, limit=limit)
    if not commits:
        typer.echo(f"No commits indexed for {repo_path}. Run `maos git index {repo_path}` first.")
        raise typer.Exit(1)

    lines = []
    for c in commits:
        lines.append(
            f"  {c['hash'][:8]} {c['author_date'][:10]} {c['author']:>12s}  "
            f"{c['message'][:80]}"
        )
    print_panel("\n".join(lines), title=f"Last {len(commits)} commits — {repo_path}")


@git_app.command("changes")
def git_changes(
    repo: str = typer.Argument(None, help="Repository path"),
    days: int = typer.Option(1, "--days", "-d", help="Look back N days"),
) -> None:
    """Show what changed in the last N days."""
    from datetime import datetime, timedelta

    rt = build_runtime()
    repo_path = _resolve_repo(rt, repo)
    if not repo_path:
        typer.echo("No repo found. Run `maos git index <repo>` first.", err=True)
        raise typer.Exit(1)

    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    commits = rt.db.commits_since(repo_path, since)
    if not commits:
        print_panel(f"No commits in the last {days} day(s) for {repo_path}.", title="Changes")
        return

    lines = [f"Commits in the last {days} day(s) — {len(commits)} total:"]
    for c in commits:
        lines.append(
            f"  {c['hash'][:8]} {c['author_date'][:10]} {c['author']:>12s}  "
            f"{c['message'][:80]}  ({c['insertions']}+ / {c['deletions']}-)"
        )
    print_panel("\n".join(lines), title="Changes")


@git_app.command("files")
def git_files(
    repo: str = typer.Argument(None, help="Repository path"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of files to show"),
) -> None:
    """Show most frequently modified files."""
    rt = build_runtime()
    repo_path = _resolve_repo(rt, repo)
    if not repo_path:
        typer.echo("No repo found. Run `maos git index <repo>` first.", err=True)
        raise typer.Exit(1)

    files = rt.db.most_modified_files(repo_path, limit=limit)
    if not files:
        typer.echo("No file data. Run `maos git index <repo>` first.")
        raise typer.Exit(1)

    lines = [f"Most modified files in {repo_path}:"]
    for name, count in files:
        lines.append(f"  {count:3d}x  {name}")
    print_panel("\n".join(lines), title="Files")


@git_app.command("explain")
def git_explain(
    query: str = typer.Argument(..., help="Question about commit history"),
) -> None:
    """Answer a question about commit history using RAG."""
    rt = build_runtime()
    from app.agents.git import GitAgent

    agent = GitAgent(planner=rt.planner, db=rt.db, retriever=rt.retriever)
    result = agent.execute(query)
    if result.ok:
        print_markdown(result.answer)
    else:
        typer.echo(result.error or "failed", err=True)
        raise typer.Exit(1)


def _resolve_repo(rt, repo: str | None) -> str | None:
    if repo:
        return str(Path(repo).expanduser().resolve())
    # Auto-detect: use the first indexed repo.
    rows = rt.db.query("SELECT DISTINCT repo FROM git_commits LIMIT 1")
    if rows:
        return rows[0]["repo"]
    # Try the current directory.
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        return str(cwd)
    return None
