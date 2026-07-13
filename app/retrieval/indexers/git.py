"""Git indexer — extracts commit-level semantic units from a repository.

For each commit, creates a chunk containing:
  - the commit message
  - the files changed
  - the diff stat (insertions/deletions)

Also stores structured metadata in the `git_commits` SQLite table for
fast structured queries ("what changed since yesterday", "most modified files").
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.log import get_logger
from app.retrieval.indexers.base import Indexer

log = get_logger(__name__)


def _git(repo: str, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        log.warning("git.cmd_failed", repo=repo, args=args, error=str(e))
        return ""


class GitIndexer(Indexer):
    """Indexes git commit history as semantic chunks + structured DB rows."""

    name = "git"

    def __init__(self, db, *, max_commits: int = 500):
        self.db = db
        self.max_commits = max_commits

    def supports(self, source: str) -> bool:
        return Path(source, ".git").exists()

    def index(self, source: str) -> list[dict]:
        repo = str(Path(source).resolve())
        if not Path(repo, ".git").exists():
            log.warning("git_indexer.not_a_repo", source=repo)
            return []

        commits = self._parse_log(repo)
        if not commits:
            return []

        chunks: list[dict] = []
        for c in commits:
            # Store structured row.
            self.db.upsert_commit(
                repo=repo,
                hash=c["hash"],
                author=c["author"],
                author_date=c["date"],
                message=c["message"],
                files_changed=c["files"],
                insertions=c["insertions"],
                deletions=c["deletions"],
            )
            # Build a semantic chunk for RAG.
            text = self._commit_to_text(c)
            chunks.append({
                "text": text,
                "metadata": {
                    "source": f"{repo}#{c['hash']}",
                    "commit_hash": c["hash"],
                    "author": c["author"],
                    "date": c["date"],
                    "files": c["files"][:500],  # truncate for metadata
                    "kind": "git_commit",
                    "chunk_index": 0,
                },
            })

        log.info("git_indexer.done", repo=repo, commits=len(chunks))
        return chunks

    def _parse_log(self, repo: str) -> list[dict]:
        """Parse `git log` into structured commit dicts."""
        # Format: hash|||author|||date|||message|||files|||insertions|||deletions
        sep = "|||"
        fmt = sep.join(["%H", "%an", "%aI", "%s", "", "", ""])
        raw = _git(repo, "log", f"--format={fmt}", f"-n {self.max_commits}", "--numstat")
        if not raw:
            return []

        commits: list[dict] = []
        lines = raw.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            parts = line.split(sep)
            if len(parts) < 4:
                i += 1
                continue

            commit_hash = parts[0]
            author = parts[1]
            date = parts[2]
            message = parts[3]

            # Collect numstat lines that follow until the next commit separator.
            files_list: list[str] = []
            insertions = 0
            deletions = 0
            i += 1
            # Skip blank lines between the format line and numstat output.
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines):
                stat_line = lines[i]
                if not stat_line.strip():
                    i += 1
                    break
                # numstat format: insertions deletions filepath
                stat_parts = stat_line.split("\t")
                if len(stat_parts) == 3:
                    try:
                        ins = int(stat_parts[0]) if stat_parts[0] != "-" else 0
                        dels = int(stat_parts[1]) if stat_parts[1] != "-" else 0
                    except ValueError:
                        ins, dels = 0, 0
                    insertions += ins
                    deletions += dels
                    files_list.append(stat_parts[2])
                else:
                    break
                i += 1

            commits.append({
                "hash": commit_hash,
                "author": author,
                "date": date,
                "message": message,
                "files": "\n".join(files_list),
                "insertions": insertions,
                "deletions": deletions,
            })

        return commits

    def _commit_to_text(self, c: dict) -> str:
        files = c["files"].replace("\n", ", ") if c["files"] else "(no files)"
        return (
            f"Commit {c['hash'][:8]}\n"
            f"Author: {c['author']}\n"
            f"Date: {c['date']}\n"
            f"Message: {c['message']}\n"
            f"Files changed ({c['insertions']}+ / {c['deletions']}-): {files}"
        )
