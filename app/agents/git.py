"""Git agent — answers questions about repository history.

Combines two approaches:
  1. Structured SQLite queries for factual questions ("what changed since
     yesterday", "most modified files", "recent commits")
  2. RAG over indexed commit chunks for explanatory questions ("explain my
     last five commits", "what was the TensorRT fix about")

The agent decides which approach to use based on the question type, and can
combine both: run a structured query, then use the results as context for
the planner to synthesize an answer.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from app.agents.base import Agent, AgentResult
from app.db import Database
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message

log = get_logger(__name__)

_TIME_RE = re.compile(
    r"\b(since|last|past|over the)\s+(today|yesterday|(\d+)\s*(days?|weeks?|months?|hours?))",
    re.IGNORECASE,
)
_RECENT_RE = re.compile(r"\b(last|recent)\s+(\d+)?\s*commits?", re.IGNORECASE)
_MOST_MODIFIED_RE = re.compile(
    r"\b(most|frequently|top).*(modified|changed|touched).*files?"
    r"|(modified|changed|touched).*files?.*(most|frequently|top)"
    r"|files?.*(modified|changed|touched).*(most|frequently|top)",
    re.IGNORECASE,
)
_EXPLAIN_RE = re.compile(r"\b(explain|describe|summarize|what).*(commit|change|did I do|was)", re.IGNORECASE)


class GitAgent(Agent):
    name = "git"
    description = (
        "Answers questions about git repository history: what changed, when, "
        "by whom, most modified files, commit explanations."
    )

    def __init__(self, *, planner: Model | None = None, db: Database | None = None, retriever=None):
        super().__init__(planner=planner, retriever=retriever)
        self.db = db

    def plan(self, goal: str) -> list[str]:
        if self._is_structured(goal):
            return ["Run structured SQLite query", "Format results"]
        return ["Retrieve relevant commits via RAG", "Synthesize explanation"]

    def execute(self, goal: str) -> AgentResult:
        if not self.db:
            return AgentResult(ok=False, error="git agent needs a database connection")

        # Try to detect which repo the user is asking about.
        repo = self._detect_repo(goal)

        if self._is_structured(goal):
            return self._structured_answer(goal, repo)
        return self._rag_answer(goal, repo)

    def health(self) -> bool:
        return self.db is not None

    # --- question classification ---------------------------------------------

    def _is_structured(self, goal: str) -> bool:
        if _TIME_RE.search(goal):
            return True
        if _MOST_MODIFIED_RE.search(goal):
            return True
        if _RECENT_RE.search(goal) and not _EXPLAIN_RE.search(goal):
            return True
        return False

    def _detect_repo(self, goal: str) -> str | None:
        """Try to find a repo path mentioned in the goal, else use the first indexed repo."""
        # Look for path-like strings in the goal.
        path_match = re.search(r"(/[\w/.\-]+)", goal)
        if path_match:
            candidate = path_match.group(1)
            if self.db.commit_count(candidate) > 0:
                return candidate
        # Fall back to any repo that has commits.
        rows = self.db.query("SELECT DISTINCT repo FROM git_commits LIMIT 1")
        if rows:
            return rows[0]["repo"]
        return None

    # --- structured queries --------------------------------------------------

    def _structured_answer(self, goal: str, repo: str | None) -> AgentResult:
        if not repo:
            return AgentResult(ok=True, answer="No git repository has been indexed yet. Run `maos git index <repo>` first.")

        # Parse time range.
        since = self._parse_since(goal)
        if since:
            commits = self.db.commits_since(repo, since)
            if not commits:
                return AgentResult(ok=True, answer=f"No commits in {repo} since {since}.")
            text = self._format_commits(commits, f"Commits since {since}")
            return AgentResult(ok=True, answer=text, data={"repo": repo, "commits": len(commits)})

        # Most modified files.
        if _MOST_MODIFIED_RE.search(goal):
            files = self.db.most_modified_files(repo, limit=10)
            if not files:
                return AgentResult(ok=True, answer=f"No file modification data for {repo}.")
            lines = [f"  {count:3d}x  {name}" for name, count in files]
            return AgentResult(
                ok=True,
                answer=f"Most modified files in {repo}:\n" + "\n".join(lines),
                data={"repo": repo, "files": files},
            )

        # Recent commits.
        limit_match = _RECENT_RE.search(goal)
        limit = 10
        if limit_match and limit_match.group(2):
            try:
                limit = int(limit_match.group(2))
            except ValueError:
                pass
        commits = self.db.recent_commits(repo, limit=limit)
        if not commits:
            return AgentResult(ok=True, answer=f"No commits indexed for {repo}.")
        text = self._format_commits(commits, f"Last {limit} commits")
        return AgentResult(ok=True, answer=text, data={"repo": repo, "commits": len(commits)})

    def _parse_since(self, goal: str) -> str | None:
        """Parse a time reference from the goal into an ISO date string."""
        now = datetime.now(UTC)
        lower = goal.lower()

        if "today" in lower:
            return now.strftime("%Y-%m-%d")
        if "yesterday" in lower:
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")

        m = re.search(r"(\d+)\s*(days?|weeks?|months?|hours?)", lower)
        if m:
            n = int(m.group(1))
            unit = m.group(2).rstrip("s")
            if unit == "day":
                return (now - timedelta(days=n)).isoformat()
            if unit == "week":
                return (now - timedelta(weeks=n)).isoformat()
            if unit == "month":
                return (now - timedelta(days=n * 30)).isoformat()
            if unit == "hour":
                return (now - timedelta(hours=n)).isoformat()
        return None

    def _format_commits(self, commits: list, title: str) -> str:
        lines = [title + ":"]
        for c in commits:
            lines.append(
                f"  {c['hash'][:8]} {c['author_date'][:10]} {c['author']:>12s}  "
                f"{c['message'][:80]}"
            )
        return "\n".join(lines)

    # --- RAG answers ---------------------------------------------------------

    def _rag_answer(self, goal: str, repo: str | None) -> AgentResult:
        if not self.retriever or not self.planner:
            # Fall back to structured if we can't do RAG.
            if repo:
                commits = self.db.recent_commits(repo, limit=10)
                if commits:
                    return AgentResult(
                        ok=True,
                        answer=self._format_commits(commits, "Recent commits"),
                        data={"repo": repo, "commits": len(commits)},
                    )
            return AgentResult(ok=False, error="git agent needs a retriever and planner for explanatory questions")

        results = self.retriever.search(goal, top_k=8, where={"kind": "git_commit"})
        git_results = results
        if not git_results:
            # Fallback: unfiltered search.
            results = self.retriever.search(goal, top_k=8)
            git_results = [r for r in results if r.metadata.get("kind") == "git_commit"]
            if not git_results:
                git_results = results
        if not git_results:
            return AgentResult(ok=True, answer="No indexed git history matched the query.")

        context = "\n\n".join(
            f"[{r.metadata.get('commit_hash', '?')[:8]}] (score={r.score:.2f})\n{r.text}"
            for r in git_results[:6]
        )
        system = (
            "You are a git intelligence agent. Answer the question using the provided "
            "commit records. Reference commit hashes when relevant. Be concise and factual."
        )
        user = f"Commit records:\n{context}\n\nQuestion: {goal}"
        resp = self.planner.chat([Message("system", system), Message("user", user)])
        return AgentResult(
            ok=True,
            answer=resp.text,
            data={"sources": [r.source for r in git_results[:6]]},
        )
