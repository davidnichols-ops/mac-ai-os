"""Evidence objects for the PR governance system.

The core principle: a boolean should represent the RESULT of a verification
process, not REPLACE the verification process. An agent can always claim
`--has-tests=True`; it cannot as easily fake a captured command execution
tied to a specific repository state (commit, working tree, timestamp).

Evidence objects are:
  - inspectable (structured fields, not opaque strings)
  - serializable (to_dict / JSON for machine-readable CLI output)
  - associated with repository state (commit hash, working dir)
  - distinguishable from agent assertions (an evidence object was produced
    by the system executing something; an assertion is just a claim)

Evidence types:
  - CommandEvidence: result of executing a shell command (tests, build, lint)
  - DiffEvidence: result of inspecting the git diff
  - ManualVerificationEvidence: structured manual verification (with required fields)
  - ReviewEvidence: sub-agent or human review (distinct from execution evidence)
  - StaleEvidence: wrapper marking evidence as stale relative to current repo state
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CommandEvidence:
    """Evidence produced by executing a shell command.

    This is the primary evidence type for test/build/lint verification.
    The system executes the command and captures the result — the agent
    does not supply the outcome.

    Attributes:
        command: The exact command executed.
        exit_code: Process exit code (0 = success).
        stdout: Captured stdout (truncated to a reasonable limit).
        stderr: Captured stderr (truncated to a reasonable limit).
        duration_seconds: Wall-clock execution time.
        timestamp: ISO-8601 UTC timestamp of execution.
        working_dir: Absolute path where the command ran.
        commit: Git commit hash at execution time (HEAD).
        dirty: Whether the working tree had uncommitted changes at execution.
        env_keys: Names of relevant environment variables set (values NOT captured).
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timestamp: str
    working_dir: str
    commit: str = ""
    dirty: bool = False
    env_keys: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if the command exited 0."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "command",
            "command": self.command,
            "exit_code": self.exit_code,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "working_dir": self.working_dir,
            "commit": self.commit,
            "dirty": self.dirty,
            "env_keys": self.env_keys,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CommandEvidence:
        return cls(
            command=d["command"],
            exit_code=d["exit_code"],
            stdout=d.get("stdout", ""),
            stderr=d.get("stderr", ""),
            duration_seconds=d.get("duration_seconds", 0.0),
            timestamp=d.get("timestamp", ""),
            working_dir=d.get("working_dir", ""),
            commit=d.get("commit", ""),
            dirty=d.get("dirty", False),
            env_keys=d.get("env_keys", []),
        )


@dataclass
class ManualVerificationEvidence:
    """Evidence from manual verification against real data.

    For bug fixes where automated tests cannot fully prove the behavior,
    the agent must provide structured manual verification:
    - what was tested (the actual command/action)
    - what was observed (the actual output)
    - before/after comparison

    This is NOT a boolean claim. The agent must describe what it actually did.
    The guard checks that the fields are non-empty and that before/after are
    distinguishable (not identical strings).
    """

    action: str  # What was done (e.g., "ran `maos onboard` with stale DB")
    observed: str  # What was observed (actual output)
    before: str  # State before the fix
    after: str  # State after the fix
    timestamp: str = ""
    commit: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def valid(self) -> bool:
        """True if the manual verification has meaningful content."""
        return (
            bool(self.action.strip())
            and bool(self.observed.strip())
            and bool(self.before.strip())
            and bool(self.after.strip())
            and self.before.strip() != self.after.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "manual",
            "action": self.action,
            "observed": self.observed,
            "before": self.before,
            "after": self.after,
            "timestamp": self.timestamp,
            "commit": self.commit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManualVerificationEvidence:
        return cls(
            action=d.get("action", ""),
            observed=d.get("observed", ""),
            before=d.get("before", ""),
            after=d.get("after", ""),
            timestamp=d.get("timestamp", ""),
            commit=d.get("commit", ""),
        )


@dataclass
class ReviewEvidence:
    """Evidence from a review (sub-agent or human).

    A review is useful evidence but is NOT execution evidence. The system
    distinguishes:
      - "sub-agent says it looks correct" → weak, not acceptable as sole evidence
      - "sub-agent reviewed the diff and identified specific risks: X, Y, Z"
        → useful, but still requires execution evidence where execution is possible

    The reviewer must identify SPECIFIC findings, not just agree.
    """

    reviewer: str  # "sub-agent", "human:david", etc.
    findings: list[str]  # Specific findings (risks, issues, confirmations)
    risks_identified: list[str]  # Specific risks called out
    timestamp: str = ""
    commit: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()

    @property
    def is_specific(self) -> bool:
        """True if the review has specific findings (not just agreement)."""
        return len(self.findings) > 0 or len(self.risks_identified) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "review",
            "reviewer": self.reviewer,
            "findings": self.findings,
            "risks_identified": self.risks_identified,
            "timestamp": self.timestamp,
            "commit": self.commit,
        }


# --- Evidence collection (system-executed) ----------------------------------


def _git_head(repo_dir: str) -> tuple[str, bool]:
    """Get the current commit hash and dirty flag for a repo.

    Returns (commit_hash, is_dirty). If git fails, returns ("", False).
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        ).stdout.strip()
        return commit, bool(status)
    except (subprocess.SubprocessError, OSError):
        return "", False


def capture_command(
    command: str,
    *,
    cwd: str | None = None,
    timeout: int = 300,
    env: dict[str, str] | None = None,
    repo_dir: str | None = None,
) -> CommandEvidence:
    """Execute a command and capture evidence of the result.

    This is the evidence-PRODUCING function. The system runs the command,
    not the agent. The agent cannot fake the exit code, stdout, or the
    commit hash at execution time.

    Args:
        command: The shell command to execute.
        cwd: Working directory (defaults to current directory).
        timeout: Maximum execution time in seconds.
        env: Additional environment variables (merged with os.environ).
        repo_dir: Repository directory for git state capture (defaults to cwd).

    Returns:
        CommandEvidence with the captured result.
    """
    work_dir = str(Path(cwd or Path.cwd()).resolve())
    repo = repo_dir or work_dir
    commit, dirty = _git_head(repo)

    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    env_keys = sorted(set(env.keys()) if env else [])

    start = time.monotonic()
    timestamp = datetime.now(UTC).isoformat()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=timeout,
            env=full_env,
        )
        exit_code = result.returncode
        # Truncate output to prevent evidence objects from being unbounded.
        stdout = result.stdout[:50_000]
        stderr = result.stderr[:50_000]
    except subprocess.TimeoutExpired as e:
        exit_code = -1
        stdout = (e.stdout or "")[:50_000] if isinstance(e.stdout, str) else ""
        stderr = f"[TIMEOUT after {timeout}s]\n{(e.stderr or '')[:50_000]}" if isinstance(e.stderr, str) else f"[TIMEOUT after {timeout}s]"
    except (subprocess.SubprocessError, OSError) as e:
        exit_code = -1
        stdout = ""
        stderr = f"[EXECUTION ERROR] {e!s}"

    duration = time.monotonic() - start

    return CommandEvidence(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=round(duration, 3),
        timestamp=timestamp,
        working_dir=work_dir,
        commit=commit,
        dirty=dirty,
        env_keys=env_keys,
    )


def evidence_to_json(evidence: CommandEvidence | ManualVerificationEvidence | ReviewEvidence) -> str:
    """Serialize an evidence object to JSON."""
    return json.dumps(evidence.to_dict(), indent=2)


def load_evidence(json_str: str) -> CommandEvidence | ManualVerificationEvidence | ReviewEvidence:
    """Deserialize an evidence object from JSON.

    The type field determines which class to instantiate.
    """
    d = json.loads(json_str)
    etype = d.get("type", "command")
    if etype == "command":
        return CommandEvidence.from_dict(d)
    if etype == "manual":
        return ManualVerificationEvidence.from_dict(d)
    if etype == "review":
        return ReviewEvidence(
            reviewer=d.get("reviewer", ""),
            findings=d.get("findings", []),
            risks_identified=d.get("risks_identified", []),
            timestamp=d.get("timestamp", ""),
            commit=d.get("commit", ""),
        )
    msg = f"Unknown evidence type: {etype}"
    raise ValueError(msg)


def is_stale(evidence: CommandEvidence | ManualVerificationEvidence, current_commit: str) -> bool:
    """Check if evidence is stale relative to the current repo state.

    Evidence is stale if:
      - it was captured at a different commit than current HEAD, AND
      - the working tree is currently clean (so the commit is meaningful)

    If the working tree is dirty, the evidence commit may not match even
    for valid evidence (the evidence was captured before the final commit),
    so we don't mark it stale in that case.
    """
    ev_commit = getattr(evidence, "commit", "")
    if not ev_commit or not current_commit:
        return False
    return ev_commit != current_commit
