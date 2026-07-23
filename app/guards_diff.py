"""Diff-aware guard — inspects the actual git diff for risk patterns.

The PR readiness system must inspect the actual diff, not merely the PR
description. A one-line change to an authorization boundary may deserve
more scrutiny than a 500-line test addition.

Severity levels:
  - PASS: no concerns
  - WARN: something worth noting but not blocking
  - BLOCK: hard violation (secrets, suspicious credential material)
  - REQUIRES_REVIEW: security/policy/MCP boundary change that needs human eyes

What we inspect:
  - changed file count, line additions/deletions
  - file types (binary, generated artifacts)
  - secrets and secret-like strings
  - debug statements (leftover print/console.log/breakpoint)
  - dependency changes (pyproject.toml, package.json, requirements.txt)
  - security-sensitive paths (auth, permission, trust, policy)
  - MCP-specific: tool schemas, tool descriptions, permission boundaries
  - unrelated files (changes outside the stated scope)

The system does NOT block all risky changes automatically. It distinguishes:
  - automatic BLOCK (secrets, credential material)
  - automatic WARN (debug statements, large diffs, dependency changes)
  - REQUIRES_REVIEW (security boundaries, MCP tool semantics, permissions)
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.guards import GuardResult, Verdict

# --- Patterns ----------------------------------------------------------------

# Secret-like patterns (high-entropy strings, known token formats).
# These are intentionally conservative — false positives are better than
# false negatives for credential material.
SECRET_PATTERNS = [
    # GitHub tokens
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"), "GitHub personal access token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "GitHub fine-grained PAT"),
    # OpenAI / Anthropic
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI API key"),
    (re.compile(r"sk-ant-[A-Za-z0-9-_]{20,}"), "Anthropic API key"),
    # AWS
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key ID"),
    (re.compile(r"aws_secret_access_key\s*[=:]\s*['\"][A-Za-z0-9/+=]{40}['\"]", re.I), "AWS secret access key"),
    # Slack
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    # Google
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "Google API key"),
    # Generic high-entropy assignments (heuristic)
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|pwd)\s*[=:]\s*['\"][A-Za-z0-9+/=_-]{20,}['\"]"), "credential assignment"),
    # Private keys
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----"), "private key block"),
]

# Debug statements that should not be in production code.
DEBUG_PATTERNS = [
    (re.compile(r"^\s*print\s*\(", re.M), "print() statement"),
    (re.compile(r"^\s*console\.log\s*\(", re.M), "console.log() statement"),
    (re.compile(r"^\s*breakpoint\(\s*\)", re.M), "breakpoint() call"),
    (re.compile(r"^\s*import\s+pdb", re.M), "pdb import"),
    (re.compile(r"^\s*debugger\s*;?\s*$", re.M), "debugger statement"),
]

# Security-sensitive path patterns — changes here require review.
SECURITY_PATH_PATTERNS = [
    re.compile(r"(?:^|/)(auth|authentication|authorize|authorization)\b", re.I),
    re.compile(r"(?:^|/)(perm|permission|permissions)\b", re.I),
    re.compile(r"(?:^|/)(trust|trusted|trustcard)\b", re.I),
    re.compile(r"(?:^|/)(policy|policies)\b", re.I),
    re.compile(r"(?:^|/)(security|secure)\b", re.I),
    re.compile(r"(?:^|/)(credential|secret|token)\b", re.I),
    re.compile(r"(?:^|/)(guard|guards)\b", re.I),
    re.compile(r"(?:^|/)(approval|approve)\b", re.I),
    re.compile(r"(?:^|/)(sandbox|escape|boundary)\b", re.I),
]

# MCP-specific path patterns — changes here affect agent behavior.
MCP_PATH_PATTERNS = [
    re.compile(r"(?:^|/)mcp/", re.I),
    re.compile(r"(?:^|/)(tool|tools)[._-]", re.I),
    re.compile(r"(?:^|/)schemas?\b", re.I),
    re.compile(r"(?:^|/)server\b", re.I),
]

# MCP-specific content patterns in diffs — tool description/schema changes.
MCP_CONTENT_PATTERNS = [
    (re.compile(r'name\s*=\s*["\']maos_'), "MCP tool name definition"),
    (re.compile(r'@mcp\.tool', re.M), "MCP tool registration"),
    (re.compile(r'description\s*=\s*["\'][^"\']{20,}["\']'), "tool description string"),
    (re.compile(r'"type"\s*:\s*"(string|number|integer|boolean|object|array)"'), "JSON schema type definition"),
    (re.compile(r'(?:inputSchema|input_schema|parameters)\s*[=:]'), "tool input schema"),
]

# Dependency files — changes affect the supply chain.
DEPENDENCY_FILES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "package-lock.json", "yarn.lock", "Cargo.toml",
    "Cargo.lock", "go.mod", "go.sum", "Gemfile", "Gemfile.lock",
}

# Binary/generated file extensions.
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".so", ".dylib", ".dll", ".exe", ".class", ".pyc", ".onnx", ".pt", ".bin"}


@dataclass
class DiffInfo:
    """Structured information about a git diff."""

    files_changed: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    file_types: dict[str, int] = field(default_factory=dict)  # extension -> count
    binary_files: list[str] = field(default_factory=list)
    dependency_changes: list[str] = field(default_factory=list)
    security_paths: list[str] = field(default_factory=list)
    mcp_paths: list[str] = field(default_factory=list)
    diff_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_changed": self.files_changed,
            "additions": self.additions,
            "deletions": self.deletions,
            "file_types": self.file_types,
            "binary_files": self.binary_files,
            "dependency_changes": self.dependency_changes,
            "security_paths": self.security_paths,
            "mcp_paths": self.mcp_paths,
        }


@dataclass
class DiffFinding:
    """A single finding from diff inspection."""

    severity: str  # "block", "warn", "review", "pass"
    category: str  # "secret", "debug", "dependency", "security_path", "mcp", "binary", "scope"
    file: str
    line: int  # 0 if file-level
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
        }


def get_diff_info(repo_dir: str, base: str = "HEAD") -> DiffInfo:
    """Gather structured information about the current diff.

    Args:
        repo_dir: Path to the git repository.
        base: Base ref to diff against (default: HEAD, i.e., uncommitted changes).
              Use "main" or "origin/main" for branch diffs.

    Returns:
        DiffInfo with structured diff metadata.
    """
    info = DiffInfo()

    # Get the list of changed files.
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base],
            capture_output=True, text=True, cwd=repo_dir, timeout=10,
        )
        files = [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.SubprocessError, OSError):
        # If --name-only fails (e.g., no HEAD yet), try --staged + --others
        try:
            r1 = subprocess.run(
                ["git", "diff", "--name-only", "--staged"],
                capture_output=True, text=True, cwd=repo_dir, timeout=10,
            )
            r2 = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True, text=True, cwd=repo_dir, timeout=10,
            )
            files = [f for f in (r1.stdout + r2.stdout).strip().split("\n") if f]
        except (subprocess.SubprocessError, OSError):
            return info

    info.files_changed = files

    # Get the full diff text (for content inspection).
    try:
        diff_result = subprocess.run(
            ["git", "diff", base],
            capture_output=True, text=True, cwd=repo_dir, timeout=15,
        )
        info.diff_text = diff_result.stdout[:200_000]  # cap at 200KB
    except (subprocess.SubprocessError, OSError):
        pass

    # Get line counts.
    try:
        stat_result = subprocess.run(
            ["git", "diff", "--shortstat", base],
            capture_output=True, text=True, cwd=repo_dir, timeout=10,
        )
        stat = stat_result.stdout
        add_match = re.search(r"(\d+) insertion", stat)
        del_match = re.search(r"(\d+) deletion", stat)
        info.additions = int(add_match.group(1)) if add_match else 0
        info.deletions = int(del_match.group(1)) if del_match else 0
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    # Categorize files.
    for f in files:
        ext = Path(f).suffix.lower()
        info.file_types[ext] = info.file_types.get(ext, 0) + 1

        if ext in BINARY_EXTENSIONS:
            info.binary_files.append(f)
        if Path(f).name in DEPENDENCY_FILES:
            info.dependency_changes.append(f)
        if any(p.search(f) for p in SECURITY_PATH_PATTERNS):
            info.security_paths.append(f)
        if any(p.search(f) for p in MCP_PATH_PATTERNS):
            info.mcp_paths.append(f)

    return info


def inspect_diff(
    repo_dir: str,
    base: str = "HEAD",
    *,
    stated_scope: str = "",
) -> GuardResult:
    """Inspect the git diff for risk patterns.

    Args:
        repo_dir: Path to the git repository.
        base: Base ref to diff against.
        stated_scope: The stated scope of the PR (for unrelated-file detection).
                      If empty, scope checking is skipped.

    Returns:
        GuardResult with findings. BLOCK for secrets, WARN for debug/scope,
        REQUIRES_REVIEW for security/MCP boundary changes.
    """
    info = get_diff_info(repo_dir, base)
    findings: list[DiffFinding] = []

    # 1. Secret detection — scan the diff text for credential material.
    for line_num, line in enumerate(info.diff_text.split("\n"), 1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]  # strip the leading +
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(added):
                findings.append(DiffFinding(
                    severity="block", category="secret",
                    file=_extract_file_from_diff_line(line_num, info.diff_text),
                    line=line_num,
                    detail=f"{label} detected in added line",
                ))

    # 2. Debug statement detection.
    for line_num, line in enumerate(info.diff_text.split("\n"), 1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]
        for pattern, label in DEBUG_PATTERNS:
            if pattern.search(added):
                findings.append(DiffFinding(
                    severity="warn", category="debug",
                    file=_extract_file_from_diff_line(line_num, info.diff_text),
                    line=line_num,
                    detail=label,
                ))

    # 3. MCP content patterns — tool description/schema changes.
    for line_num, line in enumerate(info.diff_text.split("\n"), 1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]
        for pattern, label in MCP_CONTENT_PATTERNS:
            if pattern.search(added):
                findings.append(DiffFinding(
                    severity="review", category="mcp",
                    file=_extract_file_from_diff_line(line_num, info.diff_text),
                    line=line_num,
                    detail=f"{label} — MCP tool semantics changed, requires review",
                ))

    # 4. Security-sensitive path changes.
    for f in info.security_paths:
        findings.append(DiffFinding(
            severity="review", category="security_path",
            file=f, line=0,
            detail="change to security-sensitive path requires review",
        ))

    # 5. MCP path changes.
    for f in info.mcp_paths:
        # Don't double-report if already caught by content patterns.
        if not any(fnd.file == f and fnd.category == "mcp" for fnd in findings):
            findings.append(DiffFinding(
                severity="review", category="mcp",
                file=f, line=0,
                detail="change to MCP-related path requires review",
            ))

    # 6. Dependency changes.
    for f in info.dependency_changes:
        findings.append(DiffFinding(
            severity="warn", category="dependency",
            file=f, line=0,
            detail="dependency file changed — verify no supply-chain risk",
        ))

    # 7. Binary files.
    for f in info.binary_files:
        findings.append(DiffFinding(
            severity="warn", category="binary",
            file=f, line=0,
            detail="binary file in diff — verify it is intentional",
        ))

    # 8. Scope check — files outside the stated scope.
    if stated_scope:
        scope_lower = stated_scope.lower()
        for f in info.files_changed:
            if scope_lower not in f.lower() and not any(
                p.search(f) for p in SECURITY_PATH_PATTERNS + MCP_PATH_PATTERNS
            ):
                findings.append(DiffFinding(
                    severity="warn", category="scope",
                    file=f, line=0,
                    detail=f"file outside stated scope ({stated_scope!r})",
                ))

    # 9. Large diff warning (not blocking — large diffs aren't inherently bad).
    total_lines = info.additions + info.deletions
    if total_lines > 1000:
        findings.append(DiffFinding(
            severity="warn", category="scope",
            file="(diff)", line=0,
            detail=f"large diff: {info.additions}+ {info.deletions}- ({total_lines} lines) — ensure it is not accidentally broad",
        ))

    # Determine verdict.
    blocks = [f for f in findings if f.severity == "block"]
    reviews = [f for f in findings if f.severity == "review"]
    warns = [f for f in findings if f.severity == "warn"]

    if blocks:
        return GuardResult(
            guard="DiffGuard",
            verdict=Verdict.BLOCK,
            reason=f"Diff contains blocked patterns: {'; '.join(b.detail for b in blocks)}",
            flagged_phrases=[f"{b.category}:{b.file}" for b in blocks],
        )

    if reviews:
        return GuardResult(
            guard="DiffGuard",
            verdict=Verdict.WARN,  # WARN, not BLOCK — review is advisory
            reason=f"Diff contains changes requiring review: {'; '.join(r.detail for r in reviews)}",
            flagged_phrases=[f"{r.category}:{r.file}" for r in reviews],
        )

    if warns:
        return GuardResult(
            guard="DiffGuard",
            verdict=Verdict.WARN,
            reason=f"Diff warnings: {'; '.join(w.detail for w in warns)}",
            flagged_phrases=[f"{w.category}:{w.file}" for w in warns],
        )

    return GuardResult(
        guard="DiffGuard",
        verdict=Verdict.PASS,
        reason=f"Diff clean: {len(info.files_changed)} files, {info.additions}+ {info.deletions}-",
    )


def _extract_file_from_diff_line(line_num: int, diff_text: str) -> str:
    """Extract the file path from a diff line number by scanning backwards
    for the most recent '+++' header."""
    lines = diff_text.split("\n")
    for i in range(min(line_num - 1, len(lines) - 1), -1, -1):
        line = lines[i]
        if line.startswith("+++ b/"):
            return line[6:]
        if line.startswith("+++ "):
            return line[4:]
    return "(unknown)"
