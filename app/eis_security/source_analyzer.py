"""Conservative source pattern analysis; findings require manual confirmation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSignal:
    path: str
    line: int
    pattern: str
    excerpt: str
    rationale: str
    status: str = "suspected_weakness"

PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("python-eval", r"\b(?:eval|exec)\s*\(", "dynamic code execution requires an explicit trust-boundary review"),
    ("shell-true", r"shell\s*=\s*True", "shell execution can turn data flow into command execution"),
    ("unsafe-deserialization", r"\b(?:pickle\.load|yaml\.load)\s*\(", "deserialization may execute or construct attacker-controlled objects"),
    ("hardcoded-secret", r"(?i)(?:api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}", "credential-like material should not be embedded in source"),
    ("path-boundary", r"\b(?:open|Path)\s*\([^\n]*(?:request|input|argument|path)", "user-controlled paths require traversal and symlink validation"),
)


def analyze_source(repository: Path) -> list[SourceSignal]:
    signals: list[SourceSignal] = []
    extensions = {".py", ".js", ".ts", ".tsx", ".rs", ".go", ".swift"}
    for path in repository.rglob("*"):
        if not path.is_file() or path.suffix not in extensions or ".git" in path.parts:
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            for name, expression, rationale in PATTERNS:
                if re.search(expression, line):
                    signals.append(SourceSignal(str(path.relative_to(repository)), number, name, line.strip()[:240], rationale))
    return signals
