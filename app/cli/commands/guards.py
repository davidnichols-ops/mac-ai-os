"""`maos guards` — PR governance system for an AI-agent-managed MCP server.

Commands:
  scan       — Scan text for vibe language (unsupported success claims)
  check      — Run all guards against a text + action context
  stdin      — Scan text from stdin (pipe mode)
  verify     — Execute a command and capture evidence (evidence-PRODUCING)
  diff       — Inspect the git diff for risk patterns
  readiness  — Assess PR readiness (state machine with evidence-gated transitions)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from app.guards import (
    format_guard_report,
    run_all_guards,
    scan_vibes,
    should_block,
)
from app.guards_diff import get_diff_info, inspect_diff
from app.guards_evidence import capture_command, evidence_to_json
from app.guards_readiness import (
    PRContent,
    ReadinessState,
    assess_readiness,
)
from app.runtime import build_runtime

guards_app = typer.Typer(
    name="guards",
    help="PR governance system — evidence-producing guards against vibe overload.",
    no_args_is_help=True,
)

_VERDICT_COLOR = {"pass": "\033[32m", "warn": "\033[33m", "block": "\033[31m"}
_RESET = "\033[0m"


def _color(verdict: str) -> str:
    return _VERDICT_COLOR.get(verdict, "")


@guards_app.command("scan")
def scan_cmd(
    text: str = typer.Argument(..., help="Text to scan for vibe language"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
) -> None:
    """Scan text for unsupported success claims."""
    result = scan_vibes(text)
    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2))
        return
    typer.echo(f"{_color(result.verdict.value)}[{result.verdict.value.upper()}]{_RESET} {result.guard}")
    typer.echo(f"  {result.reason}")
    if result.flagged_phrases:
        typer.echo(f"  Flagged: {', '.join(result.flagged_phrases)}")
    if result.evidence_required:
        typer.echo(f"  Evidence required: {', '.join(result.evidence_required)}")


@guards_app.command("check")
def check_cmd(
    text: str = typer.Argument(..., help="Text to scan"),
    action: str = typer.Option("", "--action", "-a", help="Action description"),
    is_pr: bool = typer.Option(False, "--pr", help="PR-related action"),
    is_bug_fix: bool = typer.Option(False, "--bug-fix", help="Bug fix (requires manual verification)"),
    has_tests: bool = typer.Option(False, "--has-tests", help="Test output provided (LEGACY: prefer --evidence)"),
    has_build: bool = typer.Option(False, "--has-build", help="Build output provided (LEGACY: prefer --evidence)"),
    has_manual: bool = typer.Option(False, "--has-manual", help="Manual verification done (LEGACY: prefer --evidence)"),
    subagent: bool = typer.Option(False, "--subagent", help="Claims sub-agent agreement"),
    empirical: bool = typer.Option(False, "--empirical", help="Has empirical evidence"),
    evidence_file: str = typer.Option("", "--evidence", "-e", help="JSON file with evidence objects (produced by `guards verify`)"),
    repo_dir: str = typer.Option("", "--repo", help="Repository dir for diff inspection"),
    diff_base: str = typer.Option("HEAD", "--diff-base", help="Base ref for diff (default: HEAD)"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
) -> None:
    """Run all guards against a text + action context.

    When --evidence is provided, the system uses evidence objects (produced by
    `guards verify`) instead of self-attested boolean flags. This is the
    preferred mode — evidence objects cannot be faked by the agent.
    """
    evidence = None
    if evidence_file:
        from app.guards_evidence import load_evidence
        ev_path = Path(evidence_file)
        if not ev_path.exists():
            typer.echo(f"Evidence file not found: {evidence_file}", err=True)
            raise typer.Exit(1)
        # The file may contain a single evidence object or a JSON array.
        raw = ev_path.read_text()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                evidence = [load_evidence(json.dumps(item)) for item in data]
            else:
                evidence = [load_evidence(raw)]
        except (json.JSONDecodeError, ValueError) as e:
            typer.echo(f"Failed to parse evidence file: {e}", err=True)
            raise typer.Exit(1) from e

    rt = build_runtime()
    results = run_all_guards(
        rt.db,
        text,
        action_description=action,
        is_pr=is_pr,
        is_bug_fix=is_bug_fix,
        has_test_output=has_tests,
        has_build_output=has_build,
        has_manual_verification=has_manual,
        claims_subagent_agreement=subagent,
        has_empirical_evidence=empirical,
        evidence=evidence,
        repo_dir=repo_dir,
        diff_base=diff_base,
    )
    if json_out:
        typer.echo(json.dumps({
            "blocked": should_block(results)[0],
            "results": [r.to_dict() for r in results],
        }, indent=2))
    else:
        typer.echo(format_guard_report(results))
    blocked, _ = should_block(results)
    if blocked:
        sys.exit(1)


@guards_app.command("stdin")
def stdin_cmd(
    action: str = typer.Option("", "--action", "-a", help="Action description"),
    is_pr: bool = typer.Option(False, "--pr", help="PR-related action"),
    is_bug_fix: bool = typer.Option(False, "--bug-fix", help="Bug fix"),
) -> None:
    """Scan text from stdin (pipe mode)."""
    text = sys.stdin.read()
    if not text.strip():
        typer.echo("No input on stdin.", err=True)
        raise typer.Exit(1)
    rt = build_runtime()
    results = run_all_guards(rt.db, text, action_description=action, is_pr=is_pr, is_bug_fix=is_bug_fix)
    typer.echo(format_guard_report(results))
    blocked, _ = should_block(results)
    if blocked:
        sys.exit(1)


@guards_app.command("verify")
def verify_cmd(
    command: str = typer.Argument(..., help="Command to execute (e.g., 'uv run pytest')"),
    cwd: str = typer.Option("", "--cwd", help="Working directory (default: current)"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout in seconds"),
    save: str = typer.Option("", "--save", help="Save evidence to this JSON file"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
) -> None:
    """Execute a command and capture verification evidence.

    This is the evidence-PRODUCING command. The system runs the command and
    captures: exit code, stdout, stderr, duration, timestamp, working dir,
    git commit, and dirty flag. The agent cannot fake this evidence.

    Examples:
      maos guards verify -- uv run pytest
      maos guards verify -- uv run ruff check app tests
      maos guards verify "npm test" --save test-evidence.json
    """
    work_dir = cwd or str(Path.cwd())
    evidence = capture_command(command, cwd=work_dir, timeout=timeout)

    if save:
        Path(save).write_text(evidence_to_json(evidence))
        typer.echo(f"Evidence saved to {save}")

    if json_out:
        typer.echo(evidence_to_json(evidence))
    else:
        status = "PASS" if evidence.success else "FAIL"
        color = "\033[32m" if evidence.success else "\033[31m"
        typer.echo(f"{color}[{status}]{_RESET} {evidence.command}")
        typer.echo(f"  Exit code: {evidence.exit_code}")
        typer.echo(f"  Duration:  {evidence.duration_seconds}s")
        typer.echo(f"  Commit:    {evidence.commit[:12] if evidence.commit else '(none)'}")
        typer.echo(f"  Dirty:     {evidence.dirty}")
        typer.echo(f"  Timestamp: {evidence.timestamp}")
        if evidence.stdout:
            preview = evidence.stdout[:2000]
            typer.echo(f"  stdout ({len(evidence.stdout)} chars, preview):")
            for line in preview.split("\n")[:20]:
                typer.echo(f"    {line}")
            if len(evidence.stdout) > 2000:
                typer.echo(f"    ... ({len(evidence.stdout) - 2000} more chars)")
        if evidence.stderr:
            preview = evidence.stderr[:2000]
            typer.echo(f"  stderr ({len(evidence.stderr)} chars, preview):")
            for line in preview.split("\n")[:20]:
                typer.echo(f"    {line}")

    if not evidence.success:
        sys.exit(1)


@guards_app.command("diff")
def diff_cmd(
    repo: str = typer.Option("", "--repo", help="Repository dir (default: current)"),
    base: str = typer.Option("HEAD", "--base", help="Base ref (default: HEAD)"),
    scope: str = typer.Option("", "--scope", help="Stated scope for unrelated-file detection"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
) -> None:
    """Inspect the git diff for risk patterns.

    Checks for: secrets, debug statements, dependency changes, security-sensitive
    paths, MCP tool/schema changes, binary files, scope violations, large diffs.

    Severity levels: PASS, WARN, BLOCK, REQUIRES_REVIEW.
    """
    repo_dir = repo or str(Path.cwd())
    result = inspect_diff(repo_dir, base, stated_scope=scope)
    info = get_diff_info(repo_dir, base)

    if json_out:
        typer.echo(json.dumps({
            "verdict": result.verdict.value,
            "reason": result.reason,
            "flagged": result.flagged_phrases,
            "diff_info": info.to_dict(),
        }, indent=2))
    else:
        typer.echo(f"{_color(result.verdict.value)}[{result.verdict.value.upper()}]{_RESET} DiffGuard")
        typer.echo(f"  {result.reason}")
        typer.echo(f"  Files: {len(info.files_changed)}, +{info.additions}/-{info.deletions}")
        if info.security_paths:
            typer.echo(f"  Security paths: {', '.join(info.security_paths)}")
        if info.mcp_paths:
            typer.echo(f"  MCP paths: {', '.join(info.mcp_paths)}")
        if info.dependency_changes:
            typer.echo(f"  Dependencies: {', '.join(info.dependency_changes)}")
        if info.binary_files:
            typer.echo(f"  Binary files: {', '.join(info.binary_files)}")
        if result.flagged_phrases:
            typer.echo(f"  Flagged: {', '.join(result.flagged_phrases)}")

    if result.verdict.value == "block":
        sys.exit(1)


@guards_app.command("readiness")
def readiness_cmd(
    repo: str = typer.Option("", "--repo", help="Repository dir (default: current)"),
    is_bug_fix: bool = typer.Option(False, "--bug-fix", help="Bug fix (requires manual verification)"),
    test_evidence: str = typer.Option("", "--test-evidence", help="JSON file with test evidence (from `guards verify --save`)"),
    build_evidence: str = typer.Option("", "--build-evidence", help="JSON file with build/lint evidence"),
    lint_evidence: str = typer.Option("", "--lint-evidence", help="JSON file with lint evidence"),
    manual_evidence: str = typer.Option("", "--manual-evidence", help="JSON file with manual verification evidence"),
    review_evidence: str = typer.Option("", "--review-evidence", help="JSON file with review evidence"),
    pr_description: str = typer.Option("", "--pr-description", help="Markdown file with PR description"),
    diff_base: str = typer.Option("HEAD", "--diff-base", help="Base ref for diff (default: HEAD)"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output"),
) -> None:
    """Assess PR readiness using the state machine.

    States: DRAFT → UNDERSTOOD → VERIFIED → PR_READY → CI_READY → MERGE_READY

    Each transition requires explicit evidence. The agent cannot skip from
    "I changed code" to "Open PR" merely by asserting that verification occurred.

    Examples:
      maos guards verify -- uv run pytest --save test.json
      maos guards verify -- uv run ruff check app tests --save lint.json
      maos guards readiness --test-evidence test.json --lint-evidence lint.json --pr-description pr.md
    """
    import subprocess

    from app.guards_evidence import ManualVerificationEvidence, ReviewEvidence, load_evidence

    repo_dir = repo or str(Path.cwd())

    # Load evidence files.
    def _load_evidence(path: str):
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            typer.echo(f"Evidence file not found: {path}", err=True)
            raise typer.Exit(1)
        return load_evidence(p.read_text())

    test_ev = _load_evidence(test_evidence)
    build_ev = _load_evidence(build_evidence)
    lint_ev = _load_evidence(lint_evidence)
    manual_ev = _load_evidence(manual_evidence) if manual_evidence else None
    review_ev = _load_evidence(review_evidence) if review_evidence else None

    # Load PR description.
    content = None
    if pr_description:
        p = Path(pr_description)
        if not p.exists():
            typer.echo(f"PR description file not found: {pr_description}", err=True)
            raise typer.Exit(1)
        content = PRContent.from_markdown(p.read_text())

    # Get current commit.
    try:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir, timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        current_commit = ""

    report = assess_readiness(
        repo_dir=repo_dir,
        content=content,
        is_bug_fix=is_bug_fix,
        test_evidence=test_ev,
        build_evidence=build_ev,
        lint_evidence=lint_ev,
        manual_evidence=manual_ev if isinstance(manual_ev, ManualVerificationEvidence) else None,
        review_evidence=review_ev if isinstance(review_ev, ReviewEvidence) else None,
        current_commit=current_commit,
        diff_base=diff_base,
    )

    if json_out:
        typer.echo(report.to_json())
    else:
        state_color = {
            ReadinessState.DRAFT: "\033[31m",
            ReadinessState.UNDERSTOOD: "\033[33m",
            ReadinessState.VERIFIED: "\033[33m",
            ReadinessState.PR_READY: "\033[32m",
            ReadinessState.CI_READY: "\033[32m",
            ReadinessState.MERGE_READY: "\033[32m",
        }
        color = state_color.get(report.state, "")
        typer.echo(f"{color}[{report.state.value.upper()}]{_RESET} PR Readiness")
        typer.echo()
        for check in report.checks:
            check_color = {"pass": "\033[32m", "warn": "\033[33m", "block": "\033[31m", "missing": "\033[31m", "stale": "\033[33m"}.get(check.status, "")
            typer.echo(f"  {check_color}[{check.status.upper()}]{_RESET} {check.name}: {check.detail}")
        if report.blockers:
            typer.echo()
            typer.echo("Blockers:")
            for b in report.blockers:
                typer.echo(f"  - {b}")
        if report.next_steps:
            typer.echo()
            typer.echo("Next steps:")
            for s in report.next_steps:
                typer.echo(f"  - {s}")

    if not report.is_pr_ready:
        sys.exit(1)
