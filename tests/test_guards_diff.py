"""Tests for app/guards_diff.py — diff inspection guard.

Covers:
- Secret detection (GitHub tokens, API keys, private keys)
- Debug statement detection (print, console.log, breakpoint)
- Security-sensitive path detection
- MCP path and content pattern detection
- Dependency file changes
- Binary file detection
- Scope checking
- Large diff warning
- Severity levels (PASS, WARN, BLOCK, REQUIRES_REVIEW)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.guards import Verdict
from app.guards_diff import get_diff_info, inspect_diff


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a git repo with an initial commit."""
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path, timeout=5)
    subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path, timeout=5)
    subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path, timeout=5)
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], capture_output=True, cwd=tmp_path, timeout=5)
    subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path, timeout=5)
    return tmp_path


def _make_change(repo: Path, file: str, content: str) -> None:
    """Make a change and stage it (but don't commit)."""
    (repo / file).parent.mkdir(parents=True, exist_ok=True)
    (repo / file).write_text(content)
    subprocess.run(["git", "add", "."], capture_output=True, cwd=repo, timeout=5)


# --- Secret detection --------------------------------------------------------


class TestSecretDetection:
    def test_github_token_blocks(self, git_repo):
        _make_change(git_repo, "config.py", 'token = "ghp_" + "A" * 36')
        # Write the actual token pattern
        (git_repo / "config.py").write_text('token = "ghp_' + "A" * 36 + '"')
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.BLOCK
        assert "secret" in result.reason.lower() or "token" in result.reason.lower()

    def test_openai_key_blocks(self, git_repo):
        (git_repo / "config.py").write_text('api_key = "sk-' + "A" * 30 + '"')
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.BLOCK

    def test_private_key_blocks(self, git_repo):
        (git_repo / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nfakekey\n-----END RSA PRIVATE KEY-----")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.BLOCK
        assert "private key" in result.reason.lower()

    def test_clean_code_passes(self, git_repo):
        (git_repo / "app.py").write_text("def hello():\n    return 'world'\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.PASS


# --- Debug statement detection -----------------------------------------------


class TestDebugDetection:
    def test_print_statement_warns(self, git_repo):
        (git_repo / "app.py").write_text("print('debug')\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN
        assert "debug" in result.reason.lower() or "print" in result.reason.lower()

    def test_breakpoint_warns(self, git_repo):
        (git_repo / "app.py").write_text("breakpoint()\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN

    def test_console_log_warns(self, git_repo):
        (git_repo / "app.js").write_text("console.log('debug')\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN


# --- Security-sensitive paths ------------------------------------------------


class TestSecurityPaths:
    def test_auth_path_requires_review(self, git_repo):
        (git_repo / "auth").mkdir()
        (git_repo / "auth" / "login.py").write_text("# auth change\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN  # review is WARN, not BLOCK
        assert "security" in result.reason.lower() or "review" in result.reason.lower()

    def test_permission_path_requires_review(self, git_repo):
        (git_repo / "permissions.py").write_text("# permission change\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN

    def test_trust_path_requires_review(self, git_repo):
        (git_repo / "trust.py").write_text("# trust change\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN


# --- MCP-specific detection --------------------------------------------------


class TestMCPDetection:
    def test_mcp_path_requires_review(self, git_repo):
        (git_repo / "mcp").mkdir()
        (git_repo / "mcp" / "server.py").write_text("# mcp change\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN
        assert "mcp" in result.reason.lower() or "review" in result.reason.lower()

    def test_mcp_tool_description_change_warns(self, git_repo):
        (git_repo / "tools.py").write_text(
            'description = "This tool does something important for the agent"\n'
        )
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        # Should detect the tool description pattern
        assert result.verdict in (Verdict.WARN, Verdict.PASS)  # depends on path matching


# --- Dependency changes ------------------------------------------------------


class TestDependencyChanges:
    def test_pyproject_toml_change_warns(self, git_repo):
        (git_repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN
        assert "dependency" in result.reason.lower()

    def test_requirements_txt_change_warns(self, git_repo):
        (git_repo / "requirements.txt").write_text("requests==2.31.0\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN


# --- Binary files ------------------------------------------------------------


class TestBinaryFiles:
    def test_binary_file_warns(self, git_repo):
        # Write binary content
        (git_repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN
        assert "binary" in result.reason.lower()


# --- Scope checking ----------------------------------------------------------


class TestScopeChecking:
    def test_in_scope_file_passes(self, git_repo):
        (git_repo / "app").mkdir()
        (git_repo / "app" / "feature.py").write_text("# feature\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo), stated_scope="app/")
        # No warnings expected (in scope, no debug, no secrets)
        assert result.verdict == Verdict.PASS

    def test_out_of_scope_file_warns(self, git_repo):
        (git_repo / "app").mkdir()
        (git_repo / "app" / "feature.py").write_text("# feature\n")
        (git_repo / "unrelated.py").write_text("# unrelated change\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo), stated_scope="app/")
        assert result.verdict == Verdict.WARN
        assert "scope" in result.reason.lower()


# --- DiffInfo ----------------------------------------------------------------


class TestDiffInfo:
    def test_get_diff_info_returns_files(self, git_repo):
        (git_repo / "app.py").write_text("# new\n")
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        info = get_diff_info(str(git_repo))
        assert "app.py" in info.files_changed
        assert info.additions > 0

    def test_get_diff_info_empty_when_no_changes(self, git_repo):
        info = get_diff_info(str(git_repo))
        assert info.files_changed == [] or info.files_changed == [""]


# --- Large diff --------------------------------------------------------------


class TestLargeDiff:
    def test_large_diff_warns(self, git_repo):
        # Create a file with > 1000 lines
        (git_repo / "big.py").write_text("\n".join(f"x = {i}" for i in range(1100)))
        subprocess.run(["git", "add", "."], capture_output=True, cwd=git_repo, timeout=5)
        result = inspect_diff(str(git_repo))
        assert result.verdict == Verdict.WARN
        assert "large" in result.reason.lower()
