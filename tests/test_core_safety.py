"""Tests for app/core/safety.py — the safety gate.

Covers:
- RiskLevel ordering
- SafetyDecision defaults
- Blocked commands (CRITICAL, always denied)
- High-risk commands (require approval when callback present, denied otherwise)
- Tool-level risk classification (HIGH/CRITICAL tools)
- Auto-approve for LOW and MEDIUM
- Approval callback invocation and error handling
- Violation recording and reset
"""

from __future__ import annotations

import pytest

from app.core.safety import (
    RiskLevel,
    SafetyDecision,
    SafetyGate,
    SafetyViolation,
)

# --- RiskLevel ---------------------------------------------------------------


class TestRiskLevel:
    def test_ordering(self):
        assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_values(self):
        assert RiskLevel.LOW == 0
        assert RiskLevel.MEDIUM == 1
        assert RiskLevel.HIGH == 2
        assert RiskLevel.CRITICAL == 3


# --- SafetyDecision ----------------------------------------------------------


class TestSafetyDecision:
    def test_defaults(self):
        d = SafetyDecision(allowed=True, risk=RiskLevel.LOW)
        assert d.reason == ""
        assert d.requires_approval is False
        assert d.approval_id is None

    def test_full(self):
        d = SafetyDecision(
            allowed=False, risk=RiskLevel.CRITICAL,
            reason="nope", requires_approval=True, approval_id="a1",
        )
        assert d.allowed is False
        assert d.risk == RiskLevel.CRITICAL
        assert d.approval_id == "a1"


# --- Blocked commands --------------------------------------------------------


class TestBlockedCommands:
    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf *",
        "rm -rf .",
        "mkfs /dev/disk0",
        "dd if=/dev/zero of=/dev/disk0",
        "shred /etc/passwd",
        "diskutil eraseDisk JHFS+ Foo /dev/disk0",
        "chmod 777 /",
        "chown -R root /",
    ])
    def test_blocked_command_denied(self, cmd):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": cmd})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.CRITICAL
        assert "blocked command" in decision.reason

    def test_blocked_command_case_insensitive(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "RM -RF /"})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.CRITICAL

    def test_blocked_command_records_violation(self):
        gate = SafetyGate()
        gate.evaluate("shell", {"command": "rm -rf /"})
        assert len(gate.violations) == 1
        v = gate.violations[0]
        assert v.level == RiskLevel.CRITICAL
        assert v.tool_name == "shell"

    def test_blocked_via_run_command_tool(self):
        gate = SafetyGate()
        decision = gate.evaluate("run_command", {"command": "rm -rf ~"})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.CRITICAL

    def test_blocked_via_git_tool_subcmd(self):
        # git tool uses 'subcmd' field — but blocked commands check 'command'/'subcmd'
        gate = SafetyGate()
        decision = gate.evaluate("git", {"subcmd": "rm -rf /"})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.CRITICAL


# --- High-risk commands ------------------------------------------------------


class TestHighRiskCommands:
    @pytest.mark.parametrize("cmd", [
        "rm old_file.txt",
        "rmdir empty_dir",
        "git reset --hard HEAD~3",
        "git push --force origin main",
        "git push -f",
        "git clean -fd",
        "git branch -D feature",
        "git rebase main",
        "sudo apt install foo",
        "pip install requests",
        "npm install lodash",
        "brew install wget",
        "kill -9 1234",
        "killall python",
        "pkill -f worker",
        "curl http://example.com",
        "wget https://example.com/file",
        "scp file host:/tmp",
        "rsync -av src/ dst/",
    ])
    def test_high_risk_with_callback_requires_approval(self, cmd):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("shell", {"command": cmd})
        assert decision.allowed is True
        assert decision.risk == RiskLevel.HIGH
        assert decision.requires_approval is True

    def test_high_risk_without_callback_denied(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "rm old.txt"})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.HIGH
        assert "requires approval" in decision.reason

    def test_safe_command_low_risk(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "ls -la"})
        assert decision.allowed is True
        assert decision.risk == RiskLevel.LOW


# --- Tool-level risk ---------------------------------------------------------


class TestToolLevelRisk:
    def test_critical_tool_without_callback_denied(self):
        gate = SafetyGate()
        decision = gate.evaluate("format_disk", {})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.CRITICAL

    def test_critical_tool_with_callback_requires_approval(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("format_disk", {})
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.risk == RiskLevel.CRITICAL

    def test_high_risk_tool_without_callback_denied(self):
        gate = SafetyGate()
        decision = gate.evaluate("file_delete", {"path": "/tmp/x"})
        assert decision.allowed is False
        assert decision.risk == RiskLevel.HIGH

    def test_high_risk_tool_with_callback_requires_approval(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("git_reset", {"ref": "HEAD~1"})
        assert decision.allowed is True
        assert decision.requires_approval is True
        assert decision.risk == RiskLevel.HIGH

    def test_declared_medium_auto_approved(self):
        gate = SafetyGate()
        decision = gate.evaluate("file_write", {"path": "/tmp/x"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.allowed is True
        assert decision.risk == RiskLevel.MEDIUM
        assert decision.requires_approval is False

    def test_declared_medium_not_auto_approved_when_disabled(self):
        gate = SafetyGate(auto_approve_medium=False)
        decision = gate.evaluate("file_write", {"path": "/tmp/x"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.allowed is False
        assert decision.risk == RiskLevel.MEDIUM

    def test_declared_low_auto_approved(self):
        gate = SafetyGate()
        decision = gate.evaluate("file_read", {"path": "/tmp/x"}, tool_risk=RiskLevel.LOW)
        assert decision.allowed is True
        assert decision.risk == RiskLevel.LOW

    def test_declared_low_not_auto_approved_when_disabled(self):
        gate = SafetyGate(auto_approve_low=False)
        decision = gate.evaluate("file_read", {"path": "/tmp/x"}, tool_risk=RiskLevel.LOW)
        assert decision.allowed is False
        assert decision.risk == RiskLevel.LOW


# --- Approval callback -------------------------------------------------------


class TestApprovalCallback:
    def test_request_approval_returns_true_when_approved(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        assert gate.request_approval(RiskLevel.HIGH, "rm", {"command": "rm x"}) is True

    def test_request_approval_returns_false_when_denied(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: False)
        assert gate.request_approval(RiskLevel.HIGH, "rm", {"command": "rm x"}) is False

    def test_request_approval_returns_false_without_callback(self):
        gate = SafetyGate()
        assert gate.request_approval(RiskLevel.HIGH, "rm", {}) is False

    def test_request_approval_swallows_callback_exception(self):
        def boom(r, t, a):
            raise RuntimeError("callback exploded")
        gate = SafetyGate(approval_callback=boom)
        assert gate.request_approval(RiskLevel.HIGH, "rm", {}) is False


# --- Violations and reset ----------------------------------------------------


class TestViolations:
    def test_violations_empty_by_default(self):
        gate = SafetyGate()
        assert gate.violations == []

    def test_violations_returns_copy(self):
        gate = SafetyGate()
        gate.evaluate("shell", {"command": "rm -rf /"})
        v = gate.violations
        v.clear()
        # mutating the returned list must not affect internal state
        assert len(gate.violations) == 1

    def test_reset_clears_violations(self):
        gate = SafetyGate()
        gate.evaluate("shell", {"command": "rm -rf /"})
        assert len(gate.violations) == 1
        gate.reset()
        assert gate.violations == []

    def test_high_risk_command_does_not_record_violation(self):
        # only hard-blocked commands record violations
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        gate.evaluate("shell", {"command": "rm old.txt"})
        assert gate.violations == []


# --- SafetyViolation dataclass -----------------------------------------------


class TestSafetyViolation:
    def test_default_args(self):
        v = SafetyViolation(level=RiskLevel.HIGH, tool_name="rm", reason="bad")
        assert v.args == {}

    def test_with_args(self):
        v = SafetyViolation(level=RiskLevel.CRITICAL, tool_name="shell", reason="x", args={"k": 1})
        assert v.args == {"k": 1}
