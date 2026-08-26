"""Safety gate tests — verify risk levels and approval enforcement.

Tests:
- LOW: file_read, file_list (auto-approved)
- MEDIUM: file_write, shell (auto-approved)
- HIGH: rm, git reset --hard (requires approval)
- CRITICAL: rm -rf / (always blocked)
"""

from __future__ import annotations

from app.core.agent_loop import AgentLoop
from app.core.safety import RiskLevel, SafetyGate
from app.models.types import ModelResponse, ToolCall
from tests.operator_runtime.conftest import (
    ScriptedModel,
    make_text_response,
    make_tool_response,
)


class TestRiskLevelClassification:
    """Verify tools are classified into the correct risk levels."""

    def test_file_read_is_low_risk(self):
        gate = SafetyGate()
        decision = gate.evaluate("file_read", {"path": "/tmp/test"}, tool_risk=RiskLevel.LOW)
        assert decision.risk == RiskLevel.LOW
        assert decision.allowed

    def test_file_list_is_low_risk(self):
        gate = SafetyGate()
        decision = gate.evaluate("file_list", {"path": "."}, tool_risk=RiskLevel.LOW)
        assert decision.risk == RiskLevel.LOW
        assert decision.allowed

    def test_file_write_is_medium_risk(self):
        gate = SafetyGate()
        decision = gate.evaluate("file_write", {"path": "/tmp/test", "content": "x"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.MEDIUM
        assert decision.allowed  # auto-approved

    def test_shell_is_medium_risk(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "echo hello"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.MEDIUM
        assert decision.allowed


class TestBlockedCommands:
    """Verify CRITICAL commands are always blocked."""

    def test_rm_rf_root_blocked(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "rm -rf /"}, tool_risk=RiskLevel.MEDIUM)
        assert not decision.allowed
        assert decision.risk == RiskLevel.CRITICAL

    def test_rm_rf_home_blocked(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "rm -rf ~"}, tool_risk=RiskLevel.MEDIUM)
        assert not decision.allowed
        assert decision.risk == RiskLevel.CRITICAL

    def test_mkfs_blocked(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "mkfs.ext4 /dev/sda1"}, tool_risk=RiskLevel.MEDIUM)
        assert not decision.allowed
        assert decision.risk == RiskLevel.CRITICAL

    def test_dd_blocked(self):
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "dd if=/dev/zero of=/dev/sda"}, tool_risk=RiskLevel.MEDIUM)
        assert not decision.allowed
        assert decision.risk == RiskLevel.CRITICAL

    def test_blocked_command_records_violation(self):
        gate = SafetyGate()
        gate.evaluate("shell", {"command": "rm -rf /"}, tool_risk=RiskLevel.MEDIUM)
        assert len(gate.violations) == 1
        assert gate.violations[0].level == RiskLevel.CRITICAL


class TestHighRiskApproval:
    """Verify HIGH risk commands require approval."""

    def test_rm_requires_approval_with_callback(self):
        """rm should require approval when a callback is present."""
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("shell", {"command": "rm /tmp/file"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.HIGH
        assert decision.requires_approval

    def test_rm_denied_without_callback(self):
        """rm should be denied when no callback is present."""
        gate = SafetyGate()
        decision = gate.evaluate("shell", {"command": "rm /tmp/file"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.HIGH
        assert not decision.allowed

    def test_git_reset_hard_requires_approval(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("shell", {"command": "git reset --hard HEAD~3"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.HIGH
        assert decision.requires_approval

    def test_git_push_force_requires_approval(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("shell", {"command": "git push --force origin main"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.HIGH
        assert decision.requires_approval

    def test_sudo_requires_approval(self):
        gate = SafetyGate(approval_callback=lambda r, t, a: True)
        decision = gate.evaluate("shell", {"command": "sudo apt update"}, tool_risk=RiskLevel.MEDIUM)
        assert decision.risk == RiskLevel.HIGH
        assert decision.requires_approval


class TestApprovalCallback:
    """Verify the approval callback is called and respected."""

    def test_approval_granted(self):
        """When callback returns True, operation proceeds."""
        called_with: list[tuple] = []
        def callback(risk, tool_name, args):
            called_with.append((risk, tool_name, args))
            return True

        gate = SafetyGate(approval_callback=callback)
        approved = gate.request_approval(RiskLevel.HIGH, "shell", {"command": "rm /tmp/x"})
        assert approved
        assert len(called_with) == 1
        assert called_with[0][0] == RiskLevel.HIGH

    def test_approval_denied(self):
        """When callback returns False, operation is blocked."""
        gate = SafetyGate(approval_callback=lambda r, t, a: False)
        approved = gate.request_approval(RiskLevel.HIGH, "shell", {"command": "rm /tmp/x"})
        assert not approved

    def test_approval_callback_exception_returns_false(self):
        """If callback throws, approval is denied (fail-safe)."""
        def bad_callback(risk, tool_name, args):
            raise RuntimeError("callback crashed")

        gate = SafetyGate(approval_callback=bad_callback)
        approved = gate.request_approval(RiskLevel.HIGH, "shell", {"command": "rm /tmp/x"})
        assert not approved


class TestSafetyInAgentLoop:
    """Verify safety gates work within the agent loop."""

    def test_blocked_command_in_loop(
        self,
        tool_executor,
        loop_config,
    ):
        """Model calls rm -rf / — should be blocked, model gets error."""
        model = ScriptedModel([
            ModelResponse(
                text="trying dangerous command",
                model="scripted",
                provider="test",
                tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "rm -rf /"})],
                finish_reason="tool_calls",
            ),
            make_text_response("Command was blocked. I cannot do that."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("delete everything")

        assert result.success  # loop completes (model gives answer after block)
        assert result.tool_calls == 1
        # The tool result should show failure
        failed_results = [
            s for s in result.trajectory.steps
            if s.step_type.value == "tool_result" and s.tool_ok is False
        ]
        assert len(failed_results) >= 1

    def test_auto_approved_medium_in_loop(
        self,
        tool_executor,
        loop_config,
        tmp_workspace,
    ):
        """Medium-risk file_write should be auto-approved in the loop."""
        model = ScriptedModel([
            make_tool_response("file_write", arguments={
                "path": str(tmp_workspace / "output.txt"),
                "content": "test content",
            }),
            make_text_response("File written successfully."),
        ])
        loop = AgentLoop(model, tool_executor, config=loop_config)
        result = loop.run("write a file")

        assert result.success
        assert result.tool_calls == 1
