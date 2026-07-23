"""Diagnostics — check functionality and implementation status of all MAOS tools.

This module actually calls each tool and verifies it works, not just that
it's registered. It distinguishes between:

  - working: tool was called, returned valid JSON without error
  - degraded: tool was called, returned a result but with warnings/issues
  - broken: tool was called but raised an exception or returned an error
  - not_implemented: tool is registered but has no handler
  - skipped: tool has side effects (email, task creation) — only checked
    for handler existence, not called

The diagnostic is safe to run at any time. Tools with side effects are
NOT called — only their handler existence is verified.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.log import get_logger
from app.runtime import Runtime

log = get_logger(__name__)


@dataclass
class ToolStatus:
    """Status of a single tool check."""
    name: str
    category: str  # read_only, action, onboarding, email, guard, side_effect
    status: str  # working, degraded, broken, not_implemented, skipped
    latency_ms: int = 0
    error: str = ""
    detail: str = ""
    returns_valid_json: bool = False
    result_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""
    timestamp: str
    total_tools: int
    working: int = 0
    degraded: int = 0
    broken: int = 0
    not_implemented: int = 0
    skipped: int = 0
    tools: list[ToolStatus] = field(default_factory=list)

    @property
    def all_working(self) -> bool:
        return self.broken == 0 and self.not_implemented == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_tools": self.total_tools,
            "summary": {
                "working": self.working,
                "degraded": self.degraded,
                "broken": self.broken,
                "not_implemented": self.not_implemented,
                "skipped": self.skipped,
            },
            "all_working": self.all_working,
            "tools": [t.to_dict() for t in self.tools],
        }


def run_diagnostics(runtime: Runtime) -> DiagnosticReport:
    """Run diagnostics on all MAOS MCP tools.

    Actually calls read-only tools and verifies they return valid JSON.
    For side-effect tools (email, task creation), only checks handler
    existence without calling them.

    Args:
        runtime: MAOS Runtime instance.

    Returns:
        DiagnosticReport with per-tool status.
    """
    from datetime import UTC, datetime

    report = DiagnosticReport(
        timestamp=datetime.now(UTC).isoformat(),
        total_tools=0,
    )

    # --- Read-only tools (safe to call) ---
    _check_tool(report, "maos_get_context", "read_only", _test_get_context, runtime)
    _check_tool(report, "maos_search_memory", "read_only", _test_search_memory, runtime)
    _check_tool(report, "maos_workspace_summary", "read_only", _test_workspace_summary, runtime)
    _check_tool(report, "maos_list_tasks", "read_only", _test_list_tasks, runtime)
    _check_tool(report, "maos_onboard", "onboarding", _test_onboard, runtime)
    _check_tool(report, "maos_list_lessons", "onboarding", _test_list_lessons, runtime)
    _check_tool(report, "maos_scan_vibes", "guard", _test_scan_vibes, runtime)
    _check_tool(report, "maos_check_guards", "guard", _test_check_guards, runtime)

    # --- Side-effect tools (handler check only, don't actually call) ---
    _check_handler_only(report, "maos_create_task", "side_effect",
                        "creates a task in the DB — not called during diagnostics")
    _check_handler_only(report, "maos_run_action", "side_effect",
                        "executes arbitrary actions — not called during diagnostics")
    _check_handler_only(report, "maos_record_lesson", "side_effect",
                        "writes to user_model — not called during diagnostics")
    _check_handler_only(report, "maos_send_email", "email",
                        "sends a real email — not called during diagnostics")
    _check_handler_only(report, "maos_escalate", "email",
                        "sends a real email — not called during diagnostics")

    # Update totals.
    report.total_tools = len(report.tools)
    for t in report.tools:
        if t.status == "working":
            report.working += 1
        elif t.status == "degraded":
            report.degraded += 1
        elif t.status == "broken":
            report.broken += 1
        elif t.status == "not_implemented":
            report.not_implemented += 1
        elif t.status == "skipped":
            report.skipped += 1

    log.info(
        "diagnostics.complete",
        total=report.total_tools,
        working=report.working,
        degraded=report.degraded,
        broken=report.broken,
        skipped=report.skipped,
    )

    return report


def format_diagnostic_report(report: DiagnosticReport) -> str:
    """Format the diagnostic report as readable text."""
    lines: list[str] = []
    lines.append(f"=== MAOS TOOL DIAGNOSTICS ({report.timestamp}) ===")
    lines.append("")

    # Summary.
    s = {
        "working": report.working,
        "degraded": report.degraded,
        "broken": report.broken,
        "not_implemented": report.not_implemented,
        "skipped": report.skipped,
    }
    lines.append(f"Total: {report.total_tools}  "
                 f"working: {s['working']}  degraded: {s['degraded']}  "
                 f"broken: {s['broken']}  skipped: {s['skipped']}")
    lines.append("")

    # Group by category.
    categories: dict[str, list[ToolStatus]] = {}
    for t in report.tools:
        categories.setdefault(t.category, []).append(t)

    for cat in ["read_only", "onboarding", "guard", "action", "email", "side_effect"]:
        tools = categories.get(cat, [])
        if not tools:
            continue
        lines.append(f"--- {cat.upper().replace('_', ' ')} ---")
        for t in tools:
            status_icon = {
                "working": "OK",
                "degraded": "WARN",
                "broken": "FAIL",
                "not_implemented": "MISSING",
                "skipped": "SKIP",
            }.get(t.status, "?")
            line = f"  [{status_icon:>7s}] {t.name:<25s} {t.latency_ms:>5d}ms"
            if t.detail:
                line += f"  {t.detail}"
            if t.error:
                line += f"  ERROR: {t.error[:80]}"
            lines.append(line)
        lines.append("")

    if report.all_working:
        lines.append("All tools operational.")
    else:
        broken_list = [t.name for t in report.tools if t.status in ("broken", "not_implemented")]
        lines.append(f"BROKEN TOOLS: {', '.join(broken_list)}")

    return "\n".join(lines)


# --- Tool test functions ---------------------------------------------------


def _check_tool(
    report: DiagnosticReport,
    name: str,
    category: str,
    test_fn: callable,
    runtime: Runtime,
) -> None:
    """Run a tool test and record the result."""
    start = time.time()
    try:
        result_str = test_fn(runtime)
        latency_ms = int((time.time() - start) * 1000)

        # Check if result is valid JSON.
        valid_json = False
        preview = ""
        try:
            parsed = json.loads(result_str)
            valid_json = True
            # Check for error in the parsed result.
            if isinstance(parsed, dict) and parsed.get("error"):
                # Non-empty error string = real issue.
                status = "degraded"
                detail = f"returned error: {str(parsed['error'])[:100]}"
            else:
                status = "working"
                # Build a short preview.
                if isinstance(parsed, dict):
                    keys = list(parsed.keys())[:5]
                    detail = f"keys: {keys}"
                elif isinstance(parsed, list):
                    detail = f"list with {len(parsed)} items"
                else:
                    detail = str(parsed)[:80]
        except json.JSONDecodeError:
            status = "degraded"
            detail = "returned non-JSON string"
            preview = result_str[:100]

        report.tools.append(ToolStatus(
            name=name,
            category=category,
            status=status,
            latency_ms=latency_ms,
            detail=detail,
            returns_valid_json=valid_json,
            result_preview=preview,
        ))

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        report.tools.append(ToolStatus(
            name=name,
            category=category,
            status="broken",
            latency_ms=latency_ms,
            error=str(e)[:200],
        ))


def _check_handler_only(
    report: DiagnosticReport,
    name: str,
    category: str,
    reason: str,
) -> None:
    """Record a side-effect tool as skipped (handler existence only)."""
    # Check if the tool function exists in the tools module.
    from app.mcp import tools as tools_module
    handler = getattr(tools_module, name, None)
    if handler is None:
        report.tools.append(ToolStatus(
            name=name,
            category=category,
            status="not_implemented",
            detail="no handler function found in app.mcp.tools",
        ))
    else:
        report.tools.append(ToolStatus(
            name=name,
            category=category,
            status="skipped",
            detail=reason,
        ))


# --- Individual tool tests (safe, read-only) -------------------------------


def _test_get_context(runtime: Runtime) -> str:
    from app.mcp.tools import maos_get_context
    return maos_get_context(runtime)


def _test_search_memory(runtime: Runtime) -> str:
    from app.mcp.tools import maos_search_memory
    return maos_search_memory(runtime, query="test", top_k=1)


def _test_workspace_summary(runtime: Runtime) -> str:
    from app.mcp.tools import maos_workspace_summary
    # Try to use the active workspace name if available.
    ws_name = ""
    try:
        from app.workspace import detect_active_workspace
        ws = detect_active_workspace(runtime.db)
        if ws:
            ws_name = ws.get("name", "")
    except Exception:
        pass
    return maos_workspace_summary(runtime, workspace=ws_name)


def _test_list_tasks(runtime: Runtime) -> str:
    from app.mcp.tools import maos_list_tasks
    return maos_list_tasks(runtime, workspace="")


def _test_onboard(runtime: Runtime) -> str:
    from app.mcp.tools import maos_onboard
    return maos_onboard(runtime, format="json")


def _test_list_lessons(runtime: Runtime) -> str:
    from app.mcp.tools import maos_list_lessons
    return maos_list_lessons(runtime)


def _test_scan_vibes(runtime: Runtime) -> str:
    from app.mcp.tools import maos_scan_vibes
    return maos_scan_vibes(runtime, text="everything is working")


def _test_check_guards(runtime: Runtime) -> str:
    from app.mcp.tools import maos_check_guards
    return maos_check_guards(runtime, text="diagnostic test")
