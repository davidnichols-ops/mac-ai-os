"""MCP tools — thin adapters over ActionExecutor and the Context API.

Every tool that performs an action goes through ActionExecutor, which
preserves:
  - permission checks
  - reversibility checks
  - action logging (action_log table)
  - event publishing (EventBus)

No tool may bypass ActionExecutor. Read-only tools (maos_get_context,
maos_search_memory) call the Context API directly since they have no
side effects.

All tool calls are instrumented with MCP telemetry: each call is logged
to the mcp_requests table and an MCP_TOOL_CALLED event is published.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.actions.models import ActionRequest
from app.events.models import MCP_TOOL_CALLED, Event
from app.log import get_logger
from app.runtime import Runtime
from app.server.context import build_context_packet

log = get_logger(__name__)


def _instrument(
    runtime: Runtime, name: str, fn: callable, *args: Any, **kwargs: Any
) -> str:
    """Wrap a tool call with telemetry logging and event publishing.

    Records to mcp_requests table and publishes MCP_TOOL_CALLED event.
    """
    start = time.time()
    success = True
    error = ""
    try:
        result = fn(runtime, *args, **kwargs)
        return result
    except Exception as e:
        success = False
        error = str(e)
        return json.dumps({"error": error})
    finally:
        latency_ms = int((time.time() - start) * 1000)
        try:
            runtime.db.log_mcp_request(
                client_name="mcp",
                kind="tool",
                name=name,
                success=success,
                latency_ms=latency_ms,
                error=error,
            )
            if runtime.bus:
                runtime.bus.publish(Event(MCP_TOOL_CALLED, {
                    "name": name, "success": success,
                    "latency_ms": latency_ms,
                }))
        except Exception:
            pass  # telemetry must never break the tool call


def maos_get_context(runtime: Runtime) -> str:
    """Get the full MAOS context packet.

    Read-only — calls the Context API directly (no side effects).
    """
    packet = build_context_packet(
        runtime.db, runtime.blackboard, retriever=runtime.retriever
    )
    return json.dumps(packet, default=str, indent=2)


def maos_search_memory(runtime: Runtime, query: str, top_k: int = 5) -> str:
    """Search MAOS memory (vector store) for content matching the query.

    Read-only — calls the retriever directly (no side effects).
    """
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        results = runtime.retriever.search(query, top_k=top_k)
        return json.dumps(
            {
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "text": r.text[:1000],
                        "source": r.source,
                        "score": round(r.score, 3),
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
            },
            default=str,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


def maos_workspace_summary(runtime: Runtime, workspace: str = "") -> str:
    """Generate a workspace summary.

    Goes through ActionExecutor → workspace_summary action.
    Preserves permissions, logging, and event publishing.
    """
    return _run_action(
        runtime,
        action="workspace_summary",
        target=workspace,
        parameters={"workspace": workspace},
    )


def maos_create_task(
    runtime: Runtime,
    title: str,
    workspace: str = "",
    priority: int = 0,
) -> str:
    """Create a new TODO task.

    Goes through ActionExecutor → create_todo action.
    Preserves permissions, logging, and event publishing.
    """
    return _run_action(
        runtime,
        action="create_todo",
        target=title,
        parameters={
            "title": title,
            "workspace": workspace,
            "priority": priority,
        },
    )


def maos_list_tasks(runtime: Runtime, workspace: str = "") -> str:
    """List pending tasks.

    Goes through ActionExecutor → list_tasks action.
    Preserves permissions, logging, and event publishing.
    """
    return _run_action(
        runtime,
        action="list_tasks",
        target=workspace,
        parameters={"workspace": workspace},
    )


def maos_run_action(
    runtime: Runtime,
    action: str,
    target: str = "",
    parameters: dict[str, Any] | None = None,
) -> str:
    """Execute any registered MAOS action by name.

    This is the generic gateway to ActionExecutor. All checks, logging,
    and event publishing are preserved.
    """
    return _run_action(
        runtime,
        action=action,
        target=target,
        parameters=parameters or {},
    )


# --- internal helper -------------------------------------------------------


def _run_action(
    runtime: Runtime,
    *,
    action: str,
    target: str = "",
    parameters: dict[str, Any] | None = None,
) -> str:
    """Execute an action through ActionExecutor and return JSON result.

    This is the ONLY path to action execution. It always:
    1. Creates an ActionRequest
    2. Calls runtime.action_executor.execute() with db in context
    3. Returns the ActionResult as JSON

    The ActionExecutor handles:
    - permission validation
    - reversibility checks
    - handler execution
    - action_log table insertion
    - EventBus publish (ACTION_EXECUTED / ACTION_DENIED)
    """
    req = ActionRequest(
        action=action,
        target=target,
        parameters=parameters or {},
    )
    result = runtime.action_executor.execute(req, context={"db": runtime.db})
    log.info(
        "mcp.tool.action",
        action=action,
        ok=result.ok,
        duration_ms=result.duration_ms,
    )
    return json.dumps(result.to_dict(), default=str, indent=2)
