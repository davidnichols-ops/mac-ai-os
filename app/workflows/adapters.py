"""Adapter architecture evaluation — LangGraph, PydanticAI, Temporal, etc.

Phase 9: This module evaluates whether MAOS should adopt an external workflow
framework or continue with its custom engine. It includes:

  1. WorkflowAdapter: An abstract interface that external frameworks
     can implement to plug into MAOS's workflow runtime.
  2. AdapterEvaluation: A structured comparison of candidate frameworks.
  3. LangGraphAdapter: A reference adapter showing how LangGraph could
     be integrated (if installed).
  4. PydanticAIAdapter: A reference adapter for PydanticAI.

The evaluation considers:
  - Durable execution (persistence, crash recovery)
  - State machine support
  - Retry/timeout/compensation
  - Permission/authorization integration
  - MCP/Context API exposure
  - Complexity and dependencies
  - Python ecosystem fit

Conclusion: MAOS's custom engine provides capabilities that none of the
evaluated frameworks offer out-of-the-box (particularly the Trustcard
integration point and MCP exposure). The adapter interface allows future
integration if needed, but the custom engine is the right choice for now.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.log import get_logger

log = get_logger(__name__)


# --- Adapter Interface ---

class WorkflowAdapter(ABC):
    """Abstract interface for external workflow framework adapters.

    An adapter wraps an external framework (LangGraph, PydanticAI, Temporal)
    and exposes it through MAOS's workflow runtime interface. This allows
    MAOS to use external frameworks while maintaining its trust boundary,
    persistence, and MCP exposure.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the adapter name (e.g., 'langgraph', 'pydantic-ai')."""

    @abstractmethod
    def can_run(self, workflow_def: dict[str, Any]) -> bool:
        """Check if this adapter can run the given workflow definition."""

    @abstractmethod
    def run(
        self, workflow_def: dict[str, Any], context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a workflow and return the result."""

    @abstractmethod
    def pause(self, run_id: int) -> bool:
        """Pause a running workflow."""

    @abstractmethod
    def resume(self, run_id: int) -> dict[str, Any]:
        """Resume a paused or crashed workflow."""

    @abstractmethod
    def cancel(self, run_id: int) -> bool:
        """Cancel a running workflow."""


# --- Evaluation Framework ---

@dataclass
class FrameworkEvaluation:
    """Structured evaluation of a workflow framework."""

    name: str
    description: str = ""
    durable_execution: bool = False
    state_machine: bool = False
    retry_support: bool = False
    timeout_support: bool = False
    compensation_support: bool = False
    permission_integration: bool = False
    mcp_exposure: bool = False
    python_native: bool = False
    dependency_weight: str = "light"  # light, medium, heavy
    learning_curve: str = "low"  # low, medium, high
    production_ready: bool = False
    notes: str = ""
    score: float = 0.0  # 0-10, computed from above

    def compute_score(self) -> float:
        """Compute a score from the evaluation criteria."""
        score = 0.0
        if self.durable_execution:
            score += 2.0
        if self.state_machine:
            score += 1.0
        if self.retry_support:
            score += 1.0
        if self.timeout_support:
            score += 0.5
        if self.compensation_support:
            score += 1.0
        if self.permission_integration:
            score += 1.5
        if self.mcp_exposure:
            score += 1.0
        if self.python_native:
            score += 0.5
        if self.production_ready:
            score += 1.5
        if self.dependency_weight == "light":
            score += 0.5
        self.score = min(score, 10.0)
        return self.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "durable_execution": self.durable_execution,
            "state_machine": self.state_machine,
            "retry_support": self.retry_support,
            "timeout_support": self.timeout_support,
            "compensation_support": self.compensation_support,
            "permission_integration": self.permission_integration,
            "mcp_exposure": self.mcp_exposure,
            "python_native": self.python_native,
            "dependency_weight": self.dependency_weight,
            "learning_curve": self.learning_curve,
            "production_ready": self.production_ready,
            "notes": self.notes,
            "score": self.compute_score(),
        }


# --- Reference Adapters (stubs) ---

class LangGraphAdapter(WorkflowAdapter):
    """Reference adapter for LangGraph.

    LangGraph is a graph-based workflow framework from LangChain. It supports
    cycles, state management, and human-in-the-loop interactions. However,
    it does not provide durable execution (persistence to disk) out of the
    box — you need to add a checkpointer.

    This adapter is a stub showing how LangGraph could be integrated.
    LangGraph is not a dependency of MAOS; this adapter would only be
    used if the user explicitly installs langgraph.
    """

    def name(self) -> str:
        return "langgraph"

    def can_run(self, workflow_def: dict[str, Any]) -> bool:
        # LangGraph can run graph-based workflows.
        return workflow_def.get("type") == "graph"

    def run(self, workflow_def: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            from langgraph.graph import StateGraph  # type: ignore # noqa: F401
        except ImportError:
            return {"ok": False, "error": "langgraph not installed"}
        # Implementation would convert MAOS workflow to LangGraph StateGraph.
        return {"ok": False, "error": "langgraph adapter is a reference stub; MAOS uses the custom engine"}

    def pause(self, run_id: int) -> bool:
        return False  # LangGraph doesn't have native pause

    def resume(self, run_id: int) -> dict[str, Any]:
        return {"ok": False, "error": "reference stub; MAOS uses the custom engine"}

    def cancel(self, run_id: int) -> bool:
        return False


class PydanticAIAdapter(WorkflowAdapter):
    """Reference adapter for PydanticAI.

    PydanticAI is a framework for building AI agent applications with
    type-safe Pydantic models. It focuses on agent-tool interactions
    rather than durable workflow execution.

    This adapter is a stub showing how PydanticAI could be integrated.
    """

    def name(self) -> str:
        return "pydantic-ai"

    def can_run(self, workflow_def: dict[str, Any]) -> bool:
        return workflow_def.get("type") == "agent"

    def run(self, workflow_def: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import pydantic_ai  # noqa: F401  # type: ignore
        except ImportError:
            return {"ok": False, "error": "pydantic-ai not installed"}
        return {"ok": False, "error": "reference stub; MAOS uses the custom engine"}

    def pause(self, run_id: int) -> bool:
        return False

    def resume(self, run_id: int) -> dict[str, Any]:
        return {"ok": False, "error": "reference stub; MAOS uses the custom engine"}

    def cancel(self, run_id: int) -> bool:
        return False


class TemporalAdapter(WorkflowAdapter):
    """Reference adapter for Temporal.

    Temporal is a durable execution platform that provides exactly the
    semantics MAOS needs (durable workflows, retry, compensation, etc.).
    However, it requires a separate Temporal server deployment, which
    adds significant infrastructure complexity.

    This adapter is a stub showing how Temporal could be integrated.
    """

    def name(self) -> str:
        return "temporal"

    def can_run(self, workflow_def: dict[str, Any]) -> bool:
        return True  # Temporal can run any workflow type

    def run(self, workflow_def: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import temporalio  # noqa: F401  # type: ignore
        except ImportError:
            return {"ok": False, "error": "temporalio not installed"}
        return {"ok": False, "error": "reference stub; MAOS uses the custom engine"}

    def pause(self, run_id: int) -> bool:
        return False  # Temporal uses signals for pause, not direct API

    def resume(self, run_id: int) -> dict[str, Any]:
        return {"ok": False, "error": "reference stub; MAOS uses the custom engine"}

    def cancel(self, run_id: int) -> bool:
        return False  # Temporal uses cancellation signals


# --- Evaluation Results ---

def get_framework_evaluations() -> list[FrameworkEvaluation]:
    """Return evaluations of all considered frameworks."""
    evaluations = [
        FrameworkEvaluation(
            name="maos-custom",
            description="MAOS's custom workflow engine (this codebase)",
            durable_execution=True,
            state_machine=True,
            retry_support=True,
            timeout_support=True,
            compensation_support=True,
            permission_integration=True,
            mcp_exposure=True,
            python_native=True,
            dependency_weight="light",
            learning_curve="low",
            production_ready=True,
            notes="Full control, no external dependencies, integrated with MAOS trust boundary.",
        ),
        FrameworkEvaluation(
            name="langgraph",
            description="Graph-based workflow framework from LangChain",
            durable_execution=False,  # requires checkpointer
            state_machine=True,
            retry_support=True,
            timeout_support=False,
            compensation_support=False,
            permission_integration=False,
            mcp_exposure=False,
            python_native=True,
            dependency_weight="heavy",  # pulls in langchain
            learning_curve="medium",
            production_ready=True,
            notes="Good for LLM-heavy workflows with cycles. Needs checkpointer for durability.",
        ),
        FrameworkEvaluation(
            name="pydantic-ai",
            description="Type-safe AI agent framework",
            durable_execution=False,
            state_machine=False,
            retry_support=False,
            timeout_support=False,
            compensation_support=False,
            permission_integration=False,
            mcp_exposure=False,
            python_native=True,
            dependency_weight="medium",
            learning_curve="low",
            production_ready=False,
            notes="Focused on agent-tool interactions, not durable workflows.",
        ),
        FrameworkEvaluation(
            name="temporal",
            description="Durable execution platform",
            durable_execution=True,
            state_machine=True,
            retry_support=True,
            timeout_support=True,
            compensation_support=True,  # via saga pattern
            permission_integration=False,
            mcp_exposure=False,
            python_native=True,  # has Python SDK
            dependency_weight="heavy",  # requires Temporal server
            learning_curve="high",
            production_ready=True,
            notes="Excellent durability, but requires separate server deployment. Overkill for local MAOS.",
        ),
        FrameworkEvaluation(
            name="prefect",
            description="Data pipeline orchestration framework",
            durable_execution=True,
            state_machine=False,
            retry_support=True,
            timeout_support=True,
            compensation_support=False,
            permission_integration=False,
            mcp_exposure=False,
            python_native=True,
            dependency_weight="medium",
            learning_curve="medium",
            production_ready=True,
            notes="Good for data pipelines, but not designed for LLM agent workflows.",
        ),
    ]

    for e in evaluations:
        e.compute_score()

    return evaluations


def get_recommended_framework() -> str:
    """Return the recommended framework based on the evaluation."""
    evaluations = get_framework_evaluations()
    best = max(evaluations, key=lambda e: e.score)
    return best.name


def format_evaluation_report() -> str:
    """Format the evaluation as a human-readable report."""
    evaluations = get_framework_evaluations()
    lines = ["Workflow Framework Evaluation", "=" * 40, ""]

    for e in sorted(evaluations, key=lambda x: x.score, reverse=True):
        lines.append(f"{e.name} (score: {e.score:.1f}/10)")
        lines.append(f"  {e.description}")
        lines.append(f"  Durable: {e.durable_execution}, StateMachine: {e.state_machine}")
        lines.append(f"  Retry: {e.retry_support}, Timeout: {e.timeout_support}")
        lines.append(f"  Compensation: {e.compensation_support}")
        lines.append(f"  Permissions: {e.permission_integration}, MCP: {e.mcp_exposure}")
        lines.append(f"  Deps: {e.dependency_weight}, Curve: {e.learning_curve}")
        lines.append(f"  Notes: {e.notes}")
        lines.append("")

    recommended = get_recommended_framework()
    lines.append(f"Recommended: {recommended}")
    return "\n".join(lines)
