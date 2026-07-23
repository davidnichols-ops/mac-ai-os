"""Tests for Phase 9: adapter architecture evaluation.

Tests:
  - WorkflowAdapter interface (abstract methods)
  - FrameworkEvaluation scoring
  - LangGraphAdapter, PydanticAIAdapter, TemporalAdapter stubs
  - get_framework_evaluations returns all frameworks
  - get_recommended_framework returns the best
  - format_evaluation_report produces readable output
"""

import pytest

from app.workflows.adapters import (
    FrameworkEvaluation,
    LangGraphAdapter,
    PydanticAIAdapter,
    TemporalAdapter,
    WorkflowAdapter,
    format_evaluation_report,
    get_framework_evaluations,
    get_recommended_framework,
)

# --- WorkflowAdapter interface tests ---

class TestWorkflowAdapter:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            WorkflowAdapter()

    def test_langgraph_adapter(self):
        adapter = LangGraphAdapter()
        assert adapter.name() == "langgraph"

    def test_pydantic_ai_adapter(self):
        adapter = PydanticAIAdapter()
        assert adapter.name() == "pydantic-ai"

    def test_temporal_adapter(self):
        adapter = TemporalAdapter()
        assert adapter.name() == "temporal"


# --- Adapter stub tests ---

class TestAdapterStubs:
    def test_langgraph_can_run_graph(self):
        adapter = LangGraphAdapter()
        assert adapter.can_run({"type": "graph"}) is True
        assert adapter.can_run({"type": "sequential"}) is False

    def test_langgraph_run_not_installed(self):
        adapter = LangGraphAdapter()
        result = adapter.run({"type": "graph"})
        # Either "not installed" or "not implemented"
        assert result["ok"] is False

    def test_pydantic_ai_can_run_agent(self):
        adapter = PydanticAIAdapter()
        assert adapter.can_run({"type": "agent"}) is True
        assert adapter.can_run({"type": "graph"}) is False

    def test_temporal_can_run_anything(self):
        adapter = TemporalAdapter()
        assert adapter.can_run({"type": "sequential"}) is True
        assert adapter.can_run({}) is True

    def test_all_adapters_pause_returns_false(self):
        for adapter in [LangGraphAdapter(), PydanticAIAdapter(), TemporalAdapter()]:
            assert adapter.pause(1) is False

    def test_all_adapters_cancel_returns_false(self):
        for adapter in [LangGraphAdapter(), PydanticAIAdapter(), TemporalAdapter()]:
            assert adapter.cancel(1) is False


# --- FrameworkEvaluation tests ---

class TestFrameworkEvaluation:
    def test_default_score(self):
        e = FrameworkEvaluation(name="test")
        assert e.score == 0.0

    def test_compute_score_all_features(self):
        e = FrameworkEvaluation(
            name="full",
            durable_execution=True,
            state_machine=True,
            retry_support=True,
            timeout_support=True,
            compensation_support=True,
            permission_integration=True,
            mcp_exposure=True,
            python_native=True,
            production_ready=True,
            dependency_weight="light",
        )
        score = e.compute_score()
        assert score == 10.0  # max score

    def test_compute_score_no_features(self):
        e = FrameworkEvaluation(name="empty", dependency_weight="heavy")
        score = e.compute_score()
        assert score == 0.0

    def test_compute_score_partial(self):
        e = FrameworkEvaluation(
            name="partial",
            durable_execution=True,
            state_machine=True,
            retry_support=True,
        )
        score = e.compute_score()
        assert 3.0 <= score <= 5.0

    def test_to_dict_includes_score(self):
        e = FrameworkEvaluation(
            name="test", durable_execution=True, production_ready=True,
        )
        d = e.to_dict()
        assert "score" in d
        assert d["score"] > 0


# --- Evaluation results tests ---

class TestEvaluationResults:
    def test_get_framework_evaluations(self):
        evals = get_framework_evaluations()
        assert len(evals) >= 4
        names = [e.name for e in evals]
        assert "maos-custom" in names
        assert "langgraph" in names
        assert "pydantic-ai" in names
        assert "temporal" in names

    def test_all_evaluations_have_scores(self):
        evals = get_framework_evaluations()
        for e in evals:
            assert e.score > 0

    def test_maos_custom_has_highest_score(self):
        """MAOS custom engine should score highest due to full feature set."""
        evals = get_framework_evaluations()
        maos = next(e for e in evals if e.name == "maos-custom")
        for e in evals:
            if e.name != "maos-custom":
                assert maos.score >= e.score, f"maos-custom ({maos.score}) should beat {e.name} ({e.score})"

    def test_get_recommended_framework(self):
        recommended = get_recommended_framework()
        assert recommended == "maos-custom"

    def test_format_evaluation_report(self):
        report = format_evaluation_report()
        assert "Workflow Framework Evaluation" in report
        assert "maos-custom" in report
        assert "Recommended" in report
        assert "score" in report
