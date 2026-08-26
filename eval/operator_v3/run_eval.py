"""CLI runner for the Adaptive Operator v3 evaluation harness.

Usage::

    # Run all tasks with the default (config-detected) model
    uv run python -m eval.operator_v3.run_eval

    # Specify a model (provider://model-name)
    uv run python -m eval.operator_v3.run_eval --model mlx://qwen3-14b

    # Run a single category
    uv run python -m eval.operator_v3.run_eval --category tool_competence

    # Custom tasks dir and output path
    uv run python -m eval.operator_v3.run_eval \
        --tasks-dir eval/operator_v3/tasks \
        --output eval/operator_v3/results/report.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config.settings import load_settings
from app.log import configure_logging, get_logger
from app.models.base import Model
from app.models.registry import build_model

# Make the eval package importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.operator_v3.eval_harness import (  # noqa: E402
    EvalHarness,
    HarnessConfig,
    save_report,
)

log = get_logger(__name__)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a ``provider://model-name`` string.

    Examples::
        mlx://qwen3-14b       -> ("mlx", "qwen3-14b")
        ollama://llama3.2     -> ("ollama", "llama3.2")
        qwen3-14b             -> ("mlx", "qwen3-14b")  # default provider
    """
    if "://" in spec:
        provider, model = spec.split("://", 1)
        return provider.strip(), model.strip()
    return "mlx", spec.strip()


def build_eval_model(model_spec: str | None) -> Model:
    """Build a Model instance for the eval, optionally overriding config.

    If ``model_spec`` is None, the planner model from loaded settings is used.
    Otherwise the spec (``provider://model``) overrides the planner config.
    """
    settings = load_settings()

    if model_spec:
        provider, model_name = parse_model_spec(model_spec)
        settings = settings.model_copy()
        settings.planner.provider = provider
        settings.planner.model = model_name
        log.info("eval.model_override", provider=provider, model=model_name)

    return build_model("planner", settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Run the Adaptive Operator v3 evaluation harness.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model spec as provider://model-name (e.g. mlx://qwen3-14b). "
        "Defaults to the planner model from config.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("eval/operator_v3/tasks"),
        help="Directory containing task definition files (YAML/JSON).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to eval/operator_v3/results/report_<timestamp>.json.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["tool_competence", "recovery", "planning", "compute_allocation"],
        help="Run only tasks in this category.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum agent loop steps per task.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-task timeout in seconds.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=".",
        help="Workspace path for tool execution.",
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Disable persisting trajectories to disk.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable summary to stdout.",
    )
    args = parser.parse_args(argv)

    # Configure logging from settings.
    settings = load_settings()
    configure_logging(settings)

    # Build the model.
    model = build_eval_model(args.model)

    # Default output path with timestamp.
    output_path = args.output
    if output_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path(f"eval/operator_v3/results/report_{ts}.json")

    # Trajectory persistence dir.
    trajectory_dir: Path | None = None
    if not args.no_trajectories:
        trajectory_dir = Path("eval/operator_v3/results/trajectories")

    # Assemble harness config.
    config = HarnessConfig(
        tasks_dir=args.tasks_dir,
        results_dir=output_path.parent,
        category=args.category,
        max_steps=args.max_steps,
        timeout_s=args.timeout,
        workspace=args.workspace,
        persist_trajectories=not args.no_trajectories,
        trajectory_dir=trajectory_dir,
    )

    # Run the harness.
    harness = EvalHarness(model=model, config=config)
    log.info(
        "eval.run.start",
        model=model.name,
        provider=model.provider,
        tasks_dir=str(args.tasks_dir),
        category=args.category,
    )

    report = harness.run()

    # Save the report.
    save_report(report, output_path)

    # Output.
    if args.summary:
        print(report.summary())
        print(f"\nReport saved to: {output_path}")
    else:
        print(report.to_json())

    # Exit code: 0 if anything passed, 1 if everything failed.
    return 0 if report.passed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
