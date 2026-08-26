"""Self-improving training loop orchestrator for the v4 post-training pipeline.

Runs an iterative loop that generates training data, trains a model, evaluates
it, identifies weak areas, and generates targeted prompts for the next
iteration — until an eval threshold is met or max iterations are reached.

Loop steps (per iteration):
  1. Generate diverse prompts (prompt_generator.py)
  2. Generate teacher responses (qwen_inference.py — Qwen v3.1 on Together AI)
  3. Run 5-reviewer improvement (review_orchestrator.py)
  4. Export SFT + DPO datasets (sft_export.py, dpo_generator.py, dpo_export.py)
  5. Trigger training (Colab T4 notebook or Together AI)
  6. Run evaluation (eval/eval_suite.py)
  7. Identify weak categories from eval results
  8. Generate targeted prompts for weak categories
  9. Repeat until eval threshold met or max iterations reached

The orchestrator runs locally on Mac. It calls the prompt generator, Qwen
inference (via Together AI API), review orchestrator, and eval suite locally.
Training happens on Colab T4 (triggered by copying data to Drive and starting
the notebook) or on Together AI.

Usage::

    # Run the full loop (defaults)
    uv run python -m training.v4_pipeline.training_loop

    # Custom config
    uv run python -m training.v4_pipeline.training_loop \\
        --max-iterations 5 --prompts-per-iteration 1000

    # Resume from the last completed iteration
    uv run python -m training.v4_pipeline.training_loop --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from training.v4_pipeline import prompt_generator
from training.v4_pipeline.dpo_export import pair_to_together_format
from training.v4_pipeline.dpo_generator import generate_cross_model_pairs
from training.v4_pipeline.dpo_generator import load_jsonl as load_dpo_jsonl
from training.v4_pipeline.qwen_inference import batch_generate
from training.v4_pipeline.review_orchestrator import run as run_review
from training.v4_pipeline.sft_export import export_sft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
LOGS_DIR = PIPELINE_DIR / "logs"
PROMPTS_DIR = PIPELINE_DIR / "prompts"
SFT_DIR = PIPELINE_DIR / "sft"
DPO_DIR = PIPELINE_DIR / "dpo"

STATE_PATH = LOGS_DIR / "training_loop_state.json"
HISTORY_PATH = LOGS_DIR / "training_loop_history.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LoopConfig:
    """Configuration for the self-improving training loop."""

    max_iterations: int = 5
    prompts_per_iteration: int = 1000
    eval_threshold: float = 0.85  # Stop when overall pass rate > 85%
    min_improvement: float = 0.02  # Stop if improvement < 2% between iterations
    weak_category_threshold: float = 0.7  # Categories below 70% get targeted prompts
    targeted_prompt_boost: int = 500  # Extra prompts for weak categories

    # Paths
    pipeline_dir: Path = PIPELINE_DIR

    # Teacher model
    teacher_model: str = "davidnicholsop-199a/adaptive-operator-v3-1-eval"
    teacher_workers: int = 32

    # Training
    training_backend: str = "colab"  # "colab" or "together"
    colab_notebook_path: str = "training/v4_pipeline/colab_qlora_training.ipynb"

    # Eval
    eval_model: str = ""  # Path to trained model (filled per iteration)

    def __post_init__(self) -> None:
        """Normalize path types after dataclass init."""
        self.pipeline_dir = Path(self.pipeline_dir)


@dataclass
class IterationResult:
    """Result of a single loop iteration."""

    iteration: int
    prompts_generated: int = 0
    responses_generated: int = 0
    sft_examples: int = 0
    dpo_pairs: int = 0
    eval_pass_rate: float = 0.0
    weak_categories: list[str] = field(default_factory=list)
    model_path: str = ""
    duration_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSummary:
    """Parsed summary of an eval run."""

    overall_pass_rate: float
    weak_categories: list[str]
    by_category: dict[str, dict[str, Any]]
    raw_report: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file (0 if file does not exist)."""
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Write records to a JSONL file (overwriting). Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def _save_state(state: dict[str, Any]) -> None:
    """Persist loop state for resume support."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _load_state() -> dict[str, Any]:
    """Load persisted loop state (or empty dict if none)."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load loop state: %s", exc)
        return {}


def _save_history(history: list[IterationResult]) -> None:
    """Persist full iteration history."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    data = [h.to_dict() for h in history]
    HISTORY_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _load_history() -> list[IterationResult]:
    """Load persisted history (or empty list if none)."""
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [IterationResult(**d) for d in data]


# ---------------------------------------------------------------------------
# Step 1: Prompt generation
# ---------------------------------------------------------------------------


def generate_prompts(config: LoopConfig, weak_categories: list[str]) -> list[dict[str, Any]]:
    """Generate diverse prompts, with a targeted boost for weak categories.

    The base ``prompts_per_iteration`` prompts are generated with the default
    category weighting. For each weak category, an additional
    ``targeted_prompt_boost`` prompts are generated and filtered to that
    category only.
    """
    logger.info("Generating %d base prompts...", config.prompts_per_iteration)
    prompts = prompt_generator.generate_prompts(config.prompts_per_iteration)

    if weak_categories:
        boost = config.targeted_prompt_boost
        logger.info(
            "Generating %d targeted prompts for weak categories: %s", boost, weak_categories
        )
        # Generate extra prompts and keep only those matching weak categories.
        extra = prompt_generator.generate_prompts(boost * len(weak_categories))
        weak_set = set(weak_categories)
        targeted = [p for p in extra if p.get("category") in weak_set]
        # Cap at the configured boost per category.
        per_cat: dict[str, list[dict[str, Any]]] = {}
        for p in targeted:
            per_cat.setdefault(p["category"], []).append(p)
        for cat in weak_categories:
            prompts.extend(per_cat.get(cat, [])[:boost])
        logger.info("Added %d targeted prompts", len(prompts) - config.prompts_per_iteration)

    return prompts


# ---------------------------------------------------------------------------
# Step 2: Teacher response generation
# ---------------------------------------------------------------------------


def generate_responses(config: LoopConfig, prompts: list[dict[str, Any]], iteration: int) -> Path:
    """Generate teacher responses via Qwen v3.1 on Together AI.

    Writes prompts to a JSONL file, then calls ``batch_generate`` which streams
    the file through the inference client with resume support. If the teacher
    endpoint is down, the function logs a warning and returns the path to any
    existing responses so the loop can continue with cached data.
    """
    prompts_path = PROMPTS_DIR / f"prompts_iter{iteration}.jsonl"
    responses_path = SFT_DIR / f"qwen_responses_iter{iteration}.jsonl"

    _write_jsonl(prompts_path, prompts)
    logger.info("Wrote %d prompts to %s", len(prompts), prompts_path)

    try:
        summary = batch_generate(
            input_path=str(prompts_path),
            output_path=str(responses_path),
            model=config.teacher_model,
            resume=True,
        )
        logger.info("Teacher generation complete: %s", summary)
    except Exception as exc:
        logger.warning("Teacher endpoint unavailable (%s). Using existing data if present.", exc)
        if not responses_path.exists():
            logger.error(
                "No existing responses at %s and teacher is down — cannot continue.", responses_path
            )
            raise

    return responses_path


# ---------------------------------------------------------------------------
# Step 3: Review orchestrator
# ---------------------------------------------------------------------------


def run_review_step(responses_path: Path, iteration: int) -> Path:
    """Run the 5-reviewer orchestrator over teacher responses.

    Returns the path to the improved-responses JSONL file.
    """
    improved_path = SFT_DIR / f"improved_responses_iter{iteration}.jsonl"
    logger.info("Running 5-reviewer orchestrator: %s -> %s", responses_path, improved_path)
    run_review(responses_path, improved_path, resume=True)
    return improved_path


# ---------------------------------------------------------------------------
# Step 4: Dataset export
# ---------------------------------------------------------------------------


def export_datasets(improved_path: Path, responses_path: Path, iteration: int) -> tuple[Path, Path]:
    """Export SFT and DPO training datasets.

    SFT: converts improved responses to Together AI chat format.
    DPO: builds cross-model preference pairs (improved=chosen, original=rejected)
         then exports to Together AI DPO format.
    """
    sft_path = SFT_DIR / f"sft_train_iter{iteration}.jsonl"
    dpo_pairs_path = DPO_DIR / f"preference_pairs_iter{iteration}.jsonl"
    dpo_path = DPO_DIR / f"dpo_train_iter{iteration}.jsonl"

    # --- SFT export ---
    logger.info("Exporting SFT dataset -> %s", sft_path)
    sft_count = export_sft(input_path=improved_path, output_path=sft_path)
    logger.info("Exported %d SFT examples", sft_count)

    # --- DPO generation ---
    logger.info("Generating DPO preference pairs -> %s", dpo_pairs_path)
    qwen_records = load_dpo_jsonl(responses_path)
    improved_records = load_dpo_jsonl(improved_path)
    pairs = generate_cross_model_pairs(qwen_records, improved_records, seen=set())
    _write_jsonl(dpo_pairs_path, pairs)
    logger.info("Generated %d DPO preference pairs", len(pairs))

    # --- DPO export (Together AI format) ---
    logger.info("Exporting DPO dataset -> %s", dpo_path)
    dpo_records: list[dict[str, Any]] = []
    for pair in pairs:
        rec = pair_to_together_format(pair)
        if rec is not None:
            dpo_records.append(rec)
    _write_jsonl(dpo_path, dpo_records)
    logger.info("Exported %d DPO training examples", len(dpo_records))

    return sft_path, dpo_path


# ---------------------------------------------------------------------------
# Step 5: Training
# ---------------------------------------------------------------------------


def train(config: LoopConfig, sft_path: Path, dpo_path: Path, iteration: int) -> str:
    """Trigger training on the configured backend.

    Returns the model path / identifier of the trained model.

    - ``colab``: copies datasets to a Google Drive folder and triggers the
      Colab notebook via the Colab API or a shell command. The model is
      uploaded to Together AI or HuggingFace after training.
    - ``together``: submits SFT then DPO fine-tuning jobs to Together AI.
    """
    if config.training_backend == "colab":
        return _train_colab(config, sft_path, dpo_path, iteration)
    elif config.training_backend == "together":
        return _train_together(config, sft_path, dpo_path, iteration)
    else:
        raise ValueError(f"Unknown training_backend: {config.training_backend!r}")


def _train_colab(config: LoopConfig, sft_path: Path, dpo_path: Path, iteration: int) -> str:
    """Copy datasets to Google Drive and trigger the Colab notebook.

    This assumes a Google Drive mount at ``~/GoogleDrive`` (or configurable via
    environment). The datasets are copied to a pipeline-specific folder, and
    the Colab notebook is triggered via the Colab runtime API or a CLI tool.

    The model path returned is a placeholder that the Colab notebook updates in
    a metadata file after training completes.
    """
    drive_root = Path.home() / "GoogleDrive"
    if not drive_root.exists():
        logger.warning("Google Drive not mounted at %s — skipping Colab training.", drive_root)
        return f"colab://iter{iteration}/pending"

    dest_dir = drive_root / "maos_v4_pipeline" / f"iter{iteration}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in (sft_path, dpo_path):
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        logger.info("Copied %s -> %s", src, dst)

    # Trigger the notebook. In production this would use the Colab API or
    # gcloud to start a managed notebook execution. Here we log the action
    # and write a trigger file the notebook watches for.
    trigger = dest_dir / "TRAIN_TRIGGER.json"
    trigger.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "sft_file": sft_path.name,
                "dpo_file": dpo_path.name,
                "notebook": config.colab_notebook_path,
                "timestamp": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote Colab training trigger: %s", trigger)

    # Check for a completion metadata file (written by the notebook).
    metadata = dest_dir / "model_metadata.json"
    if metadata.exists():
        try:
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            model_path = meta.get("model_path", "")
            if model_path:
                logger.info("Colab training complete: model=%s", model_path)
                return model_path
        except (json.JSONDecodeError, OSError):
            pass

    logger.info("Colab training triggered but not yet complete. Using pending model path.")
    return f"colab://iter{iteration}/pending"


def _train_together(config: LoopConfig, sft_path: Path, dpo_path: Path, iteration: int) -> str:
    """Submit SFT then DPO fine-tuning jobs to Together AI.

    Uses the Together AI fine-tuning API via the ``train_sft`` and ``train_dpo``
    modules. Returns the final model name.
    """
    from training.v4_pipeline.train_dpo import main as train_dpo_main
    from training.v4_pipeline.train_sft import main as train_sft_main

    logger.info("Submitting SFT job to Together AI (iteration %d)...", iteration)
    sft_argv = [
        "--training-file",
        str(sft_path),
        "--model",
        config.teacher_model,
        "--name",
        f"adaptive-operator-v4-iter{iteration}-sft",
    ]
    sft_exit = train_sft_main(sft_argv)
    if sft_exit != 0:
        logger.error("SFT training failed (exit %d)", sft_exit)
        return f"together://iter{iteration}/sft-failed"

    # Read the SFT model name from the job metadata.
    sft_meta_path = LOGS_DIR / "sft_job_metadata.json"
    sft_model = ""
    if sft_meta_path.exists():
        try:
            meta = json.loads(sft_meta_path.read_text(encoding="utf-8"))
            sft_model = meta.get("model_name", meta.get("model_output_name", ""))
        except (json.JSONDecodeError, OSError):
            pass

    if not sft_model:
        logger.warning("Could not determine SFT model name; skipping DPO stage.")
        return f"together://iter{iteration}/sft-no-model"

    logger.info("Submitting DPO job to Together AI with base model %s...", sft_model)
    dpo_argv = [
        "--training-file",
        str(dpo_path),
        "--model",
        sft_model,
        "--name",
        f"adaptive-operator-v4-iter{iteration}-dpo",
    ]
    dpo_exit = train_dpo_main(dpo_argv)
    if dpo_exit != 0:
        logger.error("DPO training failed (exit %d)", dpo_exit)
        return sft_model

    # Read the DPO model name.
    dpo_meta_path = LOGS_DIR / "dpo_job_metadata.json"
    if dpo_meta_path.exists():
        try:
            meta = json.loads(dpo_meta_path.read_text(encoding="utf-8"))
            dpo_model = meta.get("model_name", meta.get("model_output_name", ""))
            if dpo_model:
                return dpo_model
        except (json.JSONDecodeError, OSError):
            pass

    return sft_model


# ---------------------------------------------------------------------------
# Step 6: Evaluation
# ---------------------------------------------------------------------------


def evaluate(config: LoopConfig, model_path: str, iteration: int) -> EvalSummary:
    """Run the eval suite against the trained model and parse results.

    Returns an :class:`EvalSummary` with the overall pass rate, weak
    categories (below ``weak_category_threshold``), and per-category breakdown.
    """
    from training.v4_pipeline.eval.eval_suite import EvalSuite

    eval_output = LOGS_DIR / f"eval_results_iter{iteration}.json"
    logger.info("Running eval suite (model=%s) -> %s", model_path, eval_output)

    suite = EvalSuite(
        model=model_path,
        output=eval_output,
    )
    report = suite.run()

    # Parse weak categories.
    weak_cats: list[str] = []
    for cat, stats in report.by_category.items():
        rate = stats.get("pass_rate", 0.0)
        if rate < config.weak_category_threshold:
            weak_cats.append(cat)

    summary = EvalSummary(
        overall_pass_rate=report.pass_rate,
        weak_categories=weak_cats,
        by_category=report.by_category,
        raw_report=report.to_dict(),
    )
    logger.info(
        "Eval complete: pass_rate=%.1%%, weak_categories=%s",
        report.pass_rate * 100,
        weak_cats,
    )
    return summary


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_loop(config: LoopConfig, resume: bool = False) -> dict[str, Any]:
    """Run the self-improving training loop.

    Args:
        config: Loop configuration.
        resume: If True, resume from the last completed iteration using
            persisted state in ``logs/training_loop_state.json``.

    Returns:
        A dict with ``iterations`` (list of IterationResult dicts) and
        ``best_model`` (the model path with the highest eval pass rate).
    """
    # --- Resume support ---
    start_iteration = 0
    history: list[IterationResult] = []
    if resume:
        state = _load_state()
        history = _load_history()
        start_iteration = state.get("last_completed_iteration", 0)
        if start_iteration > 0:
            logger.info(
                "Resuming from iteration %d (completed=%d)", start_iteration + 1, start_iteration
            )
        # Use the previous iteration's model as the eval baseline if available.
        if history and not config.eval_model:
            config.eval_model = history[-1].model_path

    for iteration in range(start_iteration, config.max_iterations):
        iter_num = iteration + 1
        logger.info("=== Iteration %d/%d ===", iter_num, config.max_iterations)
        iter_start = time.monotonic()

        result = IterationResult(iteration=iter_num)
        try:
            # Step 1: Generate prompts (with targeted boost for weak categories).
            weak_cats = history[-1].weak_categories if history else []
            prompts = generate_prompts(config, weak_cats)
            result.prompts_generated = len(prompts)

            # Step 2: Generate teacher responses.
            responses_path = generate_responses(config, prompts, iter_num)
            result.responses_generated = _count_jsonl_lines(responses_path)

            # Step 3: Run review orchestrator.
            improved_path = run_review_step(responses_path, iter_num)

            # Step 4: Export datasets.
            sft_path, dpo_path = export_datasets(improved_path, responses_path, iter_num)
            result.sft_examples = _count_jsonl_lines(sft_path)
            result.dpo_pairs = _count_jsonl_lines(dpo_path)

            # Step 5: Train.
            model_path = train(config, sft_path, dpo_path, iter_num)
            result.model_path = model_path

            # Step 6: Evaluate.
            eval_model = (
                model_path
                if model_path and "pending" not in model_path
                else (config.eval_model or config.teacher_model)
            )
            eval_results = evaluate(config, eval_model, iter_num)
            result.eval_pass_rate = eval_results.overall_pass_rate
            result.weak_categories = eval_results.weak_categories

        except Exception as exc:
            logger.exception("Iteration %d failed: %s", iter_num, exc)
            result.error = str(exc)

        result.duration_s = round(time.monotonic() - iter_start, 2)
        history.append(result)
        _save_history(history)

        # Persist state for resume.
        _save_state(
            {
                "last_completed_iteration": iter_num,
                "last_eval_pass_rate": result.eval_pass_rate,
                "last_model_path": result.model_path,
                "config": {
                    "max_iterations": config.max_iterations,
                    "prompts_per_iteration": config.prompts_per_iteration,
                    "eval_threshold": config.eval_threshold,
                    "training_backend": config.training_backend,
                },
            }
        )

        logger.info(
            "Iteration %d complete: pass_rate=%.1%%, duration=%.1fs",
            iter_num,
            result.eval_pass_rate * 100,
            result.duration_s,
        )

        # Step 8: Check stopping criteria.
        if result.error:
            logger.warning("Iteration had errors; continuing to next iteration.")

        if result.eval_pass_rate >= config.eval_threshold:
            logger.info("Eval threshold met: %.1%%. Stopping.", result.eval_pass_rate * 100)
            break

        if iteration > start_iteration and len(history) >= 2:
            improvement = history[-1].eval_pass_rate - history[-2].eval_pass_rate
            if improvement < config.min_improvement:
                logger.info(
                    "Improvement below threshold (%.1%% < %.1%%). Stopping.",
                    improvement * 100,
                    config.min_improvement * 100,
                )
                break

    # --- Final summary ---
    if not history:
        logger.warning("No iterations completed.")
        return {"iterations": [], "best_model": ""}

    best = max(history, key=lambda h: h.eval_pass_rate)
    logger.info(
        "Loop complete: %d iterations, best pass_rate=%.1%%, best_model=%s",
        len(history),
        best.eval_pass_rate * 100,
        best.model_path,
    )

    return {
        "iterations": [h.to_dict() for h in history],
        "best_model": best.model_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> LoopConfig:
    """Build a LoopConfig from CLI arguments."""
    config = LoopConfig(
        max_iterations=args.max_iterations,
        prompts_per_iteration=args.prompts_per_iteration,
        eval_threshold=args.eval_threshold,
        min_improvement=args.min_improvement,
        weak_category_threshold=args.weak_category_threshold,
        targeted_prompt_boost=args.targeted_prompt_boost,
        teacher_model=args.teacher_model,
        training_backend=args.training_backend,
        colab_notebook_path=args.colab_notebook,
        eval_model=args.eval_model,
    )
    if args.pipeline_dir:
        config.pipeline_dir = Path(args.pipeline_dir)
    return config


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the training loop orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run the self-improving v4 post-training loop.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of loop iterations (default: 5).",
    )
    parser.add_argument(
        "--prompts-per-iteration",
        type=int,
        default=1000,
        help="Number of prompts to generate per iteration (default: 1000).",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.85,
        help="Stop when overall eval pass rate exceeds this (default: 0.85).",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="Stop if improvement between iterations is below this (default: 0.02).",
    )
    parser.add_argument(
        "--weak-category-threshold",
        type=float,
        default=0.7,
        help="Categories below this pass rate get targeted prompts (default: 0.7).",
    )
    parser.add_argument(
        "--targeted-prompt-boost",
        type=int,
        default=500,
        help="Extra prompts per weak category (default: 500).",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=LoopConfig.teacher_model,
        help="Teacher model name on Together AI.",
    )
    parser.add_argument(
        "--training-backend",
        choices=["colab", "together"],
        default="colab",
        help="Training backend to use (default: colab).",
    )
    parser.add_argument(
        "--colab-notebook",
        type=str,
        default=LoopConfig.colab_notebook_path,
        help="Path to the Colab training notebook.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="",
        help="Model to evaluate (overrides trained model path).",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default="",
        help="Override the pipeline directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed iteration.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = _build_config(args)
    result = run_loop(config, resume=args.resume)

    # Print final summary.
    print("\n" + "=" * 70)
    print("  TRAINING LOOP SUMMARY")
    print("=" * 70)
    iterations = result.get("iterations", [])
    for it in iterations:
        status = "OK" if not it.get("error") else "ERROR"
        print(
            f"  Iter {it['iteration']}: pass_rate={it['eval_pass_rate']:.1%} "
            f"prompts={it['prompts_generated']} sft={it['sft_examples']} "
            f"dpo={it['dpo_pairs']} [{status}]"
        )
    print(f"\n  Best model: {result.get('best_model', 'N/A')}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
