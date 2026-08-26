#!/usr/bin/env python3
"""EvalPlus HumanEval+ benchmark for adaptive-operator-v4.1.

Runs only our model. Comparison scores come from the public EvalPlus leaderboard:
  https://evalplus.github.io/leaderboard.html

This makes results immediately comparable without re-running other models.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path("/root/training/evalplus_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHART_PATH = RESULTS_DIR / "benchmark_comparison.png"
JSON_PATH = RESULTS_DIR / "benchmark_results.json"

# Our model
OUR_MODEL = "/dev/shm/merged_model"

# Published EvalPlus HumanEval+ pass@1 scores (greedy/temp=0)
# Source: https://evalplus.github.io/leaderboard.html (as of 2025)
PUBLISHED_SCORES = {
    "Qwen2.5-Coder-7B-Instruct": 68.9,
    "Qwen2.5-Coder-3B-Instruct": 62.2,
    "DeepSeek-Coder-6.7B-Instruct": 71.6,
    "Qwen2.5-7B-Instruct": 49.4,
    "Qwen3-8B": 65.2,
    "Llama-3.1-8B-Instruct": 47.6,
    "GPT-4o": 80.5,
    "Claude-3.5-Sonnet": 81.7,
}


def run_evalplus(model_path: str) -> dict:
    """Run EvalPlus HumanEval+ on our model."""
    cmd = [
        "python3", "-m", "evalplus.evaluate",
        "--model", model_path,
        "--dataset", "humaneval",
        "--backend", "vllm",
        "--greedy",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running EvalPlus HumanEval+ on: {model_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
        env={**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    elapsed = time.time() - t0
    
    # Parse pass@1 from output
    pass_at_1 = None
    # EvalPlus prints something like "humaneval plus  pass@1: 68.9"
    for line in result.stdout.split("\n"):
        if "pass@1" in line.lower():
            match = re.search(r"pass@1[:\s]+([\d.]+)", line, re.IGNORECASE)
            if match:
                pass_at_1 = float(match.group(1))
                break
    
    # Also check for "plus" and "base" separately
    plus_score = None
    base_score = None
    for line in result.stdout.split("\n"):
        if "plus" in line.lower() and "pass@1" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match:
                plus_score = float(match.group(1))
        if "base" in line.lower() and "pass@1" in line.lower():
            match = re.search(r"([\d.]+)", line.split("pass@1")[-1])
            if match:
                base_score = float(match.group(1))
    
    return {
        "model_path": model_path,
        "pass_at_1": pass_at_1,
        "plus_pass_at_1": plus_score,
        "base_pass_at_1": base_score,
        "elapsed_s": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr[-1000:] if result.stderr else "",
        "returncode": result.returncode,
    }


def generate_chart(our_score: float) -> None:
    """Generate comparison chart with published scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Combine our score with published scores
    all_models = {
        "Adaptive Operator v4.1 (ours)": our_score,
    }
    all_models.update(PUBLISHED_SCORES)
    
    # Sort by score descending
    sorted_models = sorted(all_models.items(), key=lambda x: x[1], reverse=True)
    names = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]
    
    # Colors — highlight our model
    colors = ["#e74c3c" if "ours" in n else "#3498db" for n in names]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("pass@1 (%)", fontsize=12)
    ax.set_title("EvalPlus HumanEval+ Benchmark\n(greedy decoding, pass@1)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Our model"),
        Patch(facecolor="#3498db", label="Published scores (EvalPlus leaderboard)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    
    # Subtitle
    fig.text(0.5, 0.01, "HumanEval+ (164 problems) | Greedy decoding | vLLM backend | L40S 48GB\n"
             "Published scores from evalplus.github.io/leaderboard.html",
             ha="center", fontsize=9, color="gray")
    
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {CHART_PATH}")


def main():
    if not Path(OUR_MODEL).exists():
        print(f"ERROR: Merged model not found at {OUR_MODEL}")
        sys.exit(1)
    
    print("Running EvalPlus HumanEval+ on our model only...")
    print("Comparison scores will come from the public EvalPlus leaderboard.\n")
    
    result = run_evalplus(OUR_MODEL)
    
    our_score = result.get("plus_pass_at_1") or result.get("pass_at_1") or 0.0
    
    # Save results
    output = {
        "our_model": {
            "path": OUR_MODEL,
            "pass_at_1": result.get("pass_at_1"),
            "plus_pass_at_1": result.get("plus_pass_at_1"),
            "base_pass_at_1": result.get("base_pass_at_1"),
            "elapsed_s": result["elapsed_s"],
            "returncode": result["returncode"],
        },
        "published_scores": PUBLISHED_SCORES,
        "stdout": result["stdout"][-5000:],
    }
    
    with open(JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULT: Our model HumanEval+ pass@1 = {our_score:.1f}%")
    print(f"Elapsed: {result['elapsed_s']:.0f}s ({result['elapsed_s']/60:.1f} min)")
    print(f"{'='*60}\n")
    
    # Print comparison table
    print(f"{'Model':<40} {'HumanEval+ pass@1':>20}")
    print("-" * 62)
    print(f"{'Adaptive Operator v4.1 (ours)':<40} {our_score:>19.1f}%")
    for name, score in sorted(PUBLISHED_SCORES.items(), key=lambda x: x[1], reverse=True):
        marker = " <" if score < our_score else (" >" if score > our_score else " =")
        print(f"{name:<40} {score:>19.1f}%{marker}")
    
    # Generate chart
    generate_chart(our_score)
    
    print(f"\nResults saved to {JSON_PATH}")
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
