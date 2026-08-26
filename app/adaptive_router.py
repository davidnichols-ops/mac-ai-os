"""Adaptive router — FAST vs THINK mode selection for every user prompt.

Wraps the local-operator's LearnedRouter (sklearn gradient boosting) to
classify each incoming prompt as FAST (direct, no reasoning) or THINK
(deep reasoning before action). The router was trained on 2,100 proxy war
trajectories across 210 benchmark tasks.

The router is lightweight (~450KB joblib file, no GPU needed) and runs
in <1ms per prediction. It uses text keyword features, inferred task
family, and difficulty estimation.

Integration point: called from Conversation.send() before dispatching to
the orchestrator. The result sets enable_thinking on the model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.log import get_logger

log = get_logger(__name__)

# Path to the trained router model (relative to local-operator project)
_ROUTER_PATH = Path.home() / "Projects/local-operator/models/learned_router.joblib"

# Keyword patterns for inferring task family from arbitrary prompts
_FAMILY_PATTERNS: dict[str, list[str]] = {
    "bug_fix": ["bug", "fix", "error", "crash", "traceback", "exception", "fail", "broken", "incorrect", "wrong", "issue", "regression"],
    "feature_add": ["add", "implement", "create", "build", "feature", "support", "new", "extend", "enable"],
    "refactor": ["refactor", "restructure", "reorganize", "cleanup", "clean up", "simplify", "extract", "move", "rename"],
    "git_tasks": ["git", "commit", "branch", "merge", "rebase", "conflict", "stash", "pull", "push", "diff", "blame"],
    "test_writing": ["test", "pytest", "unittest", "mock", "fixture", "coverage", "assert"],
    "type_errors": ["type", "mypy", "annotation", "generic", "protocol", "typing", "typevar", "pyright"],
    "packaging": ["package", "install", "dependency", "pip", "setup", "pyproject", "requirements", "poetry", "uv"],
    "ml_tasks": ["model", "train", "dataset", "tensor", "inference", "pytorch", "tensorflow", "embedding", "fine-tune", "finetune"],
    "terminal": ["shell", "command", "process", "pipe", "stdout", "stderr", "exec", "run command", "bash"],
    "research": ["search", "find", "grep", "locate", "investigate", "explore", "look up", "documentation", "docs"],
    "vision_conform": ["conform", "validate", "schema", "spec", "contract", "comply", "standard"],
    "hard_think": ["architecture", "design", "plan", "strategy", "analyze", "optimize", "migrate", "rewrite", "complex"],
    "easy_fast": ["list", "show", "display", "print", "echo", "cat", "ls", "status", "what is"],
    "trap": ["trick", "trap", "edge case", "corner case", "gotcha", "pitfall"],
}

# Difficulty estimation heuristics
_DIFFICULTY_INDICATORS = {
    "high": ["architecture", "design", "migrate", "rewrite", "refactor", "entire", "all", "complete", "comprehensive", "complex", "multi-step", "async", "concurrent", "distributed"],
    "low": ["list", "show", "print", "rename", "typo", "simple", "quick", "one line", "trivial"],
}


def _infer_family(prompt: str) -> str:
    """Infer task family from prompt text via keyword matching."""
    lowered = prompt.lower()
    scored: dict[str, int] = {}
    for family, keywords in _FAMILY_PATTERNS.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score > 0:
            scored[family] = score
    if not scored:
        return "unknown"
    return max(scored, key=scored.get)


def _infer_difficulty(prompt: str) -> int:
    """Estimate task difficulty (1-10) from prompt text."""
    lowered = prompt.lower()
    # Start at medium
    difficulty = 5
    # Adjust based on indicators
    for kw in _DIFFICULTY_INDICATORS["high"]:
        if kw in lowered:
            difficulty += 2
            break
    for kw in _DIFFICULTY_INDICATORS["low"]:
        if kw in lowered:
            difficulty -= 2
            break
    # Length-based adjustment: longer prompts tend to be more complex
    word_count = len(prompt.split())
    if word_count > 100:
        difficulty += 2
    elif word_count > 50:
        difficulty += 1
    elif word_count < 10:
        difficulty -= 1
    # Multiple sentences suggest complexity
    sentence_count = prompt.count(".") + prompt.count("!") + prompt.count("?")
    if sentence_count > 3:
        difficulty += 1
    # Clamp to 1-10
    return max(1, min(10, difficulty))


class AdaptiveRouter:
    """FAST/THINK mode router using the trained LearnedRouter model.

    The router is loaded once and cached. Each prediction takes <1ms.
    Falls back to heuristic routing if the model is unavailable.
    """

    def __init__(self, router_path: Path | None = None):
        self.router_path = router_path or _ROUTER_PATH
        self._router: Any = None
        self._loaded = False
        self._load_error: str | None = None

    def _load(self):
        """Lazily load the router model."""
        if self._loaded:
            return
        self._loaded = True
        try:
            # Add local-operator to path so we can import the router
            lo_path = str(Path.home() / "Projects/local-operator")
            if lo_path not in sys.path:
                sys.path.insert(0, lo_path)
            from local_operator.distillation.router import LearnedRouter

            router = LearnedRouter()
            router.load(str(self.router_path))
            self._router = router
            log.info("adaptive_router.loaded", path=str(self.router_path))
        except Exception as e:
            self._load_error = str(e)
            log.warning("adaptive_router.load_failed", error=str(e), fallback="heuristic")

    def route(self, prompt: str) -> tuple[str, float, dict[str, Any]]:
        """Route a user prompt to FAST or THINK mode.

        Returns:
            Tuple of (mode, confidence, metadata) where mode is "FAST" or "THINK",
            confidence is 0.0-1.0, and metadata contains the inferred family,
            difficulty, and routing method.
        """
        self._load()

        family = _infer_family(prompt)
        difficulty = _infer_difficulty(prompt)

        if self._router is not None:
            try:
                mode = self._router.predict(
                    task_description=prompt,
                    task_family=family,
                    task_difficulty=difficulty,
                )
                probs = self._router.predict_proba_record({
                    "task_description": prompt,
                    "task_family": family,
                    "task_difficulty": difficulty,
                })
                confidence = probs.get(mode, 0.5)
                log.info(
                    "adaptive_router.routed",
                    mode=mode,
                    confidence=confidence,
                    family=family,
                    difficulty=difficulty,
                    method="learned",
                )
                return mode, confidence, {
                    "family": family,
                    "difficulty": difficulty,
                    "method": "learned",
                    "probabilities": probs,
                }
            except Exception as e:
                log.warning("adaptive_router.predict_failed", error=str(e), fallback="heuristic")

        # Heuristic fallback: difficulty-based
        mode = "THINK" if difficulty >= 7 else "FAST"
        confidence = 0.6
        log.info(
            "adaptive_router.routed",
            mode=mode,
            confidence=confidence,
            family=family,
            difficulty=difficulty,
            method="heuristic",
        )
        return mode, confidence, {
            "family": family,
            "difficulty": difficulty,
            "method": "heuristic",
        }

    def is_available(self) -> bool:
        """Check if the trained router model is available."""
        self._load()
        return self._router is not None
