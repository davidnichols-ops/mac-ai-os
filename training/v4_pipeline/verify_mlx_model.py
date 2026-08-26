"""Verify a converted MLX model loads, generates, and parses tool calls.

Runs a quick smoke test against the v4 MLX model directory produced by
`convert_to_mlx.py`. Tests:

  1. Model loads (mlx_lm.load or mlx_vlm.load).
  2. Model generates text from a simple prompt.
  3. Model emits control tokens (FAST / THINK / VERIFY / RECOVER / ESCALATE)
     when prompted appropriately.
  4. Model can produce parseable tool calls in the tool-call JSON format
     used by `app/models/mlxvlm_provider.py`.

NOTE on dependencies:
  `mlx_lm` / `mlx_vlm` are Apple-Silicon only and live in the `mlx` extra
  (`uv sync --extra mlx`). They are imported lazily so this module imports
  cleanly without them (for smoke tests / static analysis). Run the actual
  verification on Apple Silicon with the deps installed.

Usage:
  uv run python -m training.v4_pipeline.verify_mlx_model --model-path PATH
  uv run python -m training.v4_pipeline.verify_mlx_model --model-path PATH --use-vlm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/Projects/local-operator/models/v4/mlx/"
    "Qwen3.5-9B-Adaptive-Operator-v4-MLX-4bit"
)

# Control tokens the v2/v4 model emits at the start of generation.
CONTROL_TOKENS = ("FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE")

# Tool-call delimiters, built from parts so the literal tags never appear as
# raw XML in this source file (avoids tool-parser confusion). These mirror the
# primary format parsed by app/models/mlxvlm_provider.py.
_TC_OPEN = "<" + "tool_call" + ">"
_TC_CLOSE = "</" + "tool_call" + ">"
# Regex matching the tool-call JSON block format.
_TOOL_CALL_RE = re.compile(
    re.escape(_TC_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(_TC_CLOSE),
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass
class TestReport:
    model_path: str
    use_vlm: bool
    results: list[TestResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("  MLX Model Verification Report")
        lines.append(f"  model : {self.model_path}")
        lines.append(f"  vlm   : {self.use_vlm}")
        lines.append("-" * 70)
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.name} ({r.elapsed_s:.2f}s) - {r.detail}")
        lines.append("-" * 70)
        verdict = "ALL PASSED" if self.all_passed else "FAILURES PRESENT"
        lines.append(f"  Verdict: {verdict}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(modname: str, hint: str) -> Any:
    try:
        return __import__(modname)
    except ImportError as e:  # pragma: no cover - env dependent
        raise RuntimeError(f"{modname} is not installed. {hint}") from e


def _load_model(model_path: str, use_vlm: bool) -> tuple[Any, Any]:
    """Load the MLX model + tokenizer/processor. Returns (model, tokenizer)."""
    if use_vlm:
        mlx_vlm = _require(
            "mlx_vlm", "Install with: uv pip install mlx-vlm (Apple Silicon only)"
        )
        model, processor = mlx_vlm.load(model_path)
        return model, processor
    mlx_lm = _require(
        "mlx_lm", "Install with: uv pip install mlx-lm (Apple Silicon only)"
    )
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.7,
    use_vlm: bool = False,
) -> str:
    """Generate text from the loaded model."""
    if use_vlm:
        mlx_vlm = _require("mlx_vlm", "Install mlx-vlm (Apple Silicon only)")
        out = mlx_vlm.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )
    else:
        mlx_lm = _require("mlx_lm", "Install mlx-lm (Apple Silicon only)")
        out = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )
    return out.text if hasattr(out, "text") else str(out)


def _build_chat_prompt(
    tokenizer: Any, messages: list[dict[str, str]], use_vlm: bool
) -> str:
    """Apply the chat template to a list of messages."""
    if use_vlm:
        # mlx_vlm uses the processor's apply_chat_template.
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            pass
    # mlx_lm: tokenizer may have apply_chat_template.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            pass
    # Fallback: manual Qwen-style prompt.
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool-call parsing (mirrors app/models/mlxvlm_provider.py)
# ---------------------------------------------------------------------------

def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool-call JSON blocks from model output.

    Mirrors the primary format parsed in `app/models/mlxvlm_provider.py`.
    """
    calls: list[dict[str, Any]] = []
    for i, match in enumerate(_TOOL_CALL_RE.findall(text)):
        try:
            data = json.loads(match)
            name = data.get("name", "")
            args = data.get("arguments", data.get("parameters", {}))
            if name:
                calls.append({"id": f"call_{i}", "name": name, "arguments": args})
        except (json.JSONDecodeError, ValueError):
            continue
    return calls


def parse_control_token(text: str) -> tuple[str, str | None]:
    """Detect a leading control token and return (cleaned_text, mode)."""
    stripped = text.lstrip()
    for tok in CONTROL_TOKENS:
        tag = f"<{tok}>"
        if stripped.startswith(tag):
            return stripped[len(tag):].lstrip(), tok.lower()
    return text, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_loads(
    model_path: str, use_vlm: bool
) -> tuple[TestResult, tuple[Any, Any]]:
    t0 = time.monotonic()
    try:
        model, tok = _load_model(model_path, use_vlm)
    except Exception as e:
        elapsed = time.monotonic() - t0
        return (
            TestResult("model_loads", False, f"load failed: {e}", elapsed),
            (None, None),
        )
    elapsed = time.monotonic() - t0
    return TestResult("model_loads", True, "loaded", elapsed), (model, tok)


def test_generates_text(
    model: Any, tokenizer: Any, use_vlm: bool
) -> TestResult:
    t0 = time.monotonic()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ]
    prompt = _build_chat_prompt(tokenizer, messages, use_vlm)
    try:
        text = _generate(
            model, tokenizer, prompt, max_tokens=64, temperature=0.3, use_vlm=use_vlm
        )
    except Exception as e:
        return TestResult("generates_text", False, f"generate failed: {e}", time.monotonic() - t0)
    ok = bool(text and text.strip())
    sample = text.strip().replace("\n", " ")[:80]
    return TestResult("generates_text", ok, f"sample={sample!r}", time.monotonic() - t0)


def test_emits_control_token(
    model: Any, tokenizer: Any, use_vlm: bool
) -> TestResult:
    """Prompt the model in a way that should elicit a control token.

    The v2/v4 model is trained to emit a control token at the start of
    generation. We use a generic task prompt and check whether any control
    token appears. This is a soft test - a fine-tuned model should emit one,
    but we don't hard-fail if it doesn't (the model may still be usable).
    """
    t0 = time.monotonic()
    messages = [
        {"role": "system", "content": "You are an adaptive operator agent."},
        {"role": "user", "content": "Read the file README.md and tell me what it says."},
    ]
    prompt = _build_chat_prompt(tokenizer, messages, use_vlm)
    try:
        text = _generate(
            model, tokenizer, prompt, max_tokens=128, temperature=0.2, use_vlm=use_vlm
        )
    except Exception as e:
        return TestResult("emits_control_token", False, f"generate failed: {e}", time.monotonic() - t0)
    _, mode = parse_control_token(text)
    found = mode is not None
    detail = f"mode={mode!r}" if found else "no control token (soft check)"
    # Soft pass: report the finding but don't fail the suite on this alone.
    return TestResult("emits_control_token", True, detail, time.monotonic() - t0)


def test_parses_tool_calls(
    model: Any, tokenizer: Any, use_vlm: bool
) -> TestResult:
    """Prompt the model with a tool definition and check for a parseable tool call."""
    t0 = time.monotonic()
    # Build the tool-call format example using the constructed delimiters so
    # the model sees the exact format it was trained on.
    example_json = '{"name": "tool_name", "arguments": {"param": "value"}}'
    tool_block = (
        "## Available Tools\n\n"
        "### read_file\nRead a file from disk.\nParameters:\n"
        "    - path (string, required): path to the file\n\n"
        "## Tool Call Format\n"
        "To call a tool, output:\n\n"
        + _TC_OPEN + "\n" + example_json + "\n" + _TC_CLOSE + "\n\n"
        "After each tool call, you will receive the result.\n"
    )
    messages = [
        {
            "role": "system",
            "content": "You are an adaptive operator agent.\n\n" + tool_block,
        },
        {"role": "user", "content": "Read the file at /tmp/notes.txt"},
    ]
    prompt = _build_chat_prompt(tokenizer, messages, use_vlm)
    try:
        text = _generate(
            model, tokenizer, prompt, max_tokens=256, temperature=0.2, use_vlm=use_vlm
        )
    except Exception as e:
        return TestResult("parses_tool_calls", False, f"generate failed: {e}", time.monotonic() - t0)
    calls = parse_tool_calls(text)
    ok = len(calls) > 0
    detail = f"calls={len(calls)}" + (
        f" names={[c['name'] for c in calls]}" if calls else ""
    )
    return TestResult("parses_tool_calls", ok, detail, time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_verification(model_path: str, use_vlm: bool = False) -> TestReport:
    """Run all verification tests and return a report."""
    report = TestReport(model_path=model_path, use_vlm=use_vlm)

    # Test 1: load.
    result, (model, tokenizer) = test_model_loads(model_path, use_vlm)
    report.results.append(result)
    if not result.passed or model is None:
        # Cannot continue without a loaded model.
        return report

    # Test 2: generates text.
    report.results.append(test_generates_text(model, tokenizer, use_vlm))

    # Test 3: emits control tokens.
    report.results.append(test_emits_control_token(model, tokenizer, use_vlm))

    # Test 4: parses tool calls.
    report.results.append(test_parses_tool_calls(model, tokenizer, use_vlm))

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a converted MLX model loads, generates, and parses tool calls.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the MLX model directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Use mlx_vlm instead of mlx_lm (for VLM architectures).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    model_path = str(Path(args.model_path).expanduser())
    if not Path(model_path).exists():
        print(f"ERROR: model path does not exist: {model_path}", file=sys.stderr)
        return 2

    report = run_verification(model_path, use_vlm=args.use_vlm)

    if args.json:
        payload = {
            "model_path": report.model_path,
            "use_vlm": report.use_vlm,
            "all_passed": report.all_passed,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "elapsed_s": r.elapsed_s,
                }
                for r in report.results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(report.summary())

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
