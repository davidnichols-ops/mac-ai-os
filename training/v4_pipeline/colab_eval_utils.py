"""Helper functions for the Colab evaluation notebook.

Provides response parsing (control tokens, tool calls), statistical
computations (Wilson CI, Fisher's exact test, Cohen's d), and result
formatting utilities. This module is self-contained (stdlib only) so it
can be copied to Google Drive and imported from a Colab notebook without
the full training package.

Usage in a Colab notebook::

    import sys
    sys.path.insert(0, "/content/drive/MyDrive/adaptive_operator_v4")
    from colab_eval_utils import (
        parse_control_token,
        parse_tool_calls,
        wilson_interval,
        fisher_exact_test,
        cohens_d,
        statistical_report,
        format_summary_table,
    )
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from typing import Any

# --------------------------------------------------------------------------- #
# Control tokens and tool-call parsing                                         #
# --------------------------------------------------------------------------- #

CONTROL_TOKENS = ["FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE"]
TOKEN_PATTERN = re.compile(r"^\[(FAST|THINK|VERIFY|RECOVER|ESCALATE)\]", re.IGNORECASE)

# Tool-call delimiters -- using chr() to avoid template confusion.
TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/tool_call" + chr(62)


def parse_control_token(text: str) -> tuple[str | None, str]:
    """Parse a leading control token from the model output.

    Returns ``(mode, remaining_text)`` where ``mode`` is one of
    ``CONTROL_TOKENS`` or ``None`` if no token was found.
    """
    stripped = text.lstrip()
    m = TOKEN_PATTERN.match(stripped)
    if m:
        mode = m.group(1).upper()
        remaining = stripped[m.end():].lstrip()
        return mode, remaining
    # Also handle <FAST> style tokens.
    for token in CONTROL_TOKENS:
        if stripped.startswith(f"<{token}>"):
            return token, stripped[len(token) + 2:].lstrip()
    return None, text


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from model text output.

    Supports three patterns:
      1. <tool_call> JSON blocks
      2. JSON in fenced code blocks (tool_call / tool_code / json)
      3. Bare JSON with name + arguments keys

    Returns a list of dicts with keys: id, name, arguments, valid, parse_error.
    """
    tool_calls: list[dict[str, Any]] = []
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Pattern 1: <tool_call> JSON blocks
    pat1 = TC_OPEN + r"\s*(\{.*?\})\s*" + TC_CLOSE
    for i, match in enumerate(re.findall(pat1, clean, re.DOTALL)):
        _try_parse_tool_call(match, i, tool_calls)
    if tool_calls:
        return tool_calls

    # Pattern 2: JSON in code blocks
    pat2 = r"```(?:tool_call|tool_code|json)\s*(\{.*?\})\s*```"
    for i, match in enumerate(re.findall(pat2, clean, re.DOTALL)):
        _try_parse_tool_call(match, i, tool_calls)
    if tool_calls:
        return tool_calls

    # Pattern 3: bare JSON with name + arguments
    pat3 = r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\})'
    for i, match in enumerate(re.findall(pat3, clean, re.DOTALL)):
        _try_parse_tool_call(match, i, tool_calls)

    return tool_calls


def _try_parse_tool_call(
    match: str, idx: int, tool_calls: list[dict[str, Any]]
) -> None:
    """Attempt to parse a JSON tool call match and append to the list."""
    try:
        data = json.loads(match)
        name = data.get("name", "")
        args = data.get("arguments", data.get("parameters", {}))
        if name:
            tool_calls.append({
                "id": f"call_{idx}",
                "name": name,
                "arguments": args if isinstance(args, dict) else {},
                "valid": True,
            })
    except (json.JSONDecodeError, ValueError) as e:
        tool_calls.append({
            "id": f"call_{idx}",
            "name": "",
            "arguments": {},
            "valid": False,
            "parse_error": str(e),
        })


def build_system_prompt(tools_schema: list[dict[str, Any]]) -> str:
    """Build the system prompt with tool definitions for the agent loop."""
    parts = [
        "You are an AI assistant that completes tasks by using tools.",
        "",
        "## Core Principles",
        "1. Inspect before acting. Read the current state before making changes.",
        "2. Verify after acting. Run tests, check files, confirm changes worked.",
        "3. Recover from failures. If a tool fails, try a different approach.",
        "4. Be efficient. Don't repeat the same action unnecessarily.",
        "5. Be honest. Don't claim success without verification.",
        "",
        "## Control Tokens",
        "Begin your response with one of these control tokens to indicate your mode:",
        "  [FAST]    -- for simple, direct actions",
        "  [THINK]   -- for complex tasks requiring reasoning",
        "  [VERIFY]  -- for verification and checking tasks",
        "  [RECOVER] -- for error recovery scenarios",
        "  [ESCALATE] -- when you need to escalate to the user",
        "",
        "## Tool Call Format",
        "To call a tool, output:",
        "",
        TC_OPEN,
        '{"name": "tool_name", "arguments": {"param": "value"}}',
        TC_CLOSE,
        "",
        "You can call multiple tools by emitting multiple " + TC_OPEN + " blocks.",
        "After each tool call, you will receive the result. Use it to decide your next action.",
        "When the task is complete, provide your final answer without any tool calls.",
        "",
        "## Available Tools",
    ]

    for t in tools_schema:
        name = t.get("name", "unknown")
        desc = t.get("description", "")
        params = t.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        parts.append(f"\n### {name}")
        parts.append(desc)
        if props:
            parts.append("Parameters:")
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else ""
                parts.append(f"  - {pname} ({ptype}{req}): {pdesc}")

    return "\n".join(parts)


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return the 12 tool schemas the v4 model has access to."""
    return [
        {
            "name": "shell",
            "description": "Run a shell command and return stdout, stderr, and exit code.",
            "parameters": {
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "file_read",
            "description": "Read the contents of a text file.",
            "parameters": {
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "file_write",
            "description": "Write text content to a file. Creates or overwrites the file.",
            "parameters": {
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "The content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "file_edit",
            "description": "Replace a specific string in a file with a new string. The old_string must appear exactly once.",
            "parameters": {
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "The string to find (must be unique)"},
                    "new_string": {"type": "string", "description": "The replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
        {
            "name": "file_list",
            "description": "List files and directories in a given path.",
            "parameters": {
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default: current directory)"},
                },
            },
        },
        {
            "name": "grep",
            "description": "Search for a pattern in file contents using ripgrep.",
            "parameters": {
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                    "glob": {"type": "string", "description": "Glob pattern to filter files"},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive search"},
                    "max_results": {"type": "integer", "description": "Maximum results (default: 100)"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "find_file",
            "description": "Find files by name pattern using glob matching.",
            "parameters": {
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '**/*.ts')"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "git",
            "description": "Run a git subcommand (status, diff, log, branch, etc.).",
            "parameters": {
                "properties": {
                    "command": {"type": "string", "description": "The git subcommand to run"},
                    "repo": {"type": "string", "description": "Path to the git repository"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web for information. Returns titles, URLs, and summaries.",
            "parameters": {
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default: 5)"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch a web page and return its content as text.",
            "parameters": {
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "todo_write",
            "description": "Create and manage a structured task list for tracking progress.",
            "parameters": {
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "List of task objects with 'content' and 'status' fields",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The task description"},
                                "status": {"type": "string", "description": "pending, in_progress, or completed"},
                            },
                        },
                    },
                },
                "required": ["todos"],
            },
        },
        {
            "name": "ask_user",
            "description": "Ask the user a question when you need clarification or a decision.",
            "parameters": {
                "properties": {
                    "question": {"type": "string", "description": "The question to ask"},
                    "options": {
                        "type": "array",
                        "description": "List of options for the user",
                        "items": {"type": "string"},
                    },
                },
                "required": ["question"],
            },
        },
    ]


# --------------------------------------------------------------------------- #
# Statistical computations (ported from eval/stats.py -- stdlib only)          #
# --------------------------------------------------------------------------- #

_Z_SCORES: dict[float, float] = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
    0.999: 3.290526731491247,
}


def _z_score(confidence: float) -> float:
    """Return the two-tailed z-score for a given confidence level."""
    if confidence in _Z_SCORES:
        return _Z_SCORES[confidence]
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    p = 1.0 - (1.0 - confidence) / 2.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549726517428572e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return x


def wilson_interval(passes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a pass rate."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if passes < 0 or passes > total:
        raise ValueError(f"passes must be in [0, total], got passes={passes}, total={total}")
    if total == 0:
        return (0.0, 0.0)
    z = _z_score(confidence)
    z2 = z * z
    n = total
    phat = passes / n
    denominator = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denominator
    margin = (z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))) / denominator
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (lower, upper)


def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def _log_binom_coeff(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)


def _hypergeometric_pmf(k: int, n: int, K: int, N: int) -> float:
    log_p = (
        _log_binom_coeff(K, k)
        + _log_binom_coeff(N - K, n - k)
        - _log_binom_coeff(N, n)
    )
    return math.exp(log_p)


def fisher_exact_test(
    v4_passes: int,
    v4_total: int,
    v3_passes: int,
    v3_total: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fisher's exact test for comparing two models on pass/fail outcomes."""
    for name, val in (("v4_passes", v4_passes), ("v4_total", v4_total),
                      ("v3_passes", v3_passes), ("v3_total", v3_total)):
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")
    if v4_passes > v4_total:
        raise ValueError(f"v4_passes ({v4_passes}) exceeds v4_total ({v4_total})")
    if v3_passes > v3_total:
        raise ValueError(f"v3_passes ({v3_passes}) exceeds v3_total ({v3_total})")

    a = v4_passes
    b = v4_total - v4_passes
    c = v3_passes
    d = v3_total - v3_passes
    n1 = v4_total
    n2 = v3_total
    s = a + c
    N = n1 + n2

    v4_rate = a / n1 if n1 > 0 else 0.0
    v3_rate = c / n2 if n2 > 0 else 0.0

    if b > 0 or c > 0:
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    else:
        odds_ratio = float("inf")

    lo = max(0, s - n2)
    hi = min(s, n1)
    p_observed = _hypergeometric_pmf(a, n1, s, N)
    p_value = 0.0
    for k in range(lo, hi + 1):
        p_k = _hypergeometric_pmf(k, n1, s, N)
        if p_k <= p_observed + 1e-15:
            p_value += p_k
    p_value = min(1.0, max(0.0, p_value))

    if v4_rate > v3_rate:
        direction = "v4_better"
    elif v4_rate < v3_rate:
        direction = "v3_better"
    else:
        direction = "no_difference"

    return {
        "p_value": p_value,
        "significant": p_value < alpha,
        "direction": direction,
        "v4_pass_rate": v4_rate,
        "v3_pass_rate": v3_rate,
        "odds_ratio": odds_ratio,
        "alpha": alpha,
    }


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2:
        raise ValueError(f"group1 must have at least 2 elements, got {len(group1)}")
    if len(group2) < 2:
        raise ValueError(f"group2 must have at least 2 elements, got {len(group2)}")
    n1, n2 = len(group1), len(group2)
    mean1 = statistics.fmean(group1)
    mean2 = statistics.fmean(group2)
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var <= 0.0:
        raise ValueError("pooled standard deviation is zero; groups have no variance")
    pooled_sd = math.sqrt(pooled_var)
    return (mean1 - mean2) / pooled_sd


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply the Bonferroni correction for multiple comparisons."""
    if not p_values:
        raise ValueError("p_values must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    m = len(p_values)
    adjusted_alpha = alpha / m
    return [p < adjusted_alpha for p in p_values]


def _bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean."""
    if not data:
        raise ValueError("data must be non-empty")
    rng = random.Random(seed)
    n = len(data)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int(math.floor((alpha / 2.0) * n_bootstrap)))
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)
    return (means[lower_idx], means[upper_idx])


def _pass_rate(results: list[dict]) -> tuple[int, int]:
    total = len(results)
    passes = sum(1 for r in results if r.get("passed", False))
    return (passes, total)


def _scores(results: list[dict]) -> list[float]:
    return [float(r.get("score", 1.0 if r.get("passed", False) else 0.0)) for r in results]


def _categories(results: list[dict]) -> list[str]:
    return sorted({r.get("category", "unknown") for r in results})


def _filter_by_category(results: list[dict], category: str) -> list[dict]:
    return [r for r in results if r.get("category", "unknown") == category]


def statistical_report(
    v4_results: list[dict],
    v3_results: list[dict],
    confidence: float = 0.95,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Produce a full statistical comparison report between v4.1 and v3.1.

    Combines Wilson confidence intervals, Fisher's exact test, bootstrap CIs,
    Cohen's d, and Bonferroni-corrected per-category significance.
    """
    v4_passes, v4_total = _pass_rate(v4_results)
    v3_passes, v3_total = _pass_rate(v3_results)
    v4_rate = v4_passes / v4_total if v4_total > 0 else 0.0
    v3_rate = v3_passes / v3_total if v3_total > 0 else 0.0
    v4_ci = wilson_interval(v4_passes, v4_total, confidence)
    v3_ci = wilson_interval(v3_passes, v3_total, confidence)

    fisher = fisher_exact_test(v4_passes, v4_total, v3_passes, v3_total, alpha=alpha)

    v4_scores = _scores(v4_results)
    v3_scores = _scores(v3_results)
    try:
        d_value = cohens_d(v4_scores, v3_scores)
    except ValueError:
        d_value = 0.0

    v4_boot: tuple[float, float] | None = None
    v3_boot: tuple[float, float] | None = None
    if v4_scores:
        v4_boot = _bootstrap_ci(v4_scores, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    if v3_scores:
        v3_seed = (seed + 1) if seed is not None else None
        v3_boot = _bootstrap_ci(v3_scores, n_bootstrap=n_bootstrap, confidence=confidence, seed=v3_seed)

    all_categories = sorted(set(_categories(v4_results)) | set(_categories(v3_results)))
    category_results: dict[str, Any] = {}
    category_p_values: list[float] = []
    category_names: list[str] = []

    for cat in all_categories:
        v4_cat = _filter_by_category(v4_results, cat)
        v3_cat = _filter_by_category(v3_results, cat)
        c_v4_passes, c_v4_total = _pass_rate(v4_cat)
        c_v3_passes, c_v3_total = _pass_rate(v3_cat)
        c_v4_rate = c_v4_passes / c_v4_total if c_v4_total > 0 else 0.0
        c_v3_rate = c_v3_passes / c_v3_total if c_v3_total > 0 else 0.0
        c_v4_ci = wilson_interval(c_v4_passes, c_v4_total, confidence)
        c_v3_ci = wilson_interval(c_v3_passes, c_v3_total, confidence)

        cat_fisher = fisher_exact_test(
            c_v4_passes, c_v4_total, c_v3_passes, c_v3_total, alpha=alpha,
        )
        category_p_values.append(cat_fisher["p_value"])
        category_names.append(cat)

        category_results[cat] = {
            "v4": {
                "passes": c_v4_passes,
                "total": c_v4_total,
                "pass_rate": c_v4_rate,
                "wilson_ci": list(c_v4_ci),
            },
            "v3": {
                "passes": c_v3_passes,
                "total": c_v3_total,
                "pass_rate": c_v3_rate,
                "wilson_ci": list(c_v3_ci),
            },
            "fisher_exact": cat_fisher,
            "bonferroni_significant": False,
        }

    if category_p_values:
        corrected = bonferroni_correction(category_p_values, alpha=alpha)
        significant_categories: list[str] = []
        for name, is_sig in zip(category_names, corrected, strict=True):
            category_results[name]["bonferroni_significant"] = is_sig
            if is_sig:
                significant_categories.append(name)
    else:
        significant_categories = []

    if fisher["significant"] and fisher["direction"] == "v4_better":
        verdict = "improvement"
    elif fisher["significant"] and fisher["direction"] == "v3_better":
        verdict = "regression"
    else:
        if abs(d_value) >= 0.5 and v4_rate > v3_rate:
            verdict = "improvement"
        elif abs(d_value) >= 0.5 and v4_rate < v3_rate:
            verdict = "regression"
        else:
            verdict = "no_change"

    return {
        "confidence": confidence,
        "alpha": alpha,
        "verdict": verdict,
        "overall": {
            "v4": {
                "passes": v4_passes,
                "total": v4_total,
                "pass_rate": v4_rate,
                "wilson_ci": list(v4_ci),
            },
            "v3": {
                "passes": v3_passes,
                "total": v3_total,
                "pass_rate": v3_rate,
                "wilson_ci": list(v3_ci),
            },
            "fisher_exact": fisher,
            "cohens_d": d_value,
            "bootstrap_ci_v4": list(v4_boot) if v4_boot else None,
            "bootstrap_ci_v3": list(v3_boot) if v3_boot else None,
        },
        "by_category": category_results,
        "bonferroni": {
            "alpha_adjusted": alpha / len(category_p_values) if category_p_values else alpha,
            "n_tests": len(category_p_values),
            "significant_categories": significant_categories,
        },
    }


# --------------------------------------------------------------------------- #
# Result formatting                                                            #
# --------------------------------------------------------------------------- #


def format_summary_table(
    v41_results: list[dict],
    v31_results: list[dict],
    stat_report: dict[str, Any],
) -> str:
    """Format a human-readable summary comparison table.

    Shows per-category pass rates, deltas, and significance, plus overall.
    """
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("  EVALUATION SUMMARY: v4.1 vs v3.1")
    lines.append("=" * 90)
    lines.append("")

    overall = stat_report["overall"]
    v4 = overall["v4"]
    v3 = overall["v3"]
    fe = overall["fisher_exact"]

    lines.append("  Overall Pass Rates:")
    lines.append(
        f"    v4.1:  {v4['passes']}/{v4['total']} = {v4['pass_rate']:.1%}  "
        f"Wilson CI [{v4['wilson_ci'][0]:.3f}, {v4['wilson_ci'][1]:.3f}]"
    )
    lines.append(
        f"    v3.1:  {v3['passes']}/{v3['total']} = {v3['pass_rate']:.1%}  "
        f"Wilson CI [{v3['wilson_ci'][0]:.3f}, {v3['wilson_ci'][1]:.3f}]"
    )
    lines.append("")
    lines.append(f"  Fisher's exact test:  p={fe['p_value']:.4f}  "
                 f"significant={fe['significant']}  direction={fe['direction']}")
    lines.append(f"  Cohen's d:            {overall['cohens_d']:.3f}")
    lines.append(f"  Verdict:              {stat_report['verdict'].upper()}")
    lines.append("")

    # Per-category table
    lines.append("  Per-Category Breakdown:")
    header = (
        f"    {'Category':<28s} {'v4.1 Rate':>10s} {'v3.1 Rate':>10s} "
        f"{'Delta':>8s} {'p-value':>10s} {'Sig?':>6s}"
    )
    lines.append(header)
    lines.append(f"    {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 6}")

    for cat in sorted(stat_report["by_category"].keys()):
        stats = stat_report["by_category"][cat]
        v4r = stats["v4"]["pass_rate"]
        v3r = stats["v3"]["pass_rate"]
        delta = (v4r - v3r) * 100
        p = stats["fisher_exact"]["p_value"]
        sig = "YES" if stats["bonferroni_significant"] else "no"
        lines.append(
            f"    {cat:<28s} {v4r:>9.1%} {v3r:>9.1%} "
            f"{delta:>+7.1f}% {p:>10.4f} {sig:>6s}"
        )

    # Overall row
    delta_overall = (v4["pass_rate"] - v3["pass_rate"]) * 100
    lines.append(f"    {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 6}")
    lines.append(
        f"    {'OVERALL':<28s} {v4['pass_rate']:>9.1%} {v3['pass_rate']:>9.1%} "
        f"{delta_overall:>+7.1f}% {fe['p_value']:>10.4f} "
        f"{'YES' if fe['significant'] else 'no':>6s}"
    )
    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


def format_report_text(report: dict[str, Any]) -> str:
    """Format a full statistical report dict as a human-readable string."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("  STATISTICAL COMPARISON REPORT: v4.1 vs v3.1")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Confidence: {report['confidence']:.0%}  Alpha: {report['alpha']}")
    lines.append(f"  Verdict:    {report['verdict'].upper()}")
    lines.append("")

    overall = report["overall"]
    v4 = overall["v4"]
    v3 = overall["v3"]
    lines.append("  Overall Pass Rates:")
    lines.append(
        f"    v4.1: {v4['passes']}/{v4['total']} = {v4['pass_rate']:.1%}  "
        f"Wilson CI [{v4['wilson_ci'][0]:.3f}, {v4['wilson_ci'][1]:.3f}]"
    )
    lines.append(
        f"    v3.1: {v3['passes']}/{v3['total']} = {v3['pass_rate']:.1%}  "
        f"Wilson CI [{v3['wilson_ci'][0]:.3f}, {v3['wilson_ci'][1]:.3f}]"
    )
    lines.append("")

    fe = overall["fisher_exact"]
    lines.append("  Fisher's Exact Test (overall):")
    lines.append(f"    p-value:     {fe['p_value']:.4f}")
    lines.append(f"    significant: {fe['significant']}")
    lines.append(f"    direction:   {fe['direction']}")
    lines.append(f"    odds ratio:  {fe['odds_ratio']:.3f}")
    lines.append("")

    lines.append(f"  Cohen's d: {overall['cohens_d']:.3f}")
    if overall.get("bootstrap_ci_v4") and overall.get("bootstrap_ci_v3"):
        lines.append(
            f"  Bootstrap CI (mean score):  v4.1 [{overall['bootstrap_ci_v4'][0]:.3f}, "
            f"{overall['bootstrap_ci_v4'][1]:.3f}]  v3.1 [{overall['bootstrap_ci_v3'][0]:.3f}, "
            f"{overall['bootstrap_ci_v3'][1]:.3f}]"
        )
    lines.append("")

    bonf = report["bonferroni"]
    lines.append("  Bonferroni Correction:")
    lines.append(f"    Tests: {bonf['n_tests']}  Adjusted alpha: {bonf['alpha_adjusted']:.4f}")
    if bonf["significant_categories"]:
        lines.append(f"    Significant categories: {', '.join(bonf['significant_categories'])}")
    else:
        lines.append("    No categories significant after correction.")
    lines.append("")

    lines.append("  Per-Category Breakdown:")
    lines.append(
        f"    {'Category':<30s} {'v4.1 Rate':>10s} {'v3.1 Rate':>10s} {'p-value':>10s} {'Sig':>5s}"
    )
    lines.append(f"    {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 5}")
    for cat, stats in sorted(report["by_category"].items()):
        v4r = stats["v4"]["pass_rate"]
        v3r = stats["v3"]["pass_rate"]
        p = stats["fisher_exact"]["p_value"]
        sig = "YES" if stats["bonferroni_significant"] else "no"
        lines.append(f"    {cat:<30s} {v4r:>9.1%} {v3r:>9.1%} {p:>10.4f} {sig:>5s}")
    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


__all__ = [
    "CONTROL_TOKENS",
    "parse_control_token",
    "parse_tool_calls",
    "build_system_prompt",
    "get_tool_schemas",
    "wilson_interval",
    "fisher_exact_test",
    "cohens_d",
    "bonferroni_correction",
    "statistical_report",
    "format_summary_table",
    "format_report_text",
]
