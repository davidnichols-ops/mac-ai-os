"""Qwen v3.1 inference client for the post-training pipeline.

Calls the adaptive-operator-v3-1-eval model hosted on Together AI via the
OpenAI-compatible API. The system prompt teaches the model about five control
tokens ([FAST], [THINK], [VERIFY], [RECOVER], [ESCALATE]) and the twelve MAOS
tools. Tool calls are emitted as XML-style tags wrapping JSON payloads.

Key entry points:
    - generate_response(): single-turn inference with control-token + tool-call parsing
    - batch_generate(): stream prompts from a JSONL file, write responses to a JSONL
      file with resume support (skips ids already present in the output file)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

# --- Model / endpoint configuration ---

DEFAULT_MODEL = "davidnicholsop-199a/adaptive-operator-v3-1-eval"
DEFAULT_BASE_URL = "https://api-inference.together.ai/v1"
DEFAULT_ENV_FILE = os.path.expanduser("~/Projects/local-operator/.env")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # exponential backoff base in seconds

# --- Control tokens ---

CONTROL_TOKENS = ("FAST", "THINK", "VERIFY", "RECOVER", "ESCALATE")
_CONTROL_TOKEN_RE = re.compile(
    r"^\s*\[(?:" + "|".join(CONTROL_TOKENS) + r")\]",
    re.IGNORECASE,
)

# --- Tool call parsing ---
# The model wraps tool calls in open/close XML tags containing a JSON object
# with "name" and "arguments" fields. We capture the inner JSON (non-greedy)
# and tolerate malformed payloads (skipping them rather than failing).
_TC_OPEN = chr(60) + "tool_call" + chr(62)
_TC_CLOSE = chr(60) + "/tool_call" + chr(62)
_TOOL_CALL_RE = re.compile(
    re.escape(_TC_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(_TC_CLOSE),
    re.DOTALL,
)


# --- System prompt ---

_SYSTEM_PROMPT_HEAD = """\
You are Adaptive Operator v3.1, an autonomous coding and operations agent running
inside the Mac AI OS (MAOS) post-training pipeline. You operate by selecting a
control mode and emitting tool calls that the host executes on your behalf.

## Control tokens

EVERY response MUST begin with exactly one control token enclosed in square
brackets. The token selects your operating mode for that turn. Choose the most
appropriate token based on the task:

[FAST]    - Simple, direct, single-step actions. Use for trivial file ops,
            shell commands, git queries, and straightforward edits where the
            answer is obvious and low-risk.
[THINK]   - Multi-step reasoning, design, planning, or non-trivial code
            generation. Use when the task requires analysis before acting.
[VERIFY]  - Actions whose results must be checked before proceeding. Use for
            destructive ops, deployments, or any step where confirming success
            matters (e.g. delete, revert, restart).
[RECOVER] - Something went wrong and you must diagnose and fix it. Use when
            an error occurred, a build broke, or a previous attempt failed.
[ESCALATE]- The situation exceeds your safe operating envelope or requires a
            human decision. Use for security incidents, irreversible changes,
            or when you lack sufficient information to proceed safely.

After the control token, write your reasoning/plan (concise) and then emit any
tool calls needed to accomplish the task. If no tool call is needed, just answer
directly after the token.

## Available tools

You have access to the following 12 tools. Emit a tool call using the format:

"""

_SYSTEM_PROMPT_TAIL = """

## Output rules

- ALWAYS start with one control token.
- Emit zero or more tool calls as needed.
- Keep reasoning concise; prefer acting over narrating.
- Never fabricate tool results; you only request actions, the host executes them.
"""

_TOOLS_DESCRIPTION = """\
1. shell
   Execute a shell command and return stdout/stderr.
   arguments: {"command": "<shell command>"}

2. file_read
   Read the contents of a file.
   arguments: {"file_path": "<absolute path>"}

3. file_write
   Write content to a file (creates or overwrites).
   arguments: {"file_path": "<absolute path>", "content": "<file content>"}

4. file_edit
   Replace a substring within a file.
   arguments: {"file_path": "<absolute path>", "old_string": "<text>", "new_string": "<text>"}

5. file_list
   List files in a directory.
   arguments: {"path": "<directory path>", "pattern": "<optional glob>"}

6. grep
   Search file contents with a regular expression.
   arguments: {"pattern": "<regex>", "path": "<directory or file>", "glob_pattern": "<optional glob>"}

7. find_file
   Find files by name pattern.
   arguments: {"pattern": "<glob>", "path": "<directory>"}

8. git
   Run a git subcommand.
   arguments: {"command": "<git subcommand>", "repo": "<optional repo path>"}

9. web_search
   Search the web for a query.
   arguments: {"query": "<search query>", "num_results": "<optional int>"}

10. web_fetch
    Fetch the content of a web page.
    arguments: {"url": "<url>"}

11. todo_write
    Write or replace the task list.
    arguments: {"todos": [{"content": "<task>", "status": "pending|in_progress|completed"}]}

12. ask_user
    Ask the user a clarifying question.
    arguments: {"question": "<question text>"}
"""


def _build_system_prompt() -> str:
    """Assemble the full system prompt with the tool-call format example."""
    example = _TC_OPEN + "\n" + '{"name": "tool_name", "arguments": {"arg1": "value1"}}' + "\n" + _TC_CLOSE
    return _SYSTEM_PROMPT_HEAD + example + "\n\n" + _TOOLS_DESCRIPTION + _SYSTEM_PROMPT_TAIL


SYSTEM_PROMPT = _build_system_prompt()


# --- Response data structures ---


@dataclass
class ToolCall:
    """A single parsed tool call."""

    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class InferenceResult:
    """Parsed response from the Qwen v3.1 model."""

    text: str
    mode: Optional[str]  # the control token, uppercased; None if missing
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    latency: float = 0.0  # seconds
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None  # set if the call ultimately failed

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


# --- API key loading ---


def load_api_key(env_file: str = DEFAULT_ENV_FILE) -> str:
    """Load the Together AI API key from a .env file or the environment."""
    # Environment takes precedence.
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        return key.strip()
    # Fall back to parsing the .env file.
    p = Path(env_file)
    if not p.exists():
        raise FileNotFoundError(
            f"TOGETHER_API_KEY not set and env file not found: {env_file}"
        )
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("TOGETHER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise KeyError(f"TOGETHER_API_KEY not found in {env_file}")


# --- Parsing helpers ---


def parse_control_token(text: str) -> tuple[Optional[str], str]:
    """Extract the leading control token and return (mode, remainder).

    Returns (None, text) if no control token is present at the start.
    """
    match = _CONTROL_TOKEN_RE.match(text)
    if not match:
        return None, text
    token = match.group(0).strip().strip("[]").upper()
    remainder = text[match.end():].lstrip()
    return token, remainder


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse all tool-call XML blocks from the text.

    Malformed JSON payloads are skipped (logged at debug level) rather than
    raising, so a single bad block does not discard the whole response.
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed tool_call payload: %r", raw)
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=name, arguments=args))
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool-call XML blocks from the text, leaving prose only."""
    return _TOOL_CALL_RE.sub("", text).strip()


# --- Core inference ---


def _extract_message_text(message: Any) -> str:
    """Extract the text content from a chat completion message.

    Qwen reasoning models on Together AI may place the full response in
    ``reasoning_content`` while leaving ``content`` null. We check both
    fields (and a few fallbacks) so the response is never silently lost.
    """
    # Primary: the standard OpenAI content field.
    content = getattr(message, "content", None)
    if content:
        return content
    # Fallback 1: Together AI / Qwen reasoning_content field.
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        return reasoning
    # Fallback 2: check the model_extra dict for non-standard fields.
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning", "content"):
            val = extra.get(key)
            if val:
                return val
    return ""


class QwenInferenceClient:
    """Client wrapping the OpenAI-compatible Together AI endpoint for Qwen v3.1."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        env_file: str = DEFAULT_ENV_FILE,
        timeout: float = DEFAULT_TIMEOUT,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        if api_key is None:
            api_key = load_api_key(env_file=env_file)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Single chat completion call (no retry)."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_response(
        self,
        prompt: str,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        retries: int = MAX_RETRIES,
    ) -> InferenceResult:
        """Generate a single response for a user prompt.

        Retries up to `retries` times with exponential backoff on transient
        API errors (rate limits, timeouts, generic API errors). Returns an
        InferenceResult with `error` set if all retries are exhausted.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        last_error: Optional[str] = None
        for attempt in range(1, retries + 1):
            start = time.monotonic()
            try:
                completion = self._chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (RateLimitError, APITimeoutError, APIError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    backoff = BACKOFF_BASE ** attempt
                    logger.warning(
                        "API error on attempt %d/%d for model %s: %s (retrying in %.1fs)",
                        attempt,
                        retries,
                        self.model,
                        last_error,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                logger.error(
                    "API call failed after %d attempts: %s", retries, last_error
                )
                break
            except Exception as exc:  # pragma: no cover - unexpected
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Unexpected error during inference")
                break

            latency = time.monotonic() - start
            choice = completion.choices[0]
            raw_text = _extract_message_text(choice.message)
            mode, remainder = parse_control_token(raw_text)
            tool_calls = parse_tool_calls(raw_text)
            usage = getattr(completion, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            return InferenceResult(
                text=remainder if mode else raw_text,
                mode=mode,
                tool_calls=tool_calls,
                finish_reason=getattr(choice, "finish_reason", None),
                latency=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        return InferenceResult(text="", mode=None, error=last_error)


# --- Convenience module-level function ---


_default_client: Optional[QwenInferenceClient] = None


def _get_default_client() -> QwenInferenceClient:
    global _default_client
    if _default_client is None:
        _default_client = QwenInferenceClient()
    return _default_client


def generate_response(
    prompt: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = MAX_RETRIES,
) -> InferenceResult:
    """Module-level convenience wrapper around QwenInferenceClient.generate_response."""
    return _get_default_client().generate_response(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )


# --- Batch generation with resume ---


def _load_processed_ids(output_path: Path) -> set[str]:
    """Read an existing output JSONL and return the set of processed prompt ids."""
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id") or rec.get("prompt_id")
            if pid:
                ids.add(pid)
    return ids


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue


def batch_generate(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    env_file: str = DEFAULT_ENV_FILE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    limit: Optional[int] = None,
    resume: bool = True,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Read prompts from a JSONL file and write responses to a JSONL file.

    Each input record is expected to have at least {"prompt": "...", "id": "..."}
    (as produced by prompt_generator). Each output record contains the original
    prompt fields plus a "response" object with the parsed InferenceResult.

    Resume support: when resume=True, prompt ids already present in the output
    file are skipped. The output file is appended to (not overwritten) so a
    partial run can be continued safely.

    Returns a summary dict with counts and timing.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = _load_processed_ids(output_path) if resume else set()
    if processed:
        logger.info("Resume: %d prompts already processed in %s", len(processed), output_path)

    client = QwenInferenceClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        env_file=env_file,
    )

    total = 0
    written = 0
    skipped = 0
    errors = 0
    mode_counts: dict[str, int] = {}
    tool_call_count = 0
    total_latency = 0.0
    total_tokens = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for rec in _iter_jsonl(input_path):
            if limit is not None and written >= limit:
                break
            total += 1
            pid = rec.get("id") or rec.get("prompt_id")
            if pid and pid in processed:
                skipped += 1
                continue
            prompt_text = rec.get("prompt")
            if not prompt_text:
                logger.warning("Skipping record without a prompt: %s", pid)
                continue

            result = client.generate_response(
                prompt_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            out_rec = dict(rec)
            out_rec["response"] = result.to_dict()
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

            if result.error:
                errors += 1
            if result.mode:
                mode_counts[result.mode] = mode_counts.get(result.mode, 0) + 1
            tool_call_count += len(result.tool_calls)
            total_latency += result.latency
            total_tokens += result.total_tokens

            if written % progress_every == 0:
                avg_lat = total_latency / written if written else 0.0
                logger.info(
                    "Progress: written=%d skipped=%d errors=%d avg_latency=%.2fs tokens=%d",
                    written,
                    skipped,
                    errors,
                    avg_lat,
                    total_tokens,
                )

    summary = {
        "total_read": total,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "mode_counts": mode_counts,
        "tool_calls": tool_call_count,
        "total_latency": round(total_latency, 3),
        "total_tokens": total_tokens,
        "output_path": str(output_path),
    }
    logger.info("Batch complete: %s", summary)
    return summary


__all__ = [
    "QwenInferenceClient",
    "InferenceResult",
    "ToolCall",
    "SYSTEM_PROMPT",
    "CONTROL_TOKENS",
    "generate_response",
    "batch_generate",
    "parse_control_token",
    "parse_tool_calls",
    "strip_tool_calls",
    "load_api_key",
]
