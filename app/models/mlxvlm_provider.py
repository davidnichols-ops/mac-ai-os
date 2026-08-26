"""MLX-VLM provider — for VLM-architecture models that mlx-lm can't load.

Some models (like the fine-tuned Qwen3.5-9B-Adaptive-Operator) use a VLM
architecture (Qwen3_5ForConditionalGeneration) that requires mlx-vlm instead
of mlx-lm. This provider wraps mlx-vlm's load/generate API behind the
standard MAOS Model interface.

Supports adaptive thinking mode (FAST/THINK) via two mechanisms:
1. Control tokens (v2): The model emits <FAST> or <THINK> at the start of
   generation. The provider reads this token to determine mode, then strips
   it from the output. No external routing needed.
2. enable_thinking flag (v1 compat): Explicitly set thinking mode. The
   provider strips thinking blocks via regex when thinking is disabled.

Tool calling: When tools are provided, tool definitions are injected into
the system prompt as formatted text. The model emits tool calls in a
structured format (XML-style tags, JSON, or pipe-delimited). The provider
parses these into ToolCall objects on ModelResponse.tool_calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.log import get_logger
from app.models.base import Model
from app.models.types import Message, ModelResponse, ToolCall

log = get_logger(__name__)


def _import_mlx_vlm():
    try:
        import mlx_vlm  # type: ignore
        return mlx_vlm
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "mlx-vlm is not installed. Install with: pip install mlx-vlm"
        ) from e


class MLXVLMModel(Model):
    """A chat/generation model served by mlx-vlm (for VLM architectures)."""

    provider = "mlxvlm"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        enable_thinking: bool | None = None,
    ):
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._mlx = _import_mlx_vlm()
        self._model = None
        self._processor = None
        log.info("mlxvlm.model.requested", model=model, thinking=enable_thinking)

    def _load(self):
        if self._model is not None:
            return
        log.info("mlxvlm.model.loading", model=self.name)
        self._model, self._processor = self._mlx.load(self.name)
        log.info("mlxvlm.model.loaded", model=self.name)

    def _build_prompt(self, messages: list[Message]) -> str:
        """Apply chat template from the processor."""
        chat_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        try:
            if hasattr(self._processor, "apply_chat_template"):
                prompt = self._processor.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt = "\n".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                    for m in chat_messages
                )
                prompt += "\n<|im_start|>assistant\n"
        except Exception:
            prompt = "\n".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                for m in chat_messages
            )
            prompt += "\n<|im_start|>assistant\n"
        return prompt

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
        self._load()
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, **kwargs)

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ModelResponse:
        import time
        t0 = time.monotonic()
        self._load()
        t_load = time.monotonic()
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        # Allow per-call override of thinking mode
        thinking = kwargs.pop("enable_thinking", self.enable_thinking)

        # If tools are provided, inject tool definitions into the conversation
        effective_messages = messages
        if tools:
            effective_messages = self._inject_tools(messages, tools)

        prompt = self._build_prompt(effective_messages)
        t_prompt = time.monotonic()

        gen_kwargs: dict = {"max_tokens": max_tokens, "temperature": temp}
        # mlx_vlm.generate accepts enable_thinking directly
        if thinking is not None:
            gen_kwargs["enable_thinking"] = thinking

        result = self._mlx.generate(
            self._model,
            self._processor,
            prompt=prompt,
            **gen_kwargs,
        )
        t_gen = time.monotonic()

        text = result.text if hasattr(result, "text") else str(result)

        # Parse control tokens (v2 model emits <FAST> or <THINK> at start)
        mode = None
        text, mode = self._parse_control_token(text)

        # If no control token was found, fall back to enable_thinking flag
        if mode is None:
            mode = "think" if thinking else "fast" if thinking is False else None

        # Strip thinking blocks if in FAST mode (either via control token or flag)
        if mode == "fast":
            text = self._strip_thinking(text)
        t_strip = time.monotonic()

        # Parse tool calls from the cleaned text
        tool_calls = self._parse_tool_calls(text) if tools else []
        finish_reason = "tool_calls" if tool_calls else "stop"

        gen_time_s = t_gen - t_prompt
        completion_tokens = len(text.split())
        tps = completion_tokens / gen_time_s if gen_time_s > 0 else 0.0

        log.info(
            "mlxvlm.timing",
            load_ms=round((t_load - t0) * 1000),
            prompt_ms=round((t_prompt - t_load) * 1000),
            gen_ms=round((t_gen - t_prompt) * 1000),
            strip_ms=round((t_strip - t_gen) * 1000),
            total_ms=round((t_strip - t0) * 1000),
            output_tokens=completion_tokens,
            thinking=thinking,
            mode=mode,
            tool_calls=len(tool_calls),
            finish_reason=finish_reason,
        )

        return ModelResponse(
            text=text,
            model=self.name,
            provider=self.provider,
            raw=result,
            mode=mode,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            generation_time_s=gen_time_s,
            completion_tokens=completion_tokens,
            tokens_per_second=tps,
        )

    @staticmethod
    def _inject_tools(messages: list[Message], tools: list[dict[str, Any]]) -> list[Message]:
        """Inject tool definitions into the system prompt.

        Since the v2 model doesn't support native function calling, we
        format tool schemas as text in the system prompt. The model
        learns to emit tool calls in the XML-style format.
        """
        tool_text_parts: list[str] = []
        for t in tools:
            # Support both OpenAI format {"type":"function","function":{...}}
            # and flat format {"name":...,"description":...}
            fn = t.get("function", t)
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            param_lines: list[str] = []
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else ""
                param_lines.append(f"    - {pname} ({ptype}{req}): {pdesc}")

            tool_text_parts.append(
                f"### {name}\n{desc}\nParameters:\n" + "\n".join(param_lines)
            )

        tool_block = (
            "## Available Tools\n\n"
            + "\n\n".join(tool_text_parts)
            + "\n\n## Tool Call Format\n"
            "To call a tool, output:\n\n"
            "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\n"
            "You can call multiple tools by emitting multiple <tool_call> blocks.\n"
            "After each tool call, you will receive the result. Use it to decide your next action.\n"
            "When the task is complete, provide your final answer without any tool calls."
        )

        # Inject into the first system message, or prepend one
        result = list(messages)
        for i, m in enumerate(result):
            if m.role == "system":
                result[i] = Message(role="system", content=m.content + "\n\n" + tool_block)
                return result
        # No system message — prepend one
        result.insert(0, Message(role="system", content=tool_block))
        return result

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Supports multiple formats for robustness across model versions:
        1. <tool_call> JSON blocks (primary format for v2+)
        2. Qwen3.5 XML-style: <function=name> <parameter=key>value</parameter> </function>
        3. Pipe-delimited: |tool_call| {...} |end|
        4. JSON code blocks: ```tool_call or ```json
        5. Bare JSON: {"name": "...", "arguments": {...}}
        """
        tool_calls: list[ToolCall] = []

        # Pattern 1: <tool_call> JSON blocks (primary)
        tc_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        tc_matches = re.findall(tc_pattern, text, re.DOTALL)
        for i, match in enumerate(tc_matches):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue

        if tool_calls:
            return tool_calls

        # Pattern 2: Qwen3.5 XML-style format
        # <function=name> <parameter=key>value</parameter> </function>
        fn_pattern = r"<function=(\w+)>(.*?)</function>"
        fn_matches = re.findall(fn_pattern, text, re.DOTALL)
        for i, (func_name, body) in enumerate(fn_matches):
            args: dict[str, Any] = {}
            param_pattern = r"<parameter=(\w+)>\s*(.*?)\s*</parameter>"
            param_matches = re.findall(param_pattern, body, re.DOTALL)
            for param_name, param_value in param_matches:
                try:
                    args[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    args[param_name] = param_value.strip()
            tool_calls.append(ToolCall(id=f"call_{i}", name=func_name, arguments=args))

        if tool_calls:
            return tool_calls

        # Pattern 3: Pipe-delimited (older Qwen)
        pipe_pattern = r"\|tool_call\|\s*(\{.*?\})\s*\|end\|"
        pipe_matches = re.findall(pipe_pattern, text, re.DOTALL)
        for i, match in enumerate(pipe_matches):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue

        if tool_calls:
            return tool_calls

        # Pattern 4: JSON in code blocks
        json_block_pattern = r"```(?:tool_call|tool_code|json)\s*(\{.*?\})\s*```"
        json_matches = re.findall(json_block_pattern, text, re.DOTALL)
        for i, match in enumerate(json_matches):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue

        if tool_calls:
            return tool_calls

        # Pattern 5: Bare JSON with name + arguments
        bare_pattern = r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\})'
        bare_matches = re.findall(bare_pattern, text, re.DOTALL)
        for i, match in enumerate(bare_matches):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue

        return tool_calls

    @staticmethod
    def _parse_control_token(text: str) -> tuple[str, str | None]:
        """Parse control tokens from v2 model output.

        The v2 model emits <FAST> or <THINK> at the start of generation.
        This method detects the token, strips it, and returns the mode.

        Returns (cleaned_text, mode) where mode is "fast", "think", or None.
        """
        stripped = text.lstrip()
        if stripped.startswith("<FAST>"):
            return stripped[6:].lstrip(), "fast"
        elif stripped.startswith("<THINK>"):
            return stripped[7:].lstrip(), "think"
        return text, None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3.5 thinking blocks from text.

        Handles two formats:
        1. <think>...</think> tags (standard Qwen3.5 thinking mode)
        2. "Thinking Process:" prefix (fine-tuned model behavior)
        """
        import re

        # Strip <think>...</think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Strip unclosed <think> block (model didn't close it)
        if "<think>" in cleaned and "</think>" not in cleaned:
            idx = cleaned.find("<think>")
            # Find the next assistant turn or end
            rest = cleaned[idx:]
            end_idx = rest.find("<|im_start|>")
            if end_idx != -1:
                cleaned = cleaned[:idx] + rest[end_idx:]
            else:
                # Try to find where thinking ends (look for a newline after a period)
                # If the model emits "Thinking Process:" and then the answer,
                # we keep everything — the model was trained to always think
                pass

        # Strip "Thinking Process:" preamble if present
        # The fine-tuned model emits this before the actual answer.
        # The thinking section is indented/numbered; the answer is the
        # last non-indented paragraph after the thinking block.
        if "Thinking Process:" in cleaned:
            # First try explicit delimiters
            patterns = [
                r"(?:Final )?Answer:\s*(.+)",
                r"Output:\s*(.+)",
                r"Result:\s*(.+)",
                r"Solution:\s*(.+)",
            ]
            stripped = None
            for pattern in patterns:
                match = re.search(pattern, cleaned, re.DOTALL)
                if match:
                    stripped = match.group(1).strip()
                    break
            if stripped is None:
                # No explicit delimiter — the answer is the last
                # paragraph after the thinking section.
                # The thinking can be indented/numbered OR a flowing
                # paragraph. The answer is always the last non-empty
                # paragraph (after the last blank line).
                lines = cleaned.split("\n")
                # Find the last blank line — answer is after it
                last_blank = -1
                for i in range(len(lines) - 1, -1, -1):
                    if not lines[i].strip():
                        last_blank = i
                        break
                if last_blank >= 0 and last_blank < len(lines) - 1:
                    stripped = "\n".join(lines[last_blank + 1:]).strip()
                    # If answer is too short, try last 2 paragraphs
                    if len(stripped) < 10 and last_blank > 0:
                        for j in range(last_blank - 1, -1, -1):
                            if not lines[j].strip():
                                stripped = "\n".join(lines[j + 1:]).strip()
                                break
                else:
                    stripped = cleaned
            cleaned = stripped

        return cleaned.strip()

    def health(self) -> bool:
        try:
            self._load()
            return self._model is not None
        except Exception as e:
            log.warning("mlxvlm.health.failed", model=self.name, error=str(e))
            return False
