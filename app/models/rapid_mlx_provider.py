"""Rapid-MLX provider for the local claude-yolo-vibes-v4 7B model.

Connects to a Rapid-MLX OpenAI-compatible endpoint (default
http://127.0.0.1:8083/v1), auto-starts the server if configured, and
translates the model's fine-tuned <tool_use> XML into MAOS ToolCall
objects.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from typing import Any

from app.log import get_logger
from app.models.api_provider import APIModel
from app.models.types import Message, ModelResponse, ToolCall

log = get_logger(__name__)

# MAOS tool name -> the name the yolo v4 model was fine-tuned on.
_MAOS_TO_YOLO_NAME: dict[str, str] = {
    "shell": "Bash",
    "file_read": "Read",
    "file_write": "Write",
    "file_edit": "Edit",
    "find_file": "Glob",
    "grep": "Grep",
    "todo_write": "TodoWrite",
}

_YOLO_TO_MAOS_NAME: dict[str, str] = {v: k for k, v in _MAOS_TO_YOLO_NAME.items()}

# Parameter names that differ between the yolo examples and MAOS schemas.
_MAOS_TO_YOLO_PARAM: dict[str, str] = {
    "path": "file_path",
}

_YOLO_TO_MAOS_PARAM: dict[str, str] = {v: k for k, v in _MAOS_TO_YOLO_PARAM.items()}

# Example values to show for required parameters in the injected prompt.
_PARAM_EXAMPLES: dict[str, str] = {
    "command": "ls -la",
    "file_path": "/path/to/file",
    "content": "file contents here",
    "old_string": "text to replace",
    "new_string": "replacement text",
    "pattern": "**/*.py",
    "todos": '[{"content": "task", "status": "pending"}]',
}

_TOOL_USE_RE = re.compile(
    r'<tool_use\s+name="([^"]+)">\s*(.*?)\s*</tool_use>', re.DOTALL
)
_PARAM_RE = re.compile(r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>', re.DOTALL)
_QWEN_TOKEN_RE = re.compile(r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>')


class RapidMLXModel(APIModel):
    """Local Rapid-MLX model with yolo v4 tool-use XML parsing."""

    provider = "rapid_mlx"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8083/v1",
        api_key: str = "device-local",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        model_path: str | None = None,
        binary: str | None = None,
        auto_start: bool = True,
    ) -> None:
        super().__init__(
            model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model_path = model_path or os.environ.get(
            "RAPID_MLX_MODEL",
            os.path.expanduser("~/AI-Models/mlx/claude-yolo-vibes-v4"),
        )
        self.bin = binary or os.environ.get(
            "RAPID_MLX_BIN",
            shutil.which("rapid-mlx") or "",
        )
        self.auto_start = auto_start
        self._server_started = False

    def _ensure_server(self) -> None:
        """Start rapid-mlx if it is not already serving and auto_start is on."""
        if self._server_started or not self.auto_start:
            return
        if self.health():
            self._server_started = True
            return

        if not self.bin or not shutil.which(self.bin):
            raise RuntimeError(
                f"rapid-mlx not found (set RAPID_MLX_BIN or config bin); "
                f"tried {self.bin!r}"
            )
        if not os.path.isdir(self.model_path):
            raise RuntimeError(f"Rapid-MLX model path not found: {self.model_path}")

        port = _port_from_url(self.base_url)
        log.info("rapid_mlx.starting_server", bin=self.bin, port=port, model=self.model_path)

        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
        cmd = [
            self.bin,
            "serve",
            self.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            self.name,
            "--enable-prefix-cache",
            "--prefix-cache-size",
            "512",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            "8",
            "--pin-system-prompt",
            "--speculative-config",
            '{"method":"suffix","num_speculative_tokens":8}',
            "--max-tokens",
            "8192",
            "--default-temperature",
            "0",
            "--default-top-p",
            "1",
            "--gpu-memory-utilization",
            "0.85",
        ]
        with open("/tmp/maos-rapid-mlx.log", "w") as log_file:
            subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.health():
                self._server_started = True
                log.info("rapid_mlx.server_ready", port=port)
                return
            time.sleep(1)
        raise RuntimeError(
            f"Rapid-MLX did not become ready on port {port} within 120s. "
            "Check /tmp/maos-rapid-mlx.log"
        )

    def chat(self, messages: list[Message], *, tools=None, **kwargs) -> ModelResponse:
        self._ensure_server()

        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        if tools:
            messages = self._inject_yolo_tools(messages, tools)

        client = self._get_client()
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=self.name,
            messages=[m.to_dict() for m in messages],
            temperature=temp,
            max_tokens=max_tokens,
        )
        t_gen = time.monotonic()

        text = resp.choices[0].message.content or ""
        text = _QWEN_TOKEN_RE.sub("", text)

        tool_calls: list[ToolCall] = []
        if tools:
            tool_calls, text = self._extract_yolo_tool_calls(text)

        finish_reason = "tool_calls" if tool_calls else "stop"
        gen_time_s = t_gen - t0
        completion_tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        tps = completion_tokens / gen_time_s if gen_time_s > 0 else 0.0

        return ModelResponse(
            text=text.strip(),
            model=self.name,
            provider=self.provider,
            raw=resp,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            generation_time_s=gen_time_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tps,
        )

    @staticmethod
    def _inject_yolo_tools(messages: list[Message], tools: list[dict[str, Any]]) -> list[Message]:
        """Replace the MAOS <tool_call> tool sections with yolo <tool_use> XML."""
        yolo_tool_block = _build_yolo_tool_block(tools)
        if not yolo_tool_block:
            return list(messages)

        result = list(messages)
        for i, m in enumerate(result):
            if m.role == "system":
                result[i] = Message(role="system", content=_rebuild_for_yolo(m.content, yolo_tool_block))
                return result
        result.insert(0, Message(role="system", content=yolo_tool_block))
        return result

    @staticmethod
    def _extract_yolo_tool_calls(text: str) -> tuple[list[ToolCall], str]:
        """Parse <tool_use> XML blocks and return (tool_calls, text_without_xml)."""
        tool_calls: list[ToolCall] = []
        remaining_text = _TOOL_USE_RE.sub(lambda m: "", text).strip()

        for i, match in enumerate(_TOOL_USE_RE.finditer(text)):
            yolo_name = match.group(1)
            maos_name = _YOLO_TO_MAOS_NAME.get(yolo_name, yolo_name)
            params_xml = match.group(2)

            args: dict[str, Any] = {}
            for pm in _PARAM_RE.finditer(params_xml):
                yolo_pname = pm.group(1)
                pval = pm.group(2).strip()
                maos_pname = _YOLO_TO_MAOS_PARAM.get(yolo_pname, yolo_pname)
                try:
                    args[maos_pname] = json.loads(pval)
                except json.JSONDecodeError:
                    args[maos_pname] = pval

            tool_calls.append(
                ToolCall(id=f"call_{i}", name=maos_name, arguments=args)
            )

        return tool_calls, remaining_text


def _build_yolo_tool_block(tools: list[dict[str, Any]]) -> str:
    """Build the yolo-style Available Tools + Tool Call Format block."""
    yolo_tools: list[str] = []
    for t in tools:
        fn = t.get("function", t)
        maos_name = fn.get("name", "unknown")
        yolo_name = _MAOS_TO_YOLO_NAME.get(maos_name)
        if not yolo_name:
            continue
        # Support both OpenAI function format and flat tool schemas.
        params = fn.get("parameters", fn)
        props = params.get("properties", {})
        required = params.get("required", []) or []

        # Example <tool_use> block.
        example_lines = [f'<tool_use name="{yolo_name}">']
        for pname in required:
            yolo_pname = _MAOS_TO_YOLO_PARAM.get(pname, pname)
            example_value = _PARAM_EXAMPLES.get(pname, _PARAM_EXAMPLES.get(yolo_pname, ""))
            example_lines.append(f'  <parameter name="{yolo_pname}">{example_value}</parameter>')
        for pname in props:
            if pname in required:
                continue
            yolo_pname = _MAOS_TO_YOLO_PARAM.get(pname, pname)
            if yolo_pname in _PARAM_EXAMPLES or pname in _PARAM_EXAMPLES:
                example_value = _PARAM_EXAMPLES.get(yolo_pname, _PARAM_EXAMPLES.get(pname, ""))
                example_lines.append(f'  <parameter name="{yolo_pname}">{example_value}</parameter>')
        example_lines.append("</tool_use>")

        # Human-readable parameter list.
        param_lines: list[str] = []
        for pname, pschema in props.items():
            yolo_pname = _MAOS_TO_YOLO_PARAM.get(pname, pname)
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req = " (required)" if pname in required else ""
            param_lines.append(f"  - {yolo_pname} ({ptype}{req}): {pdesc}")

        yolo_tools.append(
            f"### {yolo_name}\n{fn.get('description', '')}\n\n"
            + ("Parameters:\n" + "\n".join(param_lines) if param_lines else "Parameters: none")
            + "\n\nExample:\n"
            + "\n".join(example_lines)
        )

    if not yolo_tools:
        return ""

    return (
        "## Available Tools\n\n"
        + "\n\n".join(yolo_tools)
        + "\n\n## Tool Call Format (CRITICAL)\n\n"
        "You accomplish tasks by using tools. "
        "When the user asks you to create, write, edit, run, or find something, USE A TOOL.\n"
        "Do NOT just show the code in a markdown block. Execute it.\n"
        "To call a tool, output exactly ONE XML <tool_use> block using the examples above.\n"
        "Do NOT output markdown code blocks, JSON, <tool_call> tags, or prose. "
        "ONLY use <tool_use> XML.\n"
        "After each tool call, you will receive the result. Use it to decide your next action.\n"
        "When the task is complete, provide your final answer without any tool calls.\n"
        "ALL FILE PATHS MUST BE ABSOLUTE."
    )


def _rebuild_for_yolo(system_text: str, yolo_tool_block: str) -> str:
    """Keep MAOS identity/context, replace the tool sections with yolo XML."""
    parts = re.split(r"\n## ", system_text)
    prefix = parts[0]  # identity + intro (ends before first '## ')

    remaining: list[tuple[str, str]] = []
    core_principles = ""
    for part in parts[1:]:
        header_end = part.find("\n")
        header = part[:header_end] if header_end != -1 else part
        body = part[header_end + 1 :].strip() if header_end != -1 else ""
        if header in ("Tool Call Format", "Available Tools"):
            continue
        if header == "Core Principles":
            core_principles = body
            continue
        remaining.append((header, body))

    rebuilt: list[str] = [prefix]
    if core_principles:
        rebuilt.append(f"## Core Principles\n{core_principles}")
    rebuilt.append(yolo_tool_block)
    for header, body in remaining:
        rebuilt.append(f"## {header}\n{body}")
    return "\n\n".join(rebuilt)


def _port_from_url(base_url: str) -> int:
    """Extract the port from a local http base URL, defaulting to 8083."""
    m = re.search(r"127\.0\.0\.1:(\d+)", base_url)
    if m:
        return int(m.group(1))
    m = re.search(r"localhost:(\d+)", base_url)
    if m:
        return int(m.group(1))
    return 8083
