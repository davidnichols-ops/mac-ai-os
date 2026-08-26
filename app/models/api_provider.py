"""OpenAI-compatible API provider for remote inference (Together AI, etc.)."""

from __future__ import annotations

import json
import re
import time

from app.log import get_logger
from app.models.base import Model
from app.models.types import Message, ModelResponse, ToolCall

log = get_logger(__name__)

TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/tool_call" + chr(62)
FN_OPEN = chr(60) + "function="
FN_CLOSE = chr(60) + "/function" + chr(62)
PARAM_OPEN = chr(60) + "parameter="
PARAM_CLOSE = chr(60) + "/parameter" + chr(62)


class APIModel(Model):
    """Remote model via OpenAI-compatible chat completions API."""
    provider = "api"

    def __init__(self, model, *, base_url="https://api-inference.together.ai/v1", api_key="", temperature=0.7, max_tokens=2048):
        self.name = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import os

            from openai import OpenAI
            headers = {}
            if "openrouter" in self.base_url:
                headers["HTTP-Referer"] = "https://mac-ai-os.devin.local"
                headers["X-Title"] = "MAOS"
            # Resolve api_key: explicit > OPENROUTER_API_KEY > OPENAI_API_KEY.
            # This catches the case where ${VAR} wasn't expanded or the env
            # var wasn't available when the model was constructed.
            key = self.api_key
            if not key or key.startswith("${"):
                key = (
                    os.environ.get("OPENROUTER_API_KEY", "")
                    or os.environ.get("OPENAI_API_KEY", "")
                    or key
                )
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=key,
                default_headers=headers or None,
            )
        return self._client

    def generate(self, prompt, *, system=None, **kwargs):
        messages = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, **kwargs)

    def chat(self, messages, *, tools=None, **kwargs):
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        thinking = kwargs.get("thinking", kwargs.get("enable_thinking", None))
        if tools:
            messages = self._inject_tools(messages, tools)
        client = self._get_client()
        t0 = time.monotonic()
        resp = client.chat.completions.create(model=self.name, messages=[m.to_dict() for m in messages], temperature=temp, max_tokens=max_tokens)
        t_gen = time.monotonic()
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>\s*</think>\s*", "", text, flags=re.DOTALL)
        text, mode = self._parse_control_token(text)
        if mode is None:
            mode = "think" if thinking else "fast" if thinking is False else None
        if mode == "fast":
            text = self._strip_thinking(text)
        tool_calls = self._parse_tool_calls(text) if tools else []
        finish_reason = "tool_calls" if tool_calls else "stop"
        gen_time_s = t_gen - t0
        completion_tokens = resp.usage.completion_tokens if resp.usage else len(text.split())
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        tps = completion_tokens / gen_time_s if gen_time_s > 0 else 0.0
        return ModelResponse(text=text, model=self.name, provider=self.provider, raw=resp, mode=mode, tool_calls=tool_calls, finish_reason=finish_reason, generation_time_s=gen_time_s, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, tokens_per_second=tps)

    def health(self):
        try:
            self._get_client().models.list()
            return True
        except Exception:
            return False

    @staticmethod
    def _inject_tools(messages, tools):
        parts = []
        for t in tools:
            fn = t.get("function", t)
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            plines = []
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req = " (required)" if pname in required else ""
                plines.append(f"    - {pname} ({ptype}{req}): {pdesc}")
            parts.append(f"### {name}\n{desc}\nParameters:\n" + "\n".join(plines))
        tc_ex = json.dumps(dict(name="tool_name", arguments=dict(param="value")), indent=2)
        tool_block = "## Available Tools\n\n" + "\n\n".join(parts) + "\n\n## Tool Call Format\nTo call a tool, output:\n\n" + TC_OPEN + "\n" + tc_ex + "\n" + TC_CLOSE + "\n\nAfter each tool call, you will receive the result. Use it to decide your next action.\nWhen the task is complete, provide your final answer without any tool calls."
        result = list(messages)
        for i, m in enumerate(result):
            if m.role == "system":
                result[i] = Message(role="system", content=m.content + "\n\n" + tool_block)
                return result
        result.insert(0, Message(role="system", content=tool_block))
        return result

    @staticmethod
    def _parse_control_token(text):
        stripped = text.lstrip()
        m = re.match(r"^\[(FAST|THINK|VERIFY|RECOVER|ESCALATE)\]\s*", stripped, re.IGNORECASE)
        if m:
            return stripped[m.end():], m.group(1).lower()
        if stripped.startswith("<FAST>"):
            return stripped[6:].lstrip(), "fast"
        if stripped.startswith("<THINK>"):
            return stripped[7:].lstrip(), "think"
        return text, None

    @staticmethod
    def _strip_thinking(text):
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in cleaned and "</think>" not in cleaned:
            idx = cleaned.find("<think>")
            rest = cleaned[idx:]
            end_idx = rest.find("<|im_start|>")
            if end_idx != -1:
                cleaned = cleaned[:idx] + rest[end_idx:]
        return cleaned.strip()

    @staticmethod
    def _parse_tool_calls(text):
        tool_calls = []
        # Pattern 1: tool_call JSON blocks
        pat1 = TC_OPEN + r"\s*(\{.*?\})\s*" + TC_CLOSE
        for i, match in enumerate(re.findall(pat1, text, re.DOTALL)):
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
        # Pattern 2: Qwen3.5 XML-style
        pat2 = FN_OPEN + r"(\w+)" + chr(62) + r"(.*?)" + FN_CLOSE
        for i, (fn_name, body) in enumerate(re.findall(pat2, text, re.DOTALL)):
            args = {}
            pat2p = PARAM_OPEN + r"(\w+)" + chr(62) + r"\s*(.*?)\s*" + PARAM_CLOSE
            for pn, pv in re.findall(pat2p, body, re.DOTALL):
                try:
                    args[pn] = json.loads(pv)
                except (json.JSONDecodeError, ValueError):
                    args[pn] = pv.strip()
            tool_calls.append(ToolCall(id=f"call_{i}", name=fn_name, arguments=args))
        if tool_calls:
            return tool_calls
        # Pattern 3: pipe-delimited
        for i, match in enumerate(re.findall(r"\|tool_call\|\s*(\{.*?\})\s*\|end\|", text, re.DOTALL)):
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
        for i, match in enumerate(re.findall(r"```(?:tool_call|tool_code|json)\s*(\{.*?\})\s*```", text, re.DOTALL)):
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
        # Pattern 5: bare JSON
        pat5 = r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\})'
        for i, match in enumerate(re.findall(pat5, text, re.DOTALL)):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue
        return tool_calls
