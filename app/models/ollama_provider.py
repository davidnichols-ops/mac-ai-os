"""Ollama provider — works for both chat and embedding models.

Endpoint is read from config (Settings.ollama.base_url); never hardcoded.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.log import get_logger
from app.models.base import EmbeddingModel, Model
from app.models.types import EmbeddingResponse, Message, ModelResponse, ToolCall

log = get_logger(__name__)

TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/tool_call" + chr(62)


def _import_ollama():
    try:
        import ollama  # type: ignore
        return ollama
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("ollama python client not installed") from e


class OllamaModel(Model):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.name = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._ollama = _import_ollama()
        self._client = self._ollama.Client(host=base_url)
        log.info("ollama.model.init", model=model, base_url=base_url)

    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
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
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        # Use raw HTTP API instead of the Python client to avoid
        # Ollama's built-in tool call parser, which crashes on qwen3.5
        # when the model emits text that looks like tool calls.
        # MAOS handles tool call parsing itself via the system prompt.
        import requests
        payload = {
            "model": self.name,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temp, "num_predict": max_tokens},
            # Pass empty tools list to disable Ollama's built-in tool call
            # parser, which crashes on qwen3.5 with EOF errors. MAOS handles
            # tool call parsing itself via the system prompt format.
            "tools": [],
        }
        try:
            http_resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            )
            http_resp.raise_for_status()
            data = http_resp.json()
        except requests.exceptions.HTTPError as e:
            log.error("ollama.chat.http_error", model=self.name, status=e.response.status_code, body=e.response.text[:500])
            raise RuntimeError(f"Ollama chat failed: {e.response.status_code}") from e
        except Exception as e:
            log.error("ollama.chat.error", model=self.name, error=str(e))
            raise
        text = data.get("message", {}).get("content", "")
        tool_calls = self._parse_tool_calls(text) if tools else []
        finish = "tool_calls" if tool_calls else data.get("done_reason", "stop")
        return ModelResponse(
            text=text, model=self.name, provider=self.provider, raw=data,
            tool_calls=tool_calls, finish_reason=finish,
        )

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        """Parse <tool_call> JSON blocks from model output.

        Handles missing closing tags (common with smaller models).
        """
        tool_calls = []
        pat = TC_OPEN + r"\s*(\{.*?\})\s*(?:" + TC_CLOSE + r"|\n\n|$)"
        for i, match in enumerate(re.findall(pat, text, re.DOTALL)):
            try:
                data = json.loads(match)
                name = data.get("name", "")
                args = data.get("arguments", data.get("parameters", {}))
                if name:
                    tool_calls.append(ToolCall(id=f"call_{i}", name=name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue
        return tool_calls

    def health(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception as e:
            log.warning("ollama.health.failed", model=self.name, error=str(e))
            return False


class OllamaEmbeddingModel(EmbeddingModel):
    provider = "ollama"

    def __init__(self, model: str, *, base_url: str = "http://localhost:11434"):
        self.name = model
        self.base_url = base_url
        self._ollama = _import_ollama()
        self._client = self._ollama.Client(host=base_url)
        log.info("ollama.embedding.init", model=model, base_url=base_url)

    def embed(self, text: str) -> EmbeddingResponse:
        resp = self._client.embeddings(model=self.name, prompt=text)
        vec = resp.get("embedding") if isinstance(resp, dict) else resp.embedding
        return EmbeddingResponse(vector=list(vec), model=self.name, provider=self.provider)

    def health(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception as e:
            log.warning("ollama.embedding.health.failed", model=self.name, error=str(e))
            return False
