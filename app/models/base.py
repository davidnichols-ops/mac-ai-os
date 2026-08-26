"""Model abstraction.

One interface for every provider (MLX, Ollama, remote OpenAI-compatible):

    Model.generate(prompt) -> ModelResponse
    Model.chat(messages)   -> ModelResponse
    Model.embed(text)      -> EmbeddingResponse

Providers are registered in `registry.py` and selected by `provider` in config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.models.types import EmbeddingResponse, Message, ModelResponse


class Model(ABC):
    """A unified interface over chat / generation / embedding backends."""

    name: str = "base"
    provider: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, **kwargs) -> ModelResponse:
        """Single-turn generation."""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Multi-turn chat with optional tool definitions.

        When tools are provided, the model may emit tool calls in its
        response. The provider parses these into ModelResponse.tool_calls.
        """

    def embed(self, text: str) -> EmbeddingResponse:
        raise NotImplementedError(f"{self.provider}/{self.name} does not implement embed()")

    def health(self) -> bool:
        """Return True if the backend is reachable and the model is loadable."""
        return True

    def describe(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.name}


class EmbeddingModel(ABC):
    """Separate interface for embedding-only backends."""

    name: str = "base-embed"
    provider: str = "base"

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResponse:
        ...

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        return [self.embed(t) for t in texts]

    def health(self) -> bool:
        return True
