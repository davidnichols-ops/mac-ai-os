"""Ollama provider — works for both chat and embedding models.

Endpoint is read from config (Settings.ollama.base_url); never hardcoded.
"""

from __future__ import annotations

from app.log import get_logger
from app.models.base import EmbeddingModel, Model
from app.models.types import EmbeddingResponse, Message, ModelResponse

log = get_logger(__name__)


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

    def chat(self, messages: list[Message], **kwargs) -> ModelResponse:
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        resp = self._client.chat(
            model=self.name,
            messages=[m.to_dict() for m in messages],
            options={"temperature": temp, "num_predict": max_tokens},
        )
        text = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else resp.message.content
        return ModelResponse(text=text, model=self.name, provider=self.provider, raw=resp)

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
