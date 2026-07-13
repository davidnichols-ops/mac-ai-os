"""Chroma's built-in ONNX embedding function.

Uses `chromadb.utils.embedding_functions.DefaultEmbeddingFunction`, which runs
all-MiniLM-L6-v2 via onnxruntime — no PyTorch, no Ollama, ~80 MB model cached
automatically on first use. This is the most disk-safe local default.
"""

from __future__ import annotations

from app.log import get_logger
from app.models.base import EmbeddingModel
from app.models.types import EmbeddingResponse

log = get_logger(__name__)


class ChromaDefaultEmbeddingModel(EmbeddingModel):
    provider = "chroma_default"
    name = "all-MiniLM-L6-v2-onnx"

    def __init__(self):
        from chromadb.utils import embedding_functions  # type: ignore

        self._fn = embedding_functions.DefaultEmbeddingFunction()
        log.info("chroma_default.embedding.init")

    def embed(self, text: str) -> EmbeddingResponse:
        # chromadb returns a list[list[float]] for a list input; wrap accordingly.
        vec = self._fn([text])[0]
        return EmbeddingResponse(vector=[float(x) for x in vec], model=self.name, provider=self.provider)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        vecs = self._fn(texts)
        return [
            EmbeddingResponse(vector=[float(x) for x in v], model=self.name, provider=self.provider)
            for v in vecs
        ]

    def health(self) -> bool:
        try:
            self._fn(["health check"])
            return True
        except Exception as e:
            log.warning("chroma_default.embedding.health.failed", error=str(e))
            return False
