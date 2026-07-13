"""Test fixtures and shared fakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.base import EmbeddingModel, Model
from app.models.types import EmbeddingResponse, Message, ModelResponse
from app.retrieval.base import Result, Retriever


class FakeModel(Model):
    name = "fake"
    provider = "fake"

    def __init__(self, *, echo_prefix: str = "FAKE:"):
        self._prefix = echo_prefix

    def generate(self, prompt, *, system=None, **kwargs):
        return ModelResponse(text=f"{self._prefix} {prompt}", model=self.name, provider=self.provider)

    def chat(self, messages: list[Message], **kwargs):
        last = messages[-1].content
        return ModelResponse(text=f"{self._prefix} {last}", model=self.name, provider=self.provider)


class FakeEmbeddingModel(EmbeddingModel):
    name = "fake-embed"
    provider = "fake"

    def embed(self, text: str) -> EmbeddingResponse:
        # Deterministic pseudo-embedding from character codes.
        vec = [float(ord(c) % 16) for c in (text + " " * 16)[:16]]
        return EmbeddingResponse(vector=vec, model=self.name, provider=self.provider)


class FakeRetriever(Retriever):
    backend = "fake"

    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    def index(self, source: str, chunks: list[dict]) -> int:
        self._store[source] = chunks
        return len(chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[Result]:
        all_chunks = [c for chunks in self._store.values() for c in chunks]
        return [
            Result(text=c["text"], score=1.0 - i * 0.1, source=c["metadata"].get("source", "?"), metadata=c["metadata"])
            for i, c in enumerate(all_chunks[:top_k])
        ]

    def delete(self, source: str) -> int:
        self._store.pop(source, None)
        return 1


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def fake_embedding():
    return FakeEmbeddingModel()


@pytest.fixture
def fake_retriever():
    return FakeRetriever()


@pytest.fixture
def tmp_text_file(tmp_path: Path):
    f = tmp_path / "sample.md"
    f.write_text("# Title\n\nHello world. This is a test of the indexer.\n" * 3)
    return f
