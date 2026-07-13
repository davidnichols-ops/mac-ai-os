"""Retrieval registry — maps a backend name to a Retriever factory."""

from __future__ import annotations

from app.config.settings import Settings
from app.models.base import EmbeddingModel
from app.retrieval.base import Retriever
from app.retrieval.chroma_backend import ChromaRetriever


def build_retriever(settings: Settings, embedding_model: EmbeddingModel) -> Retriever:
    backend = settings.retrieval.backend
    if backend == "chroma":
        return ChromaRetriever(
            embedding_model,
            collection=settings.embeddings.collection,
            persist_dir=settings.embeddings.persist_dir,
            top_k=settings.retrieval.top_k,
        )
    raise ValueError(f"Unknown retrieval backend: {backend}")


def register_backend(name: str, factory) -> None:
    # Future: lancedb, qdrant, pgvector
    pass
