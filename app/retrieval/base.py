"""Retrieval abstraction.

    Retriever.index(source)   -> int   # number of chunks added
    Retriever.search(query)   -> list[Result]
    Retriever.delete(source)  -> int
    Retriever.update(source)  -> int

Backends (chroma today; lancedb/qdrant/pgvector later) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Result:
    text: str
    score: float
    source: str
    metadata: dict = field(default_factory=dict)


class Retriever(ABC):
    backend: str = "base"

    @abstractmethod
    def index(self, source: str, chunks: list[dict]) -> int:
        """Add chunks (each has 'text' + 'metadata') for `source`.

        Returns the number of chunks indexed.
        """

    @abstractmethod
    def search(self, query: str, *, top_k: int = 5) -> list[Result]:
        ...

    @abstractmethod
    def delete(self, source: str) -> int:
        ...

    def update(self, source: str, chunks: list[dict]) -> int:
        """Default: delete then re-index."""
        self.delete(source)
        return self.index(source, chunks)

    def health(self) -> bool:
        return True
