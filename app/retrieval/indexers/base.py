"""Indexer abstraction.

An Indexer turns a source (path, url, repo) into a list of chunk dicts
ready for a Retriever. Indexers do not talk to the vector store — that
keeps them trivially testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Indexer(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, source: str) -> list[dict]:
        """Return a list of {'text': str, 'metadata': dict} chunks."""

    def supports(self, source: str) -> bool:
        return False
