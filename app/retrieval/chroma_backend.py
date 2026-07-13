"""Chroma vector-store backend.

Implements the Retriever ABC over a persistent Chroma collection. Embeddings
are produced by the injected EmbeddingModel so the backend stays
provider-agnostic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.log import get_logger
from app.models.base import EmbeddingModel
from app.retrieval.base import Result, Retriever

log = get_logger(__name__)


def _stable_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{chunk_index}:{text[:64]}".encode()).hexdigest()
    return f"chunk-{h}"


class ChromaRetriever(Retriever):
    backend = "chroma"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        collection: str = "mac-ai-os",
        persist_dir: str = "data/embeddings",
        top_k: int = 5,
    ):
        import chromadb  # type: ignore

        self.embedding_model = embedding_model
        self.collection_name = collection
        self.top_k = top_k
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        log.info("chroma.init", collection=collection, persist_dir=persist_dir)

    def index(self, source: str, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        embeddings = [[float(x) for x in self.embedding_model.embed(c["text"]).vector] for c in chunks]
        ids = [_stable_id(source, c["metadata"].get("chunk_index", i), c["text"])
               for i, c in enumerate(chunks)]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            md = {k: v for k, v in c["metadata"].items() if isinstance(v, (str, int, float, bool))}
            metadatas.append(md)
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        log.info("chroma.indexed", source=source, count=len(chunks))
        return len(chunks)

    def search(self, query: str, *, top_k: int | None = None) -> list[Result]:
        k = top_k or self.top_k
        qvec = [float(x) for x in self.embedding_model.embed(query).vector]
        res = self._collection.query(query_embeddings=[qvec], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        results: list[Result] = []
        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            results.append(
                Result(
                    text=doc,
                    score=score,
                    source=meta.get("source", "unknown") if meta else "unknown",
                    metadata=meta or {},
                )
            )
        return results

    def delete(self, source: str) -> int:
        try:
            self._collection.delete(where={"source": source})
        except Exception as e:
            log.warning("chroma.delete.failed", source=source, error=str(e))
            return 0
        log.info("chroma.deleted", source=source)
        return 1

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def health(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:
            return False
