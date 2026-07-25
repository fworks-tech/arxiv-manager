"""Hybrid retriever combining keyword (SQL) and semantic (ChromaDB) search.

Uses lazy imports for heavy dependencies (sentence-transformers, torch)
to avoid import-time crashes when the module is loaded but not used.
"""

from __future__ import annotations

import logging
from typing import Any

from .retriever_config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_FIGURES,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_chroma_client = None
_vector_store = None
_embeddings = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
        )
    return _embeddings


def _get_vector_store(collection_name: str) -> Any:
    global _vector_store
    if _vector_store is None:
        from langchain_chroma import Chroma
        _vector_store = Chroma(
            client=_get_chroma_client(),
            collection_name=collection_name,
            embedding_function=_get_embeddings(),
        )
    return _vector_store


class HybridRetriever:
    """Retrieve figure context using keyword + semantic search.

    All heavy imports (chromadb, sentence-transformers, torch, langchain)
    are loaded lazily on first use. This module can be imported without
    triggering those dependencies.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_FIGURES,
    ) -> None:
        self._collection_name = collection_name

    def _ensure_store(self):
        from langchain_core.documents import Document
        self._Document = Document
        self._store = _get_vector_store(self._collection_name)

    def add_figure(
        self,
        figure_id: int,
        caption: str,
        figure_type: str,
        paper_title: str = "",
        difficulty: str = "",
        question: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Index a figure's caption and context into the vector store."""
        self._ensure_store()
        text_parts = [caption]
        if paper_title:
            text_parts.append(f"Paper: {paper_title}")
        if figure_type:
            text_parts.append(f"Type: {figure_type}")
        if question:
            text_parts.append(f"Question: {question}")
        text = " | ".join(text_parts)

        meta = {
            "figure_id": str(figure_id),
            "figure_type": figure_type,
            "difficulty": difficulty,
            **(metadata or {}),
        }

        doc = self._Document(page_content=text, metadata=meta)
        doc_id = self._store.add_documents([doc])
        return doc_id[0] if doc_id else ""

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Batch add texts to the vector store."""
        self._ensure_store()
        docs = [
            self._Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        return self._store.add_documents(docs)

    def search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        self._ensure_store()
        results = self._store.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
        )
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(float(score), 4),
            }
            for doc, score in results
        ]

    def search_by_metadata(
        self,
        field: str,
        value: str,
    ) -> list[dict[str, Any]]:
        """Search for documents by exact metadata field match."""
        self._ensure_store()
        results = self._store.get(where={field: value})
        docs = []
        for i in range(len(results["ids"])):
            docs.append({
                "id": results["ids"][i],
                "content": results["documents"][i] if results.get("documents") else "",
                "metadata": results["metadatas"][i] if results.get("metadatas") else {},
            })
        return docs

    def delete_figure(self, figure_id: int) -> None:
        """Remove a figure's embeddings from the index."""
        self._ensure_store()
        self._store.delete(where={"figure_id": str(figure_id)})

    def count(self) -> int:
        """Return the number of documents in the collection."""
        try:
            self._ensure_store()
            return self._store._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all documents in the collection (dev use)."""
        global _vector_store, _chroma_client
        client = _get_chroma_client()
        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass
        _vector_store = None
        self._ensure_store()
