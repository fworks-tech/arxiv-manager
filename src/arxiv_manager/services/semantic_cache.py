"""Semantic cache (CAG layer) using ChromaDB.

Stores LLM generation results keyed by prompt embeddings.
Future identical/similar prompts return cached results instead of
calling the LLM again.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from ..components.hybrid_retriever import HybridRetriever
from ..components.retriever_config import COLLECTION_CACHE, DEFAULT_CACHE_TOP_K

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache for LLM generation results.

    Uses ChromaDB to store prompt → result mappings, keyed by
    embeddings of the prompt text. Similar prompts (within a
    similarity threshold) return cached results.

    Usage:
        cache = SemanticCache()
        result = cache.get(prompt, figure_id=42)
        if result is None:
            result = llm_call(prompt)
            cache.set(prompt, result, figure_id=42)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        collection_name: str = COLLECTION_CACHE,
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._retriever = HybridRetriever(collection_name=collection_name)

    def get(
        self,
        prompt: str,
        figure_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Look up a prompt in the semantic cache.

        Returns cached result if a similar-enough prompt exists
        and hasn't expired. Returns None on miss.
        """
        filter = None
        if figure_id is not None:
            filter = {"figure_id": str(figure_id)}

        results = self._retriever.search(
            query=prompt,
            k=DEFAULT_CACHE_TOP_K,
            filter=filter,
        )

        for result in results:
            if result["score"] < self._threshold:
                continue
            cached = result["metadata"]
            ts = cached.get("cached_at", 0)
            if time.time() - ts < self._ttl:
                logger.debug("semantic_cache: HIT (score=%.4f)", result["score"])
                return {
                    "question": cached.get("question", ""),
                    "answer": cached.get("answer", ""),
                    "answer_format": cached.get("answer_format", "word"),
                    "task_type": cached.get("task_type", "chart"),
                    "_from_cache": True,
                }
            else:
                logger.debug("semantic_cache: EXPIRED (score=%.4f)", result["score"])

        logger.debug("semantic_cache: MISS")
        return None

    def set(
        self,
        prompt: str,
        result: dict[str, Any],
        figure_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a generation result in the cache."""
        meta = {
            "figure_id": str(figure_id) if figure_id else "",
            "question": result.get("question", ""),
            "answer": result.get("answer", ""),
            "answer_format": result.get("answer_format", ""),
            "task_type": result.get("task_type", ""),
            "cached_at": time.time(),
            "ttl": self._ttl,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            **(metadata or {}),
        }
        self._retriever.add_texts(
            texts=[prompt],
            metadatas=[meta],
        )
        logger.debug("semantic_cache: SET (len=%d)", len(prompt))

    def clear(self) -> None:
        """Clear the entire cache."""
        self._retriever.clear()

    def count(self) -> int:
        """Return number of cached entries."""
        return self._retriever.count()
