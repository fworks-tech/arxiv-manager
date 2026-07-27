"""CRAG pipeline — RAG + CAG working together.

Orchestrates:
1. Check semantic cache (CAG) for an identical/similar prompt
2. On cache miss: hybrid retrieve (keyword + semantic) from figure index
3. Rerank results using cross-encoder
4. Inject retrieved context into the generation prompt
5. Cache the new result for future queries

Uses a module-level lazy singleton to avoid reloading the embedding
model and ChromaDB client on every request.
"""

from __future__ import annotations

import logging
from typing import Any

from ..components.retriever_config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

# Lazy singleton
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Get or create the singleton RAGPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        logger.info("rag_pipeline: singleton created")
    return _pipeline


class RAGPipeline:
    """CRAG pipeline: CAG (semantic cache) → RAG (retrieve + rerank) → generate.

    Usage:
        rag = get_pipeline()
        context = rag.get_context(prompt="...", figure_id=42, ...)
    """

    def __init__(
        self,
        use_cache: bool = True,
        use_reranker: bool = True,
    ) -> None:
        self._use_cache = use_cache
        self._use_reranker = use_reranker
        self._cache = None
        self._retriever = None

    def _lazy_init(self) -> None:
        if self._retriever is not None:
            return
        from ..components.hybrid_retriever import HybridRetriever
        from ..services.semantic_cache import SemanticCache

        self._retriever = HybridRetriever()
        if self._use_cache:
            self._cache = SemanticCache()

    def get_context(
        self,
        query: str,
        figure_id: int | None = None,
        figure_type: str = "",
        difficulty: str = "",
        top_k: int = DEFAULT_TOP_K,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a context string for injection into a generation prompt."""
        self._lazy_init()

        # 1. Check semantic cache (CAG)
        if self._cache and figure_id:
            cached = self._cache.get(query, figure_id=figure_id)
            if cached:
                logger.info("rag_pipeline: cache HIT for figure_id=%s", figure_id)
                return {
                    "context_str": cached.get("context_str", ""),
                    "sources": [{"type": "cache"}],
                    "from_cache": True,
                }

        # 2. Hybrid retrieve from figure index
        filter_clause = {}
        if figure_type:
            filter_clause["figure_type"] = figure_type
        if difficulty:
            filter_clause["difficulty"] = difficulty

        if len(filter_clause) > 1:
            filter_clause = {"$and": [{k: {"$eq": v}} for k, v in filter_clause.items()]}

        results = self._retriever.search(
            query=query,
            k=top_k * 2,
            filter=filter_clause or None,
        )

        # 3. Rerank if enabled
        if self._use_reranker and results and len(results) > 1:
            from ..components.reranker import rerank

            results = rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # 4. Build context string
        if not results:
            return {"context_str": "", "sources": [], "from_cache": False}

        context_parts = []
        sources = []
        for r in results:
            context_parts.append(f"- {r['content']}")
            sources.append(
                {
                    "figure_id": r["metadata"].get("figure_id"),
                    "figure_type": r["metadata"].get("figure_type"),
                    "score": r.get("score", r.get("rerank_score", 0)),
                }
            )

        context_str = "Retrieved context:\n" + "\n".join(context_parts)

        return {
            "context_str": context_str,
            "sources": sources,
            "from_cache": False,
        }

    def cache_result(
        self,
        prompt: str,
        result: dict[str, Any],
        figure_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a generation result in the semantic cache."""
        self._lazy_init()
        if self._cache:
            self._cache.set(prompt, result, figure_id=figure_id, metadata=metadata)

    def index_figure(
        self,
        figure_id: int,
        caption: str,
        figure_type: str,
        paper_title: str = "",
        difficulty: str = "",
        question: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Index a figure into the vector store for future retrieval."""
        self._lazy_init()
        return self._retriever.add_figure(
            figure_id=figure_id,
            caption=caption,
            figure_type=figure_type,
            paper_title=paper_title,
            difficulty=difficulty,
            question=question,
            metadata=metadata,
        )
