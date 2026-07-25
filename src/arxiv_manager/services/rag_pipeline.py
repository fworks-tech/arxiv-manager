"""CRAG pipeline — RAG + CAG working together.

Orchestrates:
1. Check semantic cache (CAG) for an identical/similar prompt
2. On cache miss: hybrid retrieve (keyword + semantic) from figure index
3. Rerank results using cross-encoder
4. Inject retrieved context into the generation prompt
5. Cache the new result for future queries
"""

from __future__ import annotations

import logging
from typing import Any

from ..components.hybrid_retriever import HybridRetriever
from ..components.reranker import rerank
from ..components.retriever_config import DEFAULT_TOP_K
from .semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class RAGPipeline:
    """CRAG pipeline: CAG (semantic cache) → RAG (retrieve + rerank) → generate.

    Usage:
        rag = RAGPipeline()
        context = rag.get_context(
            prompt="Generate a visual-reasoning question...",
            figure_id=42,
            figure_type="chart_graph_text",
        )
        # context.context_str contains the injected text
        # Append context.context_str to your prompt before calling the LLM
    """

    def __init__(
        self,
        use_cache: bool = True,
        use_reranker: bool = True,
    ) -> None:
        self._use_cache = use_cache
        self._use_reranker = use_reranker
        self._cache = SemanticCache() if use_cache else None
        self._retriever = HybridRetriever()

    def get_context(
        self,
        query: str,
        figure_id: int | None = None,
        figure_type: str = "",
        difficulty: str = "",
        top_k: int = DEFAULT_TOP_K,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a context string for injection into a generation prompt.

        Returns:
            Dict with 'context_str' (the text to inject) and
            'sources' (list of source metadata for traceability).
        """
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

        results = self._retriever.search(
            query=query,
            k=top_k * 2,  # Fetch more for reranking
            filter=filter_clause or None,
        )

        # 3. Rerank if enabled
        if self._use_reranker and results and len(results) > 1:
            results = rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # 4. Build context string from results
        if not results:
            return {"context_str": "", "sources": [], "from_cache": False}

        context_parts = []
        sources = []
        for r in results:
            context_parts.append(f"- {r['content']}")
            sources.append({
                "figure_id": r["metadata"].get("figure_id"),
                "figure_type": r["metadata"].get("figure_type"),
                "score": r.get("score", r.get("rerank_score", 0)),
            })

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
        return self._retriever.add_figure(
            figure_id=figure_id,
            caption=caption,
            figure_type=figure_type,
            paper_title=paper_title,
            difficulty=difficulty,
            question=question,
            metadata=metadata,
        )
