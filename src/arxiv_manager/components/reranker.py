"""Cross-encoder reranker for improving retrieval quality.

Takes top-k results from the hybrid retriever and re-ranks them
using a cross-encoder model that evaluates query-document pairs directly.
"""

from __future__ import annotations

import logging
from typing import Any

from .retriever_config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder model
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(RERANKER_MODEL_NAME)
            logger.info("reranker: loaded %s", RERANKER_MODEL_NAME)
        except Exception as e:
            logger.warning("reranker: failed to load model: %s", e)
            return None
    return _reranker


def rerank(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int | None = None,
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """Re-rank documents by cross-encoder relevance score.

    Args:
        query: The search query.
        documents: List of dicts with at least 'content' key.
        top_k: Number of results to return (None = all).
        score_key: Key to store the reranker score under.

    Returns:
        Re-ranked documents sorted by relevance score (descending).
    """
    model = _get_reranker()
    if model is None:
        # Fall through: return original order with 0 scores
        for doc in documents:
            doc[score_key] = 0.0
        return documents[:top_k] if top_k else documents

    pairs = [(query, doc["content"]) for doc in documents]
    scores = model.predict(pairs)

    scored = list(zip(documents, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for doc, score in scored:
        doc[score_key] = round(float(score), 4)
        results.append(doc)

    if top_k:
        results = results[:top_k]

    return results
