"""Document grader — scores retrieved documents for relevance to a generation task.

Used in the RAG pipeline to filter out low-quality retrievals before
they reach the generation prompt.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def grade_document(
    query: str,
    document: dict[str, Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Grade a single document's relevance to a query.

    Returns the document with a 'grade' field added:
        - grade: "relevant" | "partially_relevant" | "irrelevant"
        - relevance_score: 0.0 - 1.0

    Uses the retrieval score when available, falls back to simple
    keyword overlap analysis.
    """
    score = document.get("score", 0.0)
    content = document.get("content", "").lower()
    query_lower = query.lower()

    # Boost score if query terms appear in content
    query_words = set(query_lower.split())
    content_words = set(content.split())
    overlap = len(query_words & content_words)
    keyword_score = overlap / max(len(query_words), 1)

    combined = max(score, keyword_score)

    if combined >= threshold:
        grade = "relevant"
    elif combined >= threshold * 0.5:
        grade = "partially_relevant"
    else:
        grade = "irrelevant"

    document["grade"] = grade
    document["relevance_score"] = round(combined, 4)
    return document


def filter_relevant(
    documents: list[dict[str, Any]],
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Filter to only relevant/partially-relevant documents above a score."""
    graded = [grade_document("", d) for d in documents]
    return [
        d for d in graded
        if d["relevance_score"] >= min_score
    ]
