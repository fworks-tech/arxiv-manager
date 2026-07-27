"""Adaptive router — learns from past generation performance to improve routing.

Observes success rates per (difficulty, figure_type, pipeline) combination
and adjusts routing recommendations accordingly.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import get_session
from ..models import GenerationAttempt

logger = logging.getLogger(__name__)


def get_pipeline_stats(
    figure_type: str = "",
    difficulty: str = "",
    min_samples: int = 3,
) -> dict[str, Any]:
    """Query past generation attempts and return performance stats per pipeline.

    Returns dict with pipeline names as keys and success rate + avg quality.
    """
    from sqlmodel import desc, select

    session = get_session()
    try:
        query = select(GenerationAttempt).where(
            GenerationAttempt.success,
            GenerationAttempt.validation_quality > 0,
            GenerationAttempt.generation_type != "",
        )
        if figure_type:
            query = query.where(GenerationAttempt.figure_type == figure_type)
        if difficulty:
            query = query.where(GenerationAttempt.difficulty == difficulty)

        rows = list(session.exec(query.order_by(desc(GenerationAttempt.created_at))).all())
        session.close()

        if not rows:
            return {}

        by_pipeline: dict[str, list[float]] = {}
        for r in rows:
            pipeline = r.generation_type or "draft"
            by_pipeline.setdefault(pipeline, []).append(r.validation_quality)

        stats = {}
        for pipeline, qualities in by_pipeline.items():
            if len(qualities) >= min_samples:
                stats[pipeline] = {
                    "samples": len(qualities),
                    "avg_quality": round(sum(qualities) / len(qualities), 1),
                    "max_quality": round(max(qualities), 1),
                }

        return stats
    except Exception:
        return {}


def recommend_pipeline(
    difficulty: str,
    figure_type: str,
    fallback: str = "self_critique",
) -> str:
    """Recommend the best pipeline based on historical performance.

    Falls back to the rule-based router if insufficient data.
    """
    stats = get_pipeline_stats(figure_type=figure_type, difficulty=difficulty)
    if not stats:
        return fallback

    best_pipeline = fallback
    best_quality = 0.0
    for pipeline, info in stats.items():
        if info["avg_quality"] > best_quality:
            best_quality = info["avg_quality"]
            best_pipeline = pipeline

    if best_pipeline != fallback:
        logger.info(
            "adaptive_router: %s over %s (avg=%.1f, n=%d) for diff=%s type=%s",
            best_pipeline,
            fallback,
            best_quality,
            stats.get(best_pipeline, {}).get("samples", 0),
            difficulty,
            figure_type,
        )

    return best_pipeline
