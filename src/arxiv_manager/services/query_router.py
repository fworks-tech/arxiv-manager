"""Query router — analyzes requests and routes to the best pipeline.

Routes easy requests to simple generation, challenging to RAG-enhanced,
and hardest to consensus + critique + RAG.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def route_request(
    difficulty: str,
    figure_type: str,
    complexity_score: float,
    figure_id: int | None = None,
) -> dict[str, Any]:
    """Determine the optimal pipeline for a generation request.

    Returns a routing config dict with keys:
        - pipeline: "simple" | "rag_enhanced" | "consensus" | "self_critique"
        - use_rag: bool
        - use_consensus: bool
        - use_self_critique: bool
        - n_attempts: int (for consensus)
        - max_rounds: int (for self-critique)
    """
    if difficulty == "easy":
        return {
            "pipeline": "simple",
            "use_rag": False,
            "use_consensus": False,
            "use_self_critique": False,
        }

    if difficulty == "hardest":
        return {
            "pipeline": "self_critique",
            "use_rag": True,
            "use_consensus": True,
            "use_self_critique": True,
            "n_attempts": 3,
            "max_rounds": 2,
            "strength": "maximum",
        }

    # challenging
    if complexity_score >= 0.7:
        return {
            "pipeline": "rag_enhanced",
            "use_rag": True,
            "use_consensus": False,
            "use_self_critique": True,
            "max_rounds": 1,
        }

    return {
        "pipeline": "rag_enhanced",
        "use_rag": True,
        "use_consensus": False,
        "use_self_critique": False,
    }


def select_prompt_template(
    difficulty: str,
    figure_type: str,
    feedback: str = "",
) -> str:
    """Select the appropriate prompt template name based on routing context."""
    is_spatial = figure_type == "general_image"
    if feedback:
        return "SPATIAL_REGEN_PROMPT" if is_spatial else "REGEN_PROMPT"
    if difficulty == "hardest":
        return "SPATIAL_HARDEST_PROMPT" if is_spatial else "HARDEST_PROMPT"
    if difficulty == "challenging":
        return "SPATIAL_CHALLENGING_PROMPT" if is_spatial else "CHALLENGING_PROMPT"
    if difficulty == "easy":
        return "SPATIAL_DRAFT_PROMPT" if is_spatial else "EASY_PROMPT"
    return "SPATIAL_DRAFT_PROMPT" if is_spatial else "DRAFT_PROMPT"
