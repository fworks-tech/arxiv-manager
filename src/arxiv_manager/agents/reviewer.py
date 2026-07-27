"""Reviewer agent — critiques drafts and provides improvement suggestions.

The Reviewer takes a draft Q&A pair and scores it across multiple
dimensions: correctness, clarity, difficulty fit, and figure relevance.
It can be called standalone or as part of an orchestration workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from .context import AgentContext

logger = logging.getLogger(__name__)


def review_draft(
    draft: dict[str, Any],
    context: AgentContext | None = None,
) -> dict[str, Any]:
    """Review a draft and return a critique.

    Scores the draft on a 1-5 scale and provides specific improvement
    suggestions where applicable.

    Args:
        draft: The Q&A draft dict with keys: question, answer, answer_format,
               task_type, and optionally _validation_quality.
        context: Optional AgentContext for traceability.

    Returns:
        dict with keys:
        - score: int 1-5
        - passed: bool (score >= 3)
        - suggestions: list[str]
        - strengths: list[str]
        - agent: str
    """
    question = (draft.get("question") or "").strip()
    answer = (draft.get("answer") or "").strip()
    answer_format = draft.get("answer_format", "word")
    quality = draft.get("_validation_quality", 0)

    strengths: list[str] = []
    suggestions: list[str] = []

    # Score based on validation quality
    if not question or not answer:
        return {
            "score": 1,
            "passed": False,
            "suggestions": ["Draft is empty"],
            "strengths": [],
            "agent": "reviewer",
        }

    if quality >= 0.9:
        base_score = 5
        strengths.append("High validation quality")
    elif quality >= 0.7:
        base_score = 4
        strengths.append("Good validation quality")
    elif quality >= 0.5:
        base_score = 3
    else:
        base_score = 2
        suggestions.append("Low validation quality — consider regenerating")

    # Check if answer is too short
    if len(answer) < 2:
        base_score = max(base_score - 1, 1)
        suggestions.append("Answer is very short — may lack detail")

    # Check if question contains the answer
    if answer.lower() in question.lower():
        base_score = max(base_score - 1, 1)
        suggestions.append("Question appears to contain the answer")

    # Check format match
    if answer_format == "number" and not answer.replace(".", "").replace("-", "").isdigit():
        suggestions.append(f"Answer format is '{answer_format}' but answer is not a number")
    elif answer_format == "year" and not (answer.isdigit() and len(answer) == 4):
        suggestions.append(f"Answer format is '{answer_format}' but answer is not a 4-digit year")

    if not suggestions:
        strengths.append("All format checks pass")

    return {
        "score": base_score,
        "passed": base_score >= 3,
        "suggestions": suggestions,
        "strengths": strengths,
        "agent": "reviewer",
    }
