"""LangChain tools for the agent framework.

Wraps existing functionality (validate_qa, get_figure_history, search_figures)
as LangChain Tool objects that can be used by agents.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidateQATool:
    """Tool: validate a Q&A pair against handbook rules.

    Usage:
        tool = ValidateQATool()
        result = tool.run(question="...", answer="...", answer_format="number")
    """

    name: str = "validate_qa"
    description: str = "Validate a Q&A question and answer against the handbook rules"

    def run(
        self,
        question: str,
        answer: str,
        answer_format: str = "word",
        figure_type: str = "",
        task_type: str = "",
    ) -> dict[str, Any]:
        from ..authoring.validator import validate_task

        v = validate_task(
            question=question,
            answer=answer,
            answer_format=answer_format,
            figure_type=figure_type,
            task_type=task_type,
        )
        return {
            "is_valid": v.is_valid,
            "quality_score": v.quality_score,
            "errors": v.errors,
            "warnings": v.warnings,
        }


class FigureHistoryTool:
    """Tool: retrieve generation history for a figure.

    Usage:
        tool = FigureHistoryTool()
        history = tool.run(figure_id=42)
    """

    name: str = "figure_history"
    description: str = "Get past generation attempts for a figure"

    def run(self, figure_id: int) -> str:
        from ..authoring._history_context import build_figure_history

        return build_figure_history(figure_id=figure_id)


class SearchFiguresTool:
    """Tool: search indexed figures by keywords or figure type.

    Usage:
        tool = SearchFiguresTool()
        results = tool.run(query="neural network", figure_type="chart_graph_text")
    """

    name: str = "search_figures"
    description: str = "Search indexed figures by caption content and figure type"

    def run(
        self,
        query: str,
        figure_type: str = "",
        k: int = 5,
    ) -> list[dict[str, Any]]:
        try:
            from ..components.hybrid_retriever import HybridRetriever

            r = HybridRetriever()
            return r.search(query=query, k=k, filter={"figure_type": figure_type} if figure_type else None)
        except Exception as e:
            logger.debug("search_figures: retriever unavailable: %s", e)
            return []
