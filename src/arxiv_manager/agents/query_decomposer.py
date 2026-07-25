"""Query decomposer — breaks complex generation tasks into sub-tasks.

For "hardest" difficulty, decomposes the task into manageable sub-questions
that can be answered independently and combined.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def decompose_query(
    difficulty: str,
    figure_type: str,
    prompt: str,
) -> list[dict[str, Any]]:
    """Decompose a generation prompt into sub-tasks.

    For hardest difficulty, this breaks the prompt into reasoning steps.
    For easier difficulties, returns a single task.

    Returns list of sub-task dicts with 'description' and 'order' keys.
    """
    if difficulty != "hardest":
        return [{"description": prompt, "order": 0}]

    # For hardest, suggest decomposition strategies based on figure type
    if figure_type == "chart_graph_text":
        return [
            {"description": "Identify all data panels and their labels", "order": 0},
            {"description": "For each panel, extract key data values (peaks, intersections, ratios)", "order": 1},
            {"description": "Cross-reference values across panels", "order": 2},
            {"description": "Formulate a multi-step question requiring at least 2 of the above", "order": 3},
        ]
    else:
        return [
            {"description": "Catalog all visible objects and their positions", "order": 0},
            {"description": "Identify spatial relationships (left/right, above/below, depth)", "order": 1},
            {"description": "Classify objects by attributes (color, size, type)", "order": 2},
            {"description": "Formulate a question requiring multi-attribute reasoning", "order": 3},
        ]


def format_decomposition(sub_tasks: list[dict[str, Any]]) -> str:
    """Format a list of sub-tasks into a prompt block."""
    if len(sub_tasks) <= 1:
        return sub_tasks[0]["description"] if sub_tasks else ""
    lines = []
    for t in sorted(sub_tasks, key=lambda x: x["order"]):
        lines.append(f"Step {t['order'] + 1}: {t['description']}")
    return "\n".join(lines)
