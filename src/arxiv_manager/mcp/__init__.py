"""MCP (Model Context Protocol) server and tool definitions.

Exposes the system's capabilities as MCP tools that can be called
by AI agents (including OpenCode) programmatically.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])

logger = logging.getLogger(__name__)

# Tool registry
_tools: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    description: str,
    handler: callable,
    input_schema: dict[str, Any] | None = None,
) -> None:
    """Register a tool with the MCP server."""
    _tools[name] = {
        "name": name,
        "description": description,
        "handler": handler,
        "input_schema": input_schema or {"type": "object", "properties": {}},
    }


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _tool_generate_qa(
    image_path: str,
    difficulty: str = "challenging",
    figure_type: str = "chart_graph_text",
    complexity_score: float = 0.5,
    caption: str = "",
    figure_id: int | None = None,
) -> dict[str, Any]:
    """Generate a visual-reasoning Q&A pair from an image."""
    from ..authoring.ai_draft.core import draft_qa
    result = draft_qa(
        image_path=image_path,
        difficulty=difficulty,
        figure_type=figure_type,
        complexity_score=complexity_score,
        caption=caption,
        figure_id=figure_id,
    )
    if result:
        return {"question": result.get("question", ""), "answer": result.get("answer", "")}
    return {"error": "Generation failed"}


def _tool_validate_qa(
    question: str,
    answer: str,
    answer_format: str = "word",
    figure_type: str = "",
    task_type: str = "",
) -> dict[str, Any]:
    """Validate a Q&A pair against handbook rules."""
    from ..authoring.validator import validate_task
    v = validate_task(
        question=question, answer=answer,
        answer_format=answer_format,
        figure_type=figure_type, task_type=task_type,
    )
    return {
        "is_valid": v.is_valid,
        "quality_score": v.quality_score,
        "errors": v.errors,
        "warnings": v.warnings,
    }


def _tool_figure_history(figure_id: int) -> str:
    """Get past generation attempts for a figure."""
    from ..authoring._history_context import build_figure_history
    return build_figure_history(figure_id=figure_id)


def _tool_search_figures(
    query: str,
    figure_type: str = "",
    k: int = 5,
) -> list[dict[str, Any]]:
    """Search indexed figures by caption content."""
    try:
        from ..components.hybrid_retriever import HybridRetriever
        r = HybridRetriever()
        return r.search(query=query, k=k, filter={"figure_type": figure_type} if figure_type else None)
    except Exception as e:
        logger.debug("search_figures: %s", e)
        return []


def _tool_health() -> dict[str, Any]:
    """Get system health status."""
    from ..web.routes.health import _check_api_key, _check_db
    return {
        "db": _check_db(),
        "api_key": _check_api_key(),
    }


def _tool_analytics(difficulty: str = "", figure_type: str = "") -> dict[str, Any]:
    """Get generation analytics and performance stats."""
    from ..agents.adaptive_router import get_pipeline_stats
    return get_pipeline_stats(figure_type=figure_type, difficulty=difficulty)


# Register all tools
register_tool("generate_qa", "Generate a visual-reasoning Q&A pair from an image", _tool_generate_qa, {
    "type": "object",
    "properties": {
        "image_path": {"type": "string", "description": "Path to the figure image"},
        "difficulty": {"type": "string", "enum": ["easy", "challenging", "hardest"]},
        "figure_type": {"type": "string", "enum": ["chart_graph_text", "general_image"]},
    },
    "required": ["image_path"],
})

register_tool("validate_qa", "Validate a Q&A pair against handbook rules", _tool_validate_qa, {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "answer_format": {"type": "string", "enum": ["number", "word", "phrase", "year", "percent", "integer"]},
    },
    "required": ["question", "answer"],
})

register_tool("figure_history", "Get past generation attempts for a figure", _tool_figure_history, {
    "type": "object",
    "properties": {"figure_id": {"type": "integer"}},
    "required": ["figure_id"],
})

register_tool("search_figures", "Search indexed figures by caption content", _tool_search_figures, {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "figure_type": {"type": "string"},
        "k": {"type": "integer"},
    },
    "required": ["query"],
})

register_tool("health", "Get system health status", _tool_health)
register_tool("analytics", "Get generation analytics and performance stats", _tool_analytics)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@mcp_router.get("/tools")
def list_tools() -> list[dict[str, Any]]:
    """List all available MCP tools with their schemas."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        for t in _tools.values()
    ]


@mcp_router.post("/tools/{tool_name}/call")
async def call_tool(tool_name: str, body: dict[str, Any]) -> dict[str, Any]:
    """Call an MCP tool with the given arguments."""
    tool = _tools.get(tool_name)
    if not tool:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        handler = tool["handler"]
        result = handler(**body)
        return {"result": result}
    except Exception as e:
        logger.error("mcp: tool %s failed: %s", tool_name, e)
        return {"error": str(e)}


@mcp_router.get("/health")
def mcp_health() -> dict[str, Any]:
    """MCP server health check."""
    return {"status": "ok", "tools": len(_tools)}
