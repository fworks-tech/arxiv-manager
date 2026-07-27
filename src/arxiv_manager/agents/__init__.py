"""Agents: registry, context, orchestrator, reviewer, adaptive routing, tools."""

from __future__ import annotations

from .context import AgentContext, new_context
from .orchestrator import orchestrate
from .registry import (
    AgentMetadata,
    clear_registry,
    find_agents,
    get_agent,
    list_agents,
    register_agent,
    unregister_agent,
)
from .reviewer import review_draft

__all__ = [
    "AgentMetadata", "register_agent", "get_agent", "find_agents",
    "list_agents", "unregister_agent", "clear_registry",
    "AgentContext", "new_context",
    "orchestrate",
    "review_draft",
]
