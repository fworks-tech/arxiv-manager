"""Agents: event bus, base class, registry, context, orchestrator, specialized agents."""

from __future__ import annotations

from .base import Agent
from .context import AgentContext, new_context
from .events import EventBus, PipelineEvent, get_event_bus
from .orchestrator import run_pipeline, run_regeneration
from .registry import (
    AgentMetadata,
    clear_registry,
    find_agents,
    get_agent,
    get_agent_instance,
    list_agents,
    register_agent,
    register_all_agents,
    unregister_agent,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentMetadata",
    "EventBus",
    "PipelineEvent",
    "clear_registry",
    "find_agents",
    "get_agent",
    "get_agent_instance",
    "get_event_bus",
    "list_agents",
    "new_context",
    "register_agent",
    "register_all_agents",
    "run_pipeline",
    "run_regeneration",
    "unregister_agent",
]
