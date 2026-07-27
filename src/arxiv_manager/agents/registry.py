"""Agent registry — metadata, capabilities, and lifecycle management.

Provides a simple dict-based registry for agents used in the multi-agent
collaboration system. Agents register their capabilities at startup
and can be looked up by name or capability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    version: str = "0.1.0"
    status: str = "active"  # active | inactive | degraded
    extra: dict[str, Any] = field(default_factory=dict)


_registry: dict[str, AgentMetadata] = {}


def register_agent(
    name: str,
    description: str = "",
    capabilities: list[str] | None = None,
    version: str = "0.1.0",
    **extra: Any,
) -> AgentMetadata:
    """Register an agent with the global registry.

    Args:
        name: Unique agent name.
        description: Human-readable description.
        capabilities: List of capability strings (e.g. "draft_qa", "critique").
        version: Semantic version string.
        **extra: Arbitrary extra metadata.

    Returns:
        The created AgentMetadata.
    """
    meta = AgentMetadata(
        name=name,
        description=description,
        capabilities=capabilities or [],
        version=version,
        extra=extra,
    )
    _registry[name] = meta
    logger.info("agent_registry: registered '%s' (capabilities=%s)", name, meta.capabilities)
    return meta


def get_agent(name: str) -> AgentMetadata | None:
    """Look up an agent by name."""
    return _registry.get(name)


def find_agents(capability: str, status: str = "active") -> list[AgentMetadata]:
    """Find all agents with a given capability.

    Args:
        capability: The capability to search for.
        status: Filter by status ("active", "inactive", or "" for all).

    Returns:
        List of matching AgentMetadata.
    """
    results = []
    for meta in _registry.values():
        if status and meta.status != status:
            continue
        if capability in meta.capabilities:
            results.append(meta)
    return results


def list_agents(status: str = "") -> list[AgentMetadata]:
    """List all registered agents.

    Args:
        status: Optional filter by status.

    Returns:
        List of AgentMetadata.
    """
    if not status:
        return list(_registry.values())
    return [m for m in _registry.values() if m.status == status]


def unregister_agent(name: str) -> bool:
    """Remove an agent from the registry.

    Returns:
        True if the agent was found and removed.
    """
    if name in _registry:
        del _registry[name]
        logger.info("agent_registry: unregistered '%s'", name)
        return True
    return False


def clear_registry() -> None:
    """Clear all registered agents (useful for testing)."""
    _registry.clear()
