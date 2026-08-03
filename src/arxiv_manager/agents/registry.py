"""Agent registry — metadata, capabilities, and lifecycle management.

Stores both AgentMetadata (for API/introspection) and live Agent
instances (for runtime dispatch). Populated at startup in app.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .base import Agent

logger = logging.getLogger(__name__)


@dataclass
class AgentMetadata:
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    subscribe_events: list[str] = field(default_factory=list)
    version: str = "0.1.0"
    status: str = "active"  # active | inactive | degraded
    extra: dict[str, Any] = field(default_factory=dict)


_registry: dict[str, AgentMetadata] = {}
_instances: dict[str, Agent] = {}


def register_agent(
    name: str,
    description: str = "",
    capabilities: list[str] | None = None,
    subscribe_events: list[str] | None = None,
    version: str = "0.1.0",
    instance: Agent | None = None,
    **extra: Any,
) -> AgentMetadata:
    """Register an agent with the global registry."""
    meta = AgentMetadata(
        name=name,
        description=description,
        capabilities=capabilities or [],
        subscribe_events=subscribe_events or [],
        version=version,
        extra=extra,
    )
    _registry[name] = meta
    if instance is not None:
        _instances[name] = instance
    logger.info("agent_registry: registered '%s' (capabilities=%s, events=%s)",
                name, meta.capabilities, meta.subscribe_events)
    return meta


def get_agent(name: str) -> AgentMetadata | None:
    return _registry.get(name)


def get_agent_instance(name: str) -> Agent | None:
    """Return the live Agent instance registered under *name*."""
    return _instances.get(name)


def find_agents(capability: str, status: str = "active") -> list[AgentMetadata]:
    """Find all agents with a given capability."""
    return [
        meta for meta in _registry.values()
        if (not status or meta.status == status) and capability in meta.capabilities
    ]


def list_agents(status: str = "") -> list[AgentMetadata]:
    if not status:
        return list(_registry.values())
    return [m for m in _registry.values() if m.status == status]


def unregister_agent(name: str) -> bool:
    if name in _registry:
        del _registry[name]
        _instances.pop(name, None)
        logger.info("agent_registry: unregistered '%s'", name)
        return True
    return False


def clear_registry() -> None:
    _registry.clear()
    _instances.clear()


def register_all_agents() -> None:
    """Register all built-in agents. Called once at app startup."""
    from .determinism import DeterminismCheckerAgent
    from .fact_checker import FactCheckerAgent
    from .generator import GeneratorAgent
    from .issue_analyst import IssueAnalystAgent
    from .reviewer import ReviewerAgent
    from .self_critique import SelfCritiqueAgent
    from .verifier import VerifierAgent

    agents = [
        IssueAnalystAgent(),
        GeneratorAgent(),
        SelfCritiqueAgent(),
        FactCheckerAgent(),
        DeterminismCheckerAgent(),
        VerifierAgent(),
        ReviewerAgent(),
    ]

    for agent in agents:
        register_agent(
            name=agent.name,
            description=agent.__class__.__doc__ or "",
            capabilities=agent.capabilities,
            subscribe_events=agent.subscribe_events,
            instance=agent,
        )

    logger.info("agent_registry: registered %d agents at startup", len(agents))
