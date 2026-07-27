"""Agent context — shared state across multi-agent workflows.

AgentContext carries trace_id, figure metadata, artifacts, and delegation
chain information through a multi-agent orchestration flow. Each agent
receives and can modify the context, enabling collaboration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    trace_id: str
    figure_id: int
    difficulty: str
    figure_type: str
    user_id: str | None = None
    attempt_id: int | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    parent_context_id: str | None = None
    model_name: str = ""
    delegation_chain: list[str] = field(default_factory=list)

    def fork(self, agent_name: str, **overrides: Any) -> AgentContext:
        """Create a child context for subtask delegation.

        Args:
            agent_name: The name of the agent receiving the child context.
            **overrides: Fields to override in the child context.

        Returns:
            A new AgentContext linked to this one.
        """
        child = AgentContext(
            trace_id=self.trace_id,
            figure_id=self.figure_id,
            difficulty=self.difficulty,
            figure_type=self.figure_type,
            user_id=self.user_id,
            attempt_id=self.attempt_id,
            artifacts={**self.artifacts},
            parent_context_id=self.trace_id,
            model_name=self.model_name,
            delegation_chain=self.delegation_chain + [agent_name],
        )
        for key, value in overrides.items():
            if hasattr(child, key):
                setattr(child, key, value)
        return child

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact in the shared context."""
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Retrieve an artifact from the shared context."""
        return self.artifacts.get(key, default)


def new_context(
    figure_id: int,
    difficulty: str,
    figure_type: str,
    user_id: str | None = None,
    trace_id: str | None = None,
) -> AgentContext:
    """Create a fresh AgentContext.

    Args:
        figure_id: The figure being processed.
        difficulty: Difficulty level ("easy", "challenging", "hardest").
        figure_type: Type of figure ("chart_graph_text", "general_image").
        user_id: Optional user identifier.
        trace_id: Optional trace ID (auto-generated if not provided).

    Returns:
        A new AgentContext.
    """
    return AgentContext(
        trace_id=trace_id or uuid.uuid4().hex[:16],
        figure_id=figure_id,
        difficulty=difficulty,
        figure_type=figure_type,
        user_id=user_id,
    )
