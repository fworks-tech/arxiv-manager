"""Abstract base class for all agents in the pipeline.

Every agent has a name, a set of capabilities, a list of events it
subscribes to, and a process() method that receives a PipelineEvent
and returns zero or more new PipelineEvents.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import PipelineEvent

logger = logging.getLogger(__name__)


class Agent(abc.ABC):
    """Abstract base for pipeline agents.

    Subclasses must define:
        name:           Unique string identifier (e.g. "generator").
        capabilities:   List of capability strings (e.g. ["draft_qa"]).
        subscribe_events: List of event type strings this agent handles.

    Subclasses must implement:
        process(event) -> list[PipelineEvent]
    """

    name: str
    capabilities: list[str]
    subscribe_events: list[str]

    @abc.abstractmethod
    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Handle an incoming event and return new events to emit.

        Args:
            event: The PipelineEvent triggering this agent.

        Returns:
            List of new PipelineEvents to emit (may be empty).
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
