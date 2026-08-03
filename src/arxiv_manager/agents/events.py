"""Event bus — in-process pub/sub for agent communication.

Thread-safe, synchronous dispatch within a single worker process.
Agents subscribe to named events and emit new events from their
process() method. The orchestrator collects emitted events and
feeds them back into the bus until the pipeline quiesces.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .context import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class PipelineEvent:
    """A single event flowing through the agent pipeline.

    Attributes:
        event_type:   One of the EVENT_TYPES constants.
        context:      The AgentContext carrying artifacts between agents.
        source_agent: Name of the agent that emitted this event.
        metadata:     Arbitrary extra data (error messages, scores, etc.).
    """

    event_type: str
    context: AgentContext
    source_agent: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Event type constants
EVENT_TYPES = (
    "issue_reported",
    "draft_generated",
    "draft_validated",
    "fact_checked",
    "determinism_checked",
    "answer_verified",
    "review_completed",
    "regeneration_requested",
    "pipeline_completed",
    "pipeline_failed",
)

# Terminal events that stop the pipeline
TERMINAL_EVENTS = {"pipeline_completed", "pipeline_failed"}

# A subscriber is a callable that receives a PipelineEvent and returns
# a list of new PipelineEvents to emit (possibly empty).
Subscriber = Callable[[PipelineEvent], list[PipelineEvent]]


class EventBus:
    """Thread-safe in-process pub/sub event dispatcher.

    Dispatch is synchronous: emit() calls every subscriber in registration
    order and collects their returned events. Re-entrant emits (a subscriber
    emitting a new event) are appended to a pending queue and flushed after
    the current batch completes, preventing infinite recursion.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = {t: [] for t in EVENT_TYPES}
        self._lock = threading.Lock()
        self._pending: list[PipelineEvent] = []
        self._dispatching = False

    def subscribe(self, event_type: str, handler: Subscriber) -> None:
        """Register *handler* to be called when *event_type* fires."""
        if event_type not in self._subscribers:
            raise ValueError(f"Unknown event type: {event_type!r}")
        with self._lock:
            self._subscribers[event_type].append(handler)
        logger.debug("event_bus: subscribed %s to %s", getattr(handler, "__name__", handler), event_type)

    def unsubscribe(self, event_type: str, handler: Subscriber) -> None:
        """Remove *handler* from the subscriber list for *event_type*."""
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            try:
                subs.remove(handler)
            except ValueError:
                pass

    def emit(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Dispatch *event* to all subscribers and return all produced events.

        Thread-safe. Re-entrant calls from within subscribers are queued
        and flushed after the current dispatch completes.
        """
        with self._lock:
            if self._dispatching:
                self._pending.append(event)
                return []

            self._dispatching = True

        all_produced: list[PipelineEvent] = []
        try:
            all_produced.extend(self._dispatch(event))
            # Flush re-entrant events
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    batch = self._pending[:]
                    self._pending.clear()
                for pending_event in batch:
                    all_produced.extend(self._dispatch(pending_event))
        finally:
            with self._lock:
                self._dispatching = False

        return all_produced

    def _dispatch(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Call all subscribers for event.event_type and collect results."""
        with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))

        produced: list[PipelineEvent] = []
        for handler in handlers:
            try:
                result = handler(event)
                if result:
                    produced.extend(result)
            except Exception as exc:
                logger.error(
                    "event_bus: handler %s failed on %s: %s",
                    getattr(handler, "__name__", handler),
                    event.event_type,
                    exc,
                    exc_info=True,
                )
        return produced

    def subscriber_count(self, event_type: str | None = None) -> int:
        """Return the number of registered subscribers."""
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(v) for v in self._subscribers.values())

    def reset(self) -> None:
        """Remove all subscribers (useful for testing)."""
        with self._lock:
            for key in self._subscribers:
                self._subscribers[key].clear()
            self._pending.clear()
            self._dispatching = False


# Module-level singleton (one per worker process)
_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the module-level EventBus singleton, creating it on first call."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Reset the global bus (for tests)."""
    global _bus
    with _bus_lock:
        if _bus is not None:
            _bus.reset()
        _bus = None
