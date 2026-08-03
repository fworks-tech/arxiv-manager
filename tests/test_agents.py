"""Tests for the agent event bus, base class, and registry."""

from __future__ import annotations

import pytest

from arxiv_manager.agents.base import Agent
from arxiv_manager.agents.context import new_context
from arxiv_manager.agents.events import EventBus, PipelineEvent

# ---------------------------------------------------------------------------
# EventBus tests
# ---------------------------------------------------------------------------


class TestEventBus:
    def setup_method(self):
        self.bus = EventBus()

    def test_subscribe_and_emit(self):
        received = []

        def handler(event: PipelineEvent):
            received.append(event)
            return []

        self.bus.subscribe("draft_generated", handler)
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        evt = PipelineEvent(event_type="draft_generated", context=ctx)
        produced = self.bus.emit(evt)

        assert len(received) == 1
        assert received[0] is evt
        assert produced == []

    def test_emit_returns_produced_events(self):
        def handler(event: PipelineEvent):
            return [PipelineEvent(event_type="draft_validated", context=event.context)]

        self.bus.subscribe("draft_generated", handler)
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        produced = self.bus.emit(PipelineEvent(event_type="draft_generated", context=ctx))

        assert len(produced) == 1
        assert produced[0].event_type == "draft_validated"

    def test_reentrant_emit_is_deferred(self):
        events_seen = []

        def inner_handler(event: PipelineEvent):
            events_seen.append("inner")
            return []

        def outer_handler(event: PipelineEvent):
            events_seen.append("outer")
            # This re-entrant emit should be deferred
            self.bus.emit(PipelineEvent(event_type="draft_validated", context=event.context))
            return []

        self.bus.subscribe("draft_generated", outer_handler)
        self.bus.subscribe("draft_validated", inner_handler)

        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        self.bus.emit(PipelineEvent(event_type="draft_generated", context=ctx))

        assert events_seen == ["outer", "inner"]

    def test_handler_exception_does_not_break_pipeline(self):
        def bad_handler(event: PipelineEvent):
            raise ValueError("boom")

        def good_handler(event: PipelineEvent):
            return [PipelineEvent(event_type="draft_validated", context=event.context)]

        self.bus.subscribe("draft_generated", bad_handler)
        self.bus.subscribe("draft_generated", good_handler)

        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        produced = self.bus.emit(PipelineEvent(event_type="draft_generated", context=ctx))

        # pipeline_failed from the bad handler + draft_validated from the good handler
        assert len(produced) == 2
        assert produced[0].event_type == "pipeline_failed"
        assert produced[0].source_agent == "event_bus"
        assert produced[1].event_type == "draft_validated"
        assert ctx.errors

    def test_subscriber_count(self):
        assert self.bus.subscriber_count("draft_generated") == 0

        def h(e):
            return []

        self.bus.subscribe("draft_generated", h)
        self.bus.subscribe("draft_generated", h)
        assert self.bus.subscriber_count("draft_generated") == 2
        assert self.bus.subscriber_count() == 2

    def test_reset(self):
        def h(e):
            return []

        self.bus.subscribe("draft_generated", h)
        self.bus.reset()
        assert self.bus.subscriber_count() == 0

    def test_unknown_event_type_raises(self):
        with pytest.raises(ValueError, match="Unknown event type"):
            self.bus.subscribe("nonexistent", lambda e: [])


# ---------------------------------------------------------------------------
# AgentContext tests
# ---------------------------------------------------------------------------


class TestAgentContext:
    def test_new_context(self):
        ctx = new_context(figure_id=42, difficulty="hardest", figure_type="chart_graph_text")
        assert ctx.figure_id == 42
        assert ctx.difficulty == "hardest"
        assert ctx.figure_type == "chart_graph_text"
        assert ctx.pipeline_status == "pending"
        assert ctx.errors == []

    def test_set_get_artifact(self):
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.set_artifact("question", "What is X?")
        assert ctx.get_artifact("question") == "What is X?"
        assert ctx.get_artifact("missing", "default") == "default"

    def test_add_error(self):
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.add_error("something went wrong")
        assert ctx.errors == ["something went wrong"]
        assert ctx.pipeline_status == "failed"

    def test_fork_copies_errors(self):
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.add_error("original error")
        child = ctx.fork("generator")
        assert child.errors == ["original error"]
        assert child.pipeline_status == "failed"


# ---------------------------------------------------------------------------
# Agent base class tests
# ---------------------------------------------------------------------------


class TestAgentBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Agent()  # type: ignore

    def test_concrete_agent(self):
        class DummyAgent(Agent):
            name = "dummy"
            capabilities = ["test"]
            subscribe_events = ["draft_generated"]

            def process(self, event):
                return []

        agent = DummyAgent()
        assert agent.name == "dummy"
        assert "test" in agent.capabilities
        assert repr(agent) == "<DummyAgent name='dummy'>"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self):
        from arxiv_manager.agents.registry import clear_registry
        clear_registry()

    def teardown_method(self):
        from arxiv_manager.agents.registry import clear_registry
        clear_registry()

    def test_register_and_get(self):
        from arxiv_manager.agents.registry import get_agent, get_agent_instance, register_agent

        class DummyAgent(Agent):
            name = "dummy"
            capabilities = ["test"]
            subscribe_events = []

            def process(self, event):
                return []

        instance = DummyAgent()
        register_agent("dummy", "Test agent", ["test"], ["draft_generated"], instance=instance)

        meta = get_agent("dummy")
        assert meta is not None
        assert meta.name == "dummy"
        assert meta.capabilities == ["test"]

        inst = get_agent_instance("dummy")
        assert inst is instance

    def test_find_agents(self):
        from arxiv_manager.agents.registry import find_agents, register_agent

        register_agent("a", capabilities=["draft_qa"])
        register_agent("b", capabilities=["review"])

        found = find_agents("draft_qa")
        assert len(found) == 1
        assert found[0].name == "a"

    def test_unregister(self):
        from arxiv_manager.agents.registry import get_agent, register_agent, unregister_agent

        register_agent("temp")
        assert get_agent("temp") is not None
        assert unregister_agent("temp") is True
        assert get_agent("temp") is None

    def test_register_all_agents(self):
        from arxiv_manager.agents.registry import list_agents, register_all_agents

        register_all_agents()
        agents = list_agents()
        names = {a.name for a in agents}
        assert "generator" in names
        assert "reviewer" in names
        assert "issue_analyst" in names
        assert "fact_checker" in names
        assert "determinism_checker" in names
        assert "verifier" in names
        assert "self_critique" in names
