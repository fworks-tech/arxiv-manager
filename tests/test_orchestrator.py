"""Tests for agents/orchestrator.py — event-driven pipeline."""

from __future__ import annotations

from arxiv_manager.agents.context import new_context
from arxiv_manager.agents.events import EventBus, PipelineEvent
from arxiv_manager.agents.orchestrator import _subscribe_agents


class TestSubscribeAgents:
    def test_all_agents_subscribed(self):
        bus = EventBus()
        _subscribe_agents(bus)

        # Each agent subscribes to at least one event
        assert bus.subscriber_count("issue_reported") >= 1
        assert bus.subscriber_count("regeneration_requested") >= 1
        assert bus.subscriber_count("draft_generated") >= 1
        assert bus.subscriber_count("draft_validated") >= 1
        assert bus.subscriber_count("fact_checked") >= 1
        assert bus.subscriber_count("determinism_checked") >= 1
        assert bus.subscriber_count("answer_verified") >= 1


class TestPipelineEndToEnd:
    def test_easy_pipeline_skips_heavy_gates(self):
        """Easy tasks skip self-critique, fact-check, determinism."""
        from arxiv_manager.agents.determinism import DeterminismCheckerAgent
        from arxiv_manager.agents.fact_checker import FactCheckerAgent
        from arxiv_manager.agents.self_critique import SelfCritiqueAgent

        bus = EventBus()
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.set_artifact("image_path", "/tmp/fake.png")

        # Track which events fire
        events_fired = []

        def tracker(event: PipelineEvent):
            events_fired.append(event.event_type)
            return []

        for evt_type in ["draft_generated", "draft_validated", "fact_checked",
                         "determinism_checked", "answer_verified", "review_completed",
                         "pipeline_completed"]:
            bus.subscribe(evt_type, tracker)

        # Manually drive: generator → self_critique (skips) → fact_checker (skips)
        # → determinism (skips) → reviewer → pipeline_completed
        sc = SelfCritiqueAgent()
        fc = FactCheckerAgent()
        det = DeterminismCheckerAgent()

        # We can't call generator without a real image, so test the skip agents directly
        sc_event = PipelineEvent(event_type="draft_generated", context=ctx)
        sc_results = sc.process(sc_event)
        assert sc_results[0].metadata.get("skipped") is True

        fc_event = PipelineEvent(event_type="draft_validated", context=ctx)
        fc_results = fc.process(fc_event)
        assert fc_results[0].metadata.get("skipped") is True

        det_event = PipelineEvent(event_type="fact_checked", context=ctx)
        det_results = det.process(det_event)
        assert det_results[0].metadata.get("skipped") is True

    def test_issue_analyst_emits_regeneration(self):
        from arxiv_manager.agents.issue_analyst import IssueAnalystAgent

        analyst = IssueAnalystAgent()
        ctx = new_context(figure_id=1, difficulty="hardest", figure_type="chart")

        event = PipelineEvent(
            event_type="issue_reported",
            context=ctx,
            metadata={"issue_report": {"reason": "too_easy", "description": "Qwen 4/4"}},
        )

        results = analyst.process(event)
        assert len(results) == 1
        assert results[0].event_type == "regeneration_requested"
        assert "too_easy" in ctx.get_artifact("issue_hints")

    def test_reviewer_marks_pipeline_completed(self):
        from arxiv_manager.agents.reviewer import ReviewerAgent

        rev = ReviewerAgent()
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.set_artifact("draft", {"question": "Q?", "answer": "A", "_validation_quality": 0.9})

        event = PipelineEvent(event_type="answer_verified", context=ctx)
        results = rev.process(event)

        assert len(results) == 1
        assert results[0].event_type == "pipeline_completed"
        assert ctx.pipeline_status == "completed"
