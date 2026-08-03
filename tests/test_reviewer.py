"""Tests for agents/reviewer.py — draft review and critique."""

from __future__ import annotations

from unittest.mock import patch

from arxiv_manager.agents.context import new_context
from arxiv_manager.agents.events import PipelineEvent
from arxiv_manager.agents.reviewer import ReviewerAgent


class TestReviewerAgent:
    def test_review_empty_draft(self):
        rev = ReviewerAgent()
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.set_artifact("draft", {"question": "", "answer": ""})

        event = PipelineEvent(event_type="answer_verified", context=ctx)
        results = rev.process(event)
        assert results[0].metadata["score"] == 1
        assert results[0].metadata["passed"] is False

    @patch("arxiv_manager.agents.reviewer.os.environ", {"OPENCODE_API_KEY": ""})
    def test_review_high_quality_draft(self):
        rev = ReviewerAgent()
        ctx = new_context(figure_id=1, difficulty="challenging", figure_type="chart")
        ctx.set_artifact("draft", {
            "question": "What is the peak value in panel A?",
            "answer": "42",
            "_validation_quality": 0.95,
        })

        event = PipelineEvent(event_type="answer_verified", context=ctx)
        results = rev.process(event)

        assert results[0].event_type == "pipeline_completed"
        review = ctx.get_artifact("review")
        assert review["score"] >= 4
        assert review["passed"] is True

    @patch("arxiv_manager.agents.reviewer.os.environ", {"OPENCODE_API_KEY": ""})
    def test_review_low_quality_draft(self):
        rev = ReviewerAgent()
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.set_artifact("draft", {
            "question": "What is X?",
            "answer": "A",
            "_validation_quality": 0.3,
        })

        event = PipelineEvent(event_type="answer_verified", context=ctx)
        rev.process(event)

        review = ctx.get_artifact("review")
        assert review["score"] <= 3

    @patch("arxiv_manager.agents.reviewer.os.environ", {"OPENCODE_API_KEY": ""})
    def test_review_answer_in_question(self):
        rev = ReviewerAgent()
        ctx = new_context(figure_id=1, difficulty="easy", figure_type="chart")
        ctx.set_artifact("draft", {
            "question": "The answer is 42. What is the answer?",
            "answer": "42",
            "_validation_quality": 0.8,
        })

        event = PipelineEvent(event_type="answer_verified", context=ctx)
        rev.process(event)

        review = ctx.get_artifact("review")
        assert any("contains the answer" in s for s in review["suggestions"])
