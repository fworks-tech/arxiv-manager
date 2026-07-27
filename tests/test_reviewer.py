"""Tests for agents/reviewer.py — draft review and critique."""

from __future__ import annotations

from arxiv_manager.agents.context import new_context
from arxiv_manager.agents.reviewer import review_draft


class TestReviewDraft:

    def test_review_empty_draft(self):
        result = review_draft({"question": "", "answer": ""})
        assert result["score"] == 1
        assert result["passed"] is False
        assert "empty" in result["suggestions"][0].lower()

    def test_review_high_quality(self):
        draft = {
            "question": "What is the maximum value shown in the bar chart?",
            "answer": "42",
            "answer_format": "number",
            "_validation_quality": 0.95,
        }
        result = review_draft(draft)
        assert result["score"] >= 4
        assert result["passed"] is True
        assert "High validation quality" in result["strengths"]

    def test_review_medium_quality(self):
        draft = {
            "question": "What color is the bar?",
            "answer": "blue",
            "answer_format": "word",
            "_validation_quality": 0.55,
        }
        result = review_draft(draft)
        assert result["score"] == 3
        assert result["passed"] is True

    def test_review_low_quality(self):
        draft = {
            "question": "What is x?",
            "answer": "42",
            "answer_format": "number",
            "_validation_quality": 0.3,
        }
        result = review_draft(draft)
        assert result["score"] == 2
        assert result["passed"] is False
        assert any("regenerating" in s.lower() for s in result["suggestions"])

    def test_review_very_short_answer(self):
        draft = {
            "question": "What is x?",
            "answer": "X",
            "answer_format": "word",
            "_validation_quality": 0.8,
        }
        result = review_draft(draft)
        assert result["score"] <= 4
        assert any("short" in s.lower() for s in result["suggestions"])

    def test_review_answer_in_question(self):
        draft = {
            "question": "Is the answer 42?",
            "answer": "42",
            "answer_format": "number",
            "_validation_quality": 0.8,
        }
        result = review_draft(draft)
        assert any("contain the answer" in s.lower() for s in result["suggestions"])

    def test_review_format_mismatch_number(self):
        draft = {
            "question": "What color is it?",
            "answer": "blue",
            "answer_format": "number",
            "_validation_quality": 0.8,
        }
        result = review_draft(draft)
        assert any(s for s in result["suggestions"] if "number" in s.lower())

    def test_review_format_mismatch_year(self):
        draft = {
            "question": "What year?",
            "answer": "not-a-year",
            "answer_format": "year",
            "_validation_quality": 0.8,
        }
        result = review_draft(draft)
        assert any(s for s in result["suggestions"] if "year" in s.lower())

    def test_review_with_context(self):
        ctx = new_context(1, "challenging", "chart_graph_text")
        draft = {
            "question": "What is the ratio?",
            "answer": "3.5",
            "answer_format": "number",
            "_validation_quality": 0.85,
        }
        result = review_draft(draft, ctx)
        assert result["agent"] == "reviewer"
        assert result["score"] >= 4

    def test_review_perfect_passes(self):
        draft = {
            "question": "What is the maximum temperature shown?",
            "answer": "37.2",
            "answer_format": "number",
            "_validation_quality": 0.95,
        }
        result = review_draft(draft)
        assert result["passed"] is True
        assert result["score"] == 5
