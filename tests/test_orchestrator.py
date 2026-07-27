"""Tests for agents/orchestrator.py — orchestration workflow."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from arxiv_manager.agents.context import new_context
from arxiv_manager.agents.orchestrator import orchestrate

# All patches target source modules (not the lazy-importing orchestrator)
DRAFT_PATH = "arxiv_manager.authoring.ai_draft.core.draft_qa"
REVIEW_PATH = "arxiv_manager.agents.reviewer.review_draft"
DECOMP_PATH = "arxiv_manager.agents.query_decomposer.decompose_query"


class TestOrchestrate:

    def test_orchestrate_basic(self):
        ctx = new_context(1, "challenging", "chart_graph_text")

        with patch(DRAFT_PATH) as mock_draft:
            mock_draft.return_value = {
                "question": "What is the peak value?",
                "answer": "42",
                "_validation_quality": 0.85,
            }
            with patch(REVIEW_PATH) as mock_review:
                mock_review.return_value = {
                    "score": 4,
                    "passed": True,
                    "suggestions": [],
                    "strengths": ["High quality"],
                    "agent": "reviewer",
                }

                result = orchestrate(ctx, "Generate a question", "/tmp/test.png")

        assert result["question"] == "What is the peak value?"
        assert result["answer"] == "42"

    def test_orchestrate_fallback_on_failure(self):
        ctx = new_context(1, "challenging", "chart_graph_text")

        with patch(DRAFT_PATH) as mock_draft:
            mock_draft.return_value = None
            mock_fallback = patch(
                "arxiv_manager.agents.orchestrator._fallback_generate",
                return_value={},
            )
            with mock_fallback:
                result = orchestrate(ctx, "test", "/tmp/test.png")
        assert result == {}

    def test_orchestrate_delegates_review_for_hardest(self):
        ctx = new_context(1, "hardest", "chart_graph_text")

        with patch(DRAFT_PATH) as mock_draft:
            mock_draft.return_value = {
                "question": "Q?",
                "answer": "A",
                "_validation_quality": 0.6,
            }
            with patch(REVIEW_PATH) as mock_review:
                mock_review.return_value = {
                    "score": 2,
                    "passed": False,
                    "suggestions": ["Low quality"],
                    "strengths": [],
                    "agent": "reviewer",
                }
                result = orchestrate(ctx, "test", "/tmp/test.png")

        assert result["question"] == "Q?"
        mock_review.assert_called_once()

    def test_orchestrate_skips_review_for_easy(self):
        ctx = new_context(1, "easy", "chart_graph_text")

        with patch(DRAFT_PATH) as mock_draft:
            mock_draft.return_value = {
                "question": "Q?",
                "answer": "A",
                "_validation_quality": 0.8,
            }
            with patch(REVIEW_PATH) as mock_review:
                result = orchestrate(ctx, "test", "/tmp/test.png")

        assert result["question"] == "Q?"
        mock_review.assert_not_called()

    def test_orchestrate_sets_artifacts(self):
        ctx = new_context(1, "challenging", "chart_graph_text")

        with patch(DRAFT_PATH) as mock_draft:
            mock_draft.return_value = {
                "question": "Q?",
                "answer": "A",
                "_validation_quality": 0.9,
            }
            with patch(REVIEW_PATH) as mock_review:
                mock_review.return_value = {
                    "score": 5, "passed": True,
                    "suggestions": [], "strengths": [],
                    "agent": "reviewer",
                }
                orchestrate(ctx, "test", "/tmp/test.png")

        assert ctx.get_artifact("prompt") == "test"
        assert ctx.get_artifact("image_path") == "/tmp/test.png"
        assert ctx.get_artifact("review") is not None


class TestOrchestratePlan:

    def test_plan_subtasks_called_for_hardest(self):
        ctx = new_context(1, "hardest", "chart_graph_text")

        with patch(DECOMP_PATH) as mock_decomp:
            mock_decomp.return_value = [
                {"description": "Step 1", "order": 0},
                {"description": "Step 2", "order": 1},
            ]
            with patch(DRAFT_PATH) as mock_draft:
                mock_draft.return_value = {
                    "question": "Q?", "answer": "A", "_validation_quality": 0.8,
                }
                with patch(REVIEW_PATH) as mock_r:
                    mock_r.return_value = {
                        "score": 4, "passed": True,
                        "suggestions": [], "strengths": [],
                        "agent": "reviewer",
                    }
                    orchestrate(ctx, "test", "/tmp/test.png")

        mock_decomp.assert_called_once_with(
            difficulty="hardest",
            figure_type="chart_graph_text",
            prompt="test",
        )
