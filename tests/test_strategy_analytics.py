"""Tests for strategy classification and verdict aggregation."""

import pytest

import arxiv_manager.models  # noqa: F401  (register tables in SQLModel.metadata before db fixtures)
from arxiv_manager.analytics.strategies import (
    build_strategy_analytics,
    classify_strategy,
    task_verdict_sources,
)


class TestClassifyStrategy:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("How many red boxes are in panel 1?", "counting"),
            ("Count the rows in panels (a) and (b).", "counting"),
            ("What is the sum of the two confidence values?", "cross_panel_sum_diff"),
            ("What is the difference between panel A and panel B?", "cross_panel_sum_diff"),
            ("Which panel has the larger value?", "comparison"),
            ("How many positions does it drop?", "rank"),
            ("What is the percentage point increase?", "percentage_change"),
            ("What element is above the header in panel 2?", "spatial"),
            ("What is the label of the y-axis?", "single_lookup"),
            ("Trace the flow from start to end.", "other"),
            ("", "other"),
        ],
    )
    def test_patterns(self, question, expected):
        assert classify_strategy(question) == expected


class TestVerdictSources:
    def test_manual_passes_are_sources(self, db_session, sample_task):
        sample_task.qwen_passes = 2
        sample_task.gemini_passes = 0
        sample_task.total_runs = 4
        db_session.add(sample_task)
        db_session.commit()

        sources = task_verdict_sources(db_session, sample_task)
        verdicts = [s["verdict"] for s in sources]
        assert "pass" in verdicts  # qwen pass present
        sources_qwen = [s for s in sources if s["source"] == "qwen"]
        assert sources_qwen and sources_qwen[0]["detail"] == "2/4"


class TestBuildStrategyAnalytics:
    def test_empty_db(self, db_session):
        analytics = build_strategy_analytics(session=db_session)
        assert analytics["total_tasks"] == 0
        assert analytics["order"] == []

    def test_aggregates_by_strategy(self, db_session, sample_figure):
        from arxiv_manager.models import Task

        db_session.add(
            Task(
                title="T1",
                figure_id=sample_figure.id,
                question="How many boxes are in panel 1?",
                answer="3",
                answer_format="number",
                task_type="chart",
                difficulty="hardest",
                status="draft",
                image_path="figures/test_figure.png",
                qwen_passes=0,
                gemini_passes=0,
            )
        )
        db_session.add(
            Task(
                title="T2",
                figure_id=sample_figure.id,
                question="Which panel has the larger value?",
                answer="B",
                answer_format="word",
                task_type="chart",
                difficulty="hardest",
                status="draft",
                image_path="figures/test_figure.png",
                qwen_passes=1,
                gemini_passes=4,
            )
        )
        db_session.commit()

        analytics = build_strategy_analytics(session=db_session)
        assert analytics["total_tasks"] == 2
        assert set(analytics["order"]) == {"counting", "comparison"}
        counting = analytics["strategies"]["counting"]
        assert counting["count"] == 1
        assert counting["hardest"] == 1
        assert counting["qwen_passes_any"] == 0
        comparison = analytics["strategies"]["comparison"]
        assert comparison["qwen_passes_any"] == 1
        assert comparison["gemini_passes_any"] == 1
