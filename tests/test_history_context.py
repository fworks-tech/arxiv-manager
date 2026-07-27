"""Tests for history context and few-shot example retrieval."""

import pytest

from arxiv_manager.authoring._history_context import (
    build_figure_history,
    get_few_shot_examples,
    inject_history_into_prompt,
    select_best_model,
)
from arxiv_manager.models import GenerationAttempt


@pytest.fixture(autouse=True)
def _patch_get_session(monkeypatch, db_session):
    """Make get_session() return the test DB session."""
    monkeypatch.setattr("arxiv_manager.db.get_session", lambda: db_session)


class TestSelectBestModel:
    def test_no_data_returns_default(self, db_session, sample_figure):
        result = select_best_model(
            figure_type="chart_graph_text",
            difficulty="challenging",
            default_model="minimax-m3",
            min_attempts=3,
        )
        assert result == "minimax-m3"

    def test_insufficient_data_returns_default(self, db_session, sample_figure):
        db_session.add(GenerationAttempt(
            figure_id=sample_figure.id,
            model_name="kimi-k2.7-code",
            validation_quality=95,
            success=True,
            validation_is_valid=True,
            figure_type="chart_graph_text",
            difficulty="challenging",
            generated_question="Q?",
        ))
        db_session.commit()
        result = select_best_model(
            figure_type="chart_graph_text",
            difficulty="challenging",
            default_model="minimax-m3",
            min_attempts=3,
        )
        assert result == "minimax-m3"

    def test_sufficient_data_picks_best(self, db_session, sample_figure):
        for model, quality, count in [("model-a", 90, 3), ("model-b", 70, 3)]:
            for _ in range(count):
                db_session.add(GenerationAttempt(
                    figure_id=sample_figure.id,
                    model_name=model,
                    validation_quality=quality,
                    success=True,
                    validation_is_valid=True,
                    figure_type="chart_graph_text",
                    difficulty="challenging",
                    generated_question=f"Q {model} {_}",
                ))
        db_session.commit()
        result = select_best_model(
            figure_type="chart_graph_text",
            difficulty="challenging",
            default_model="minimax-m3",
            min_attempts=3,
        )
        assert result == "model-a"


class TestGetFewShotExamples:
    def test_no_data_returns_empty(self, db_session):
        examples = get_few_shot_examples(figure_type="chart_graph_text", difficulty="challenging")
        assert examples == []

    def test_filters_by_figure_type(self, db_session, sample_figure):
        db_session.add(GenerationAttempt(
            figure_id=sample_figure.id,
            model_name="test",
            validation_quality=90,
            success=True,
            validation_is_valid=True,
            figure_type="chart_graph_text",
            difficulty="challenging",
            generated_question="Q1",
            generated_answer="A1",
        ))
        db_session.add(GenerationAttempt(
            figure_id=sample_figure.id,
            model_name="test",
            validation_quality=85,
            success=True,
            validation_is_valid=True,
            figure_type="general_image",
            difficulty="challenging",
            generated_question="Q2",
            generated_answer="A2",
        ))
        db_session.commit()
        examples = get_few_shot_examples(figure_type="chart_graph_text", limit=5)
        assert len(examples) == 1
        assert examples[0]["question"] == "Q1"


class TestBuildFigureHistory:
    def test_no_history_returns_empty(self, db_session):
        result = build_figure_history(figure_id=999)
        assert result == ""

    def test_with_history_returns_string(self, db_session, sample_figure):
        db_session.add(GenerationAttempt(
            figure_id=sample_figure.id,
            generation_type="draft",
            difficulty="challenging",
            generated_question="Test question?",
            generated_answer="42",
            validation_quality=85,
            success=True,
        ))
        db_session.commit()
        result = build_figure_history(sample_figure.id)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test question?" in result


class TestInjectHistoryIntoPrompt:
    def test_no_context_returns_base(self):
        result = inject_history_into_prompt("Base prompt")
        assert result == "Base prompt"

    def test_appends_previous_question(self):
        result = inject_history_into_prompt(
            "Base prompt",
            previous_question="What was asked before?",
        )
        assert "Base prompt" in result
        assert "What was asked before?" in result
        assert "SUBSTANTIALLY DIFFERENT" in result

    def test_still_carries_base(self):
        result = inject_history_into_prompt(
            "Original prompt",
            figure_type="chart_graph_text",
            difficulty="challenging",
        )
        assert "Original prompt" in result
