"""Tests for the GenerationAttempt model and log_generation_attempt function."""

import pytest

from arxiv_manager.authoring._draft_telemetry import log_generation_attempt
from arxiv_manager.models import GenerationAttempt


@pytest.fixture(autouse=True)
def _patch_get_session(monkeypatch, db_session):
    """Make get_session() return the test DB session."""
    monkeypatch.setattr("arxiv_manager.db.get_session", lambda: db_session)


class TestGenerationAttemptModel:
    def test_defaults(self):
        """GenerationAttempt() with minimum fields has sensible defaults."""
        a = GenerationAttempt(figure_id=1)
        assert a.figure_id == 1
        assert a.task_id is None
        assert a.parent_attempt_id is None
        assert a.attempt_number == 0
        assert a.generation_type == ""
        assert a.prompt_text_hash == ""
        assert a.prompt_version_id == ""
        assert a.success is False
        assert a.validation_quality == 0.0
        assert a.critique_score == 0
        assert a.elapsed_ms == 0

    def test_custom_values(self):
        """GenerationAttempt stores all custom values."""
        a = GenerationAttempt(
            figure_id=1,
            task_id=2,
            parent_attempt_id=3,
            attempt_number=2,
            generation_type="critique",
            source_route="api_draft_qa",
            prompt_template_name="CHALLENGING_PROMPT",
            prompt_text_hash="a1b2c3d4e5f6a1b2c3d4",
            prompt_version_id="CHALLENGING_PROMPT@a1b2c3d4e5f6",
            difficulty="challenging",
            figure_type="chart_graph_text",
            complexity_score=0.75,
            model_name="minimax-m3",
            max_tokens=16000,
            timeout_s=240,
            raw_response='{"question":"Q?","answer":"42"}',
            reasoning_trace="I need to count the bars...",
            generated_question="Test question?",
            generated_answer="42",
            generated_answer_format="number",
            generated_task_type="chart",
            validation_quality=85.0,
            validation_is_valid=True,
            validation_errors='["Answer too short"]',
            critique_score=4,
            success=True,
            elapsed_ms=18234,
        )
        assert a.figure_id == 1
        assert a.task_id == 2
        assert a.attempt_number == 2
        assert a.generation_type == "critique"
        assert a.prompt_text_hash == "a1b2c3d4e5f6a1b2c3d4"
        assert a.prompt_version_id == "CHALLENGING_PROMPT@a1b2c3d4e5f6"
        assert a.success is True
        assert a.validation_quality == 85.0
        assert a.critique_score == 4
        assert a.elapsed_ms == 18234

    def test_nullable_fields(self):
        """figure_id, task_id, parent_attempt_id are nullable."""
        a = GenerationAttempt()
        assert a.figure_id is None
        assert a.task_id is None
        assert a.parent_attempt_id is None


class TestLogGenerationAttempt:
    def test_writes_to_db(self, db_session, sample_figure):
        figure_id = sample_figure.id
        log_generation_attempt(
            figure_id=figure_id,
            attempt_number=1,
            generation_type="draft",
            source_route="test",
            prompt_template_name="TEST_PROMPT",
            prompt_text_hash="abc123def456abc12345",
            prompt_version_id="TEST_PROMPT@abc123def456",
            difficulty="challenging",
            figure_type="chart_graph_text",
            generated_question="Test?",
            generated_answer="42",
            success=True,
        )
        records = list(db_session.exec(__import__("sqlmodel").select(GenerationAttempt)).all())
        assert len(records) == 1
        r = records[0]
        assert r.figure_id == figure_id
        assert r.generated_question == "Test?"
        assert r.prompt_text_hash == "abc123def456abc12345"
        assert r.prompt_version_id == "TEST_PROMPT@abc123def456"

    def test_skips_when_no_figure_or_task(self):
        """log_generation_attempt skips when both figure_id and task_id are None."""
        log_generation_attempt(
            attempt_number=1,
            generation_type="draft",
        )
        # No exception should be raised

    def test_handles_nullable_fields(self, db_session, sample_figure):
        """All optional fields can be omitted."""
        log_generation_attempt(
            figure_id=sample_figure.id,
            attempt_number=1,
        )
        records = list(db_session.exec(__import__("sqlmodel").select(GenerationAttempt)).all())
        assert len(records) == 1
        r = records[0]
        assert r.generated_question == ""
        assert r.reasoning_trace == ""
        assert r.critique_score == 0
