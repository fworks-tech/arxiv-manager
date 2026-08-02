"""Tests for Realm verdict ingestion and difficulty auto-adjustment."""

import pytest
from sqlmodel import Session, select

import arxiv_manager.models  # noqa: F401  (register tables in SQLModel.metadata before db fixtures)
from arxiv_manager.tracking import mark_submitted, record_realm_verdict


@pytest.fixture(autouse=True)
def _patch_get_session(monkeypatch, db_engine):
    """Each get_session() call opens a fresh session on the test DB (like production)."""

    def _new_session():
        return Session(db_engine)

    monkeypatch.setattr("arxiv_manager.db.get_session", _new_session)
    monkeypatch.setattr("arxiv_manager.tracking.get_session", _new_session)


def _task_row(db_engine, task_id):
    with Session(db_engine) as s:
        return s.get(arxiv_manager.models.Task, task_id)


class TestRecordRealmVerdict:
    def test_invalid_verdict_raises(self, db_session, sample_task):
        with pytest.raises(ValueError):
            record_realm_verdict(sample_task.id, "not-a-verdict")

    def test_too_easy_downgrades_hardest(self, db_session, sample_task, db_engine):
        sample_task.difficulty = "hardest"
        sample_task.status = "submitted"
        db_session.add(sample_task)
        db_session.commit()
        mark_submitted(sample_task.id)  # create a submission log row

        record_realm_verdict(sample_task.id, "too_easy", "Qwen passed 4/4")
        task = _task_row(db_engine, sample_task.id)
        assert task.difficulty == "challenging"

    def test_too_easy_floor_stays_easy(self, db_session, sample_task, db_engine):
        sample_task.difficulty = "easy"
        sample_task.status = "submitted"
        db_session.add(sample_task)
        db_session.commit()
        mark_submitted(sample_task.id)

        record_realm_verdict(sample_task.id, "too_easy")
        assert _task_row(db_engine, sample_task.id).difficulty == "easy"

    def test_too_hard_upgrades_challenging(self, db_session, sample_task, db_engine):
        sample_task.difficulty = "challenging"
        sample_task.status = "submitted"
        db_session.add(sample_task)
        db_session.commit()
        mark_submitted(sample_task.id)

        record_realm_verdict(sample_task.id, "too_hard", "Gemini also failed")
        assert _task_row(db_engine, sample_task.id).difficulty == "hardest"

    def test_too_hard_ceiling_stays_hardest(self, db_session, sample_task, db_engine):
        sample_task.difficulty = "hardest"
        sample_task.status = "submitted"
        db_session.add(sample_task)
        db_session.commit()
        mark_submitted(sample_task.id)

        record_realm_verdict(sample_task.id, "too_hard")
        assert _task_row(db_engine, sample_task.id).difficulty == "hardest"

    def test_approved_keeps_difficulty(self, db_session, sample_task, db_engine):
        sample_task.difficulty = "challenging"
        sample_task.status = "submitted"
        db_session.add(sample_task)
        db_session.commit()
        mark_submitted(sample_task.id)

        record_realm_verdict(sample_task.id, "approved")
        assert _task_row(db_engine, sample_task.id).difficulty == "challenging"

    def test_writes_submission_log(self, db_session, sample_task, db_engine):
        sample_task.status = "submitted"
        db_session.add(sample_task)
        db_session.commit()
        mark_submitted(sample_task.id)

        record_realm_verdict(sample_task.id, "too_easy", "notes here")
        with Session(db_engine) as s:
            log = s.exec(select(arxiv_manager.models.SubmissionLog).where(
                arxiv_manager.models.SubmissionLog.task_id == sample_task.id
            )).all()[-1]
        assert log.review_status == "too_easy"
        assert log.reviewer_notes == "notes here"

    def test_missing_task_returns_none(self, db_session):
        assert record_realm_verdict(99999, "too_easy") is None
