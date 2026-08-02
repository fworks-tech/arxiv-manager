"""Tests for the scheduler worker's regenerate_task job handler."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def worker_db(tmp_path, monkeypatch):
    """File-based DB so queue + worker handlers share state."""
    from arxiv_manager.scheduler import models as _sched_models  # noqa: F401

    db_path = tmp_path / "worker_test.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(engine)

    def _fake_session():
        return Session(engine)

    import arxiv_manager.scheduler.queue as q_mod

    monkeypatch.setattr(q_mod, "get_session", _fake_session)
    return q_mod


class TestRegenerateTaskJob:
    def test_regenerate_job_success(self, worker_db, monkeypatch):
        """Worker completes the job with the regeneration result."""
        from arxiv_manager.scheduler.worker import _execute_regenerate_task

        calls: dict = {}

        def _fake_run(task_id, difficulty, source_route):
            calls["task_id"] = task_id
            calls["difficulty"] = difficulty
            calls["source_route"] = source_route
            return {"ok": True, "question": "Q?", "answer": "42", "answer_format": "number", "task_type": "chart"}

        # The worker lazily does `from ..web.routes.task_routes import run_regeneration`
        # at job-execution time, so patching the module attribute is sufficient.
        import arxiv_manager.web.routes.task_routes as tr_mod

        monkeypatch.setattr(tr_mod, "run_regeneration", _fake_run)

        job = worker_db.enqueue("regenerate_task", {"task_id": 7, "difficulty": "hardest"}, max_attempts=1)
        _execute_regenerate_task(job.id, {"task_id": 7, "difficulty": "hardest"})

        status = worker_db.get_job_status(job.id)
        assert status["status"] == "done"
        result = json.loads(status["result"])
        assert result["ok"] is True
        assert result["answer"] == "42"
        assert calls == {"task_id": 7, "difficulty": "hardest", "source_route": "scheduler_worker"}

    def test_regenerate_job_failure(self, worker_db, monkeypatch):
        """Worker marks the job failed when the pipeline returns an error."""
        from arxiv_manager.scheduler.worker import _execute_regenerate_task

        def _fake_run(task_id, difficulty, source_route):
            return {"ok": False, "error": "Determinism check failed"}

        import arxiv_manager.web.routes.task_routes as tr_mod

        monkeypatch.setattr(tr_mod, "run_regeneration", _fake_run)

        job = worker_db.enqueue("regenerate_task", {"task_id": 7, "difficulty": "hardest"}, max_attempts=1)
        _execute_regenerate_task(job.id, {"task_id": 7, "difficulty": "hardest"})

        status = worker_db.get_job_status(job.id)
        assert status["status"] == "failed"
        result = json.loads(status["result"])
        assert "Determinism check failed" in result["error"]

    def test_regenerate_job_missing_task_id(self, worker_db):
        """Worker fails the job when the payload lacks task_id."""
        from arxiv_manager.scheduler.worker import _execute_regenerate_task

        job = worker_db.enqueue("regenerate_task", {"difficulty": "hardest"}, max_attempts=1)
        _execute_regenerate_task(job.id, {"difficulty": "hardest"})

        status = worker_db.get_job_status(job.id)
        assert status["status"] == "failed"
        assert "missing task_id" in json.loads(status["result"])["error"]
