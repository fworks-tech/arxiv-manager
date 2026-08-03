"""Tests for scheduler/queue.py — DB-backed job queue."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, SQLModel, create_engine

from arxiv_manager.scheduler.queue import (
    cancel_job,
    complete_job,
    dequeue,
    enqueue,
    fail_job,
    get_job_status,
    list_queue,
    queue_depth,
)


@pytest.fixture
def queue_engine(tmp_path):
    """In-memory SQLite engine for queue tests."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test_queue.db'}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_session(queue_engine, monkeypatch):
    """Patch get_session to return a session bound to the test engine."""

    def _fake_session():
        return Session(queue_engine)

    import arxiv_manager.scheduler.queue as q_mod

    monkeypatch.setattr(q_mod, "get_session", _fake_session)


class TestEnqueue:
    def test_enqueue_basic(self, patch_session):
        job = enqueue("generate_qa", {"image_path": "/tmp/test.png", "difficulty": "challenging"})
        assert job.id is not None
        assert job.type == "generate_qa"
        assert job.status == "queued"
        assert job.priority == 0
        assert job.max_attempts == 3
        payload = json.loads(job.payload)
        assert payload["image_path"] == "/tmp/test.png"

    def test_enqueue_with_priority(self, patch_session):
        job = enqueue("rag_index", priority=10, max_attempts=5)
        assert job.priority == 10
        assert job.max_attempts == 5

    def test_enqueue_empty_payload(self, patch_session):
        job = enqueue("validate_batch")
        assert job.payload == "{}"


class TestDequeue:
    def test_dequeue_returns_oldest_highest_priority(self, patch_session):
        enqueue("generate_qa", priority=1)
        enqueue("generate_qa", priority=5)
        enqueue("generate_qa", priority=3)

        job = dequeue()
        assert job is not None
        assert job.priority == 5
        assert job.status == "running"

    def test_dequeue_empty_returns_none(self, patch_session):
        assert dequeue() is None

    def test_dequeue_does_not_return_running_jobs(self, patch_session):
        enqueue("generate_qa", priority=5)
        job1 = dequeue()
        assert job1 is not None
        assert dequeue() is None


class TestCompleteAndFail:
    def test_complete_job(self, patch_session):
        job = enqueue("generate_qa")
        completed = complete_job(job.id, {"question": "test", "answer": "42"})
        assert completed.status == "done"
        result = json.loads(completed.result)
        assert result["question"] == "test"

    def test_fail_job_retries(self, patch_session):
        job = enqueue("generate_qa", max_attempts=3)
        for _ in range(2):
            failed = fail_job(job.id, "transient error")
            assert failed.status == "queued"
            assert dequeue() is not None  # requeued

        failed = fail_job(job.id, "final error")
        assert failed.status == "failed"
        assert dequeue() is None  # not requeued

    def test_complete_nonexistent(self, patch_session):
        assert complete_job(9999, {}) is None

    def test_fail_nonexistent(self, patch_session):
        assert fail_job(9999, "error") is None


class TestCancel:
    def test_cancel_queued(self, patch_session):
        job = enqueue("generate_qa")
        cancelled = cancel_job(job.id)
        assert cancelled.status == "cancelled"

    def test_cancel_does_not_affect_done(self, patch_session):
        job = enqueue("generate_qa")
        complete_job(job.id)
        cancelled = cancel_job(job.id)
        assert cancelled.status == "done"

    def test_cancel_returns_none_for_missing(self, patch_session):
        assert cancel_job(9999) is None


class TestQuery:
    def test_get_job_status(self, patch_session):
        job = enqueue("generate_qa", priority=3)
        status = get_job_status(job.id)
        assert status["type"] == "generate_qa"
        assert status["status"] == "queued"
        assert status["priority"] == 3

    def test_get_job_status_missing(self, patch_session):
        assert get_job_status(9999) is None

    def test_list_queue_orders_by_newest(self, patch_session):
        import time

        j1 = enqueue("type_a")
        time.sleep(0.01)
        j2 = enqueue("type_b")
        jobs = list_queue(limit=10)
        assert jobs[0]["id"] == j2.id
        assert jobs[1]["id"] == j1.id

    def test_queue_depth(self, patch_session):
        enqueue("generate_qa")
        enqueue("generate_qa")
        assert queue_depth() == 2

    def test_queue_depth_running_not_counted(self, patch_session):
        enqueue("generate_qa")
        job = dequeue()
        assert job is not None
        assert queue_depth() == 0


class TestStaleJobRequeue:
    def test_requeue_stale_running_jobs(self, patch_session, queue_engine, monkeypatch):
        """Worker startup resets stale 'running' jobs back to queued."""
        from arxiv_manager.scheduler.worker import _requeue_stale_running_jobs

        # _requeue_stale_running_jobs imports get_session from ..db directly
        monkeypatch.setattr(
            "arxiv_manager.db.get_session",
            lambda: Session(queue_engine),
        )

        job = enqueue("generate_qa")
        claimed = dequeue()
        assert claimed is not None
        assert claimed.status == "running"
        assert queue_depth() == 0

        _requeue_stale_running_jobs()

        status = get_job_status(job.id)
        assert status["status"] == "queued"
        assert dequeue() is not None  # claimable again


class TestWorkerPidFile:
    def test_pid_file_write_and_remove(self, tmp_path, monkeypatch):
        """Worker writes its PID for orphan detection and removes it on exit."""
        import arxiv_manager.scheduler.worker as w_mod
        from arxiv_manager import storage as st_mod

        monkeypatch.setattr(st_mod, "STORAGE_DIR", tmp_path)

        w_mod._write_pid_file()
        pid_file = tmp_path / w_mod._PID_FILE_NAME
        assert pid_file.exists()
        assert int(pid_file.read_text().strip()) > 0

        w_mod._remove_pid_file()
        assert not pid_file.exists()

    def test_orphan_detection_via_pid_file(self, tmp_path, monkeypatch):
        """manager._orphan_worker_pid returns a live PID from the pid file."""
        import arxiv_manager.scheduler.manager as m_mod
        from arxiv_manager import storage as st_mod

        monkeypatch.setattr(st_mod, "STORAGE_DIR", tmp_path)

        # PID 1 never exists as a real process we can probe, so the manager
        # must treat a stale pid file as no-orphan on failure.
        assert m_mod._orphan_worker_pid() is None

        (tmp_path / "_scheduler_worker.pid").write_text("999999999")
        assert m_mod._orphan_worker_pid() is None

        (tmp_path / "_scheduler_worker.pid").write_text("0")
        assert m_mod._orphan_worker_pid() is None

        (tmp_path / "_scheduler_worker.pid").write_text("not-a-pid")
        assert m_mod._orphan_worker_pid() is None
