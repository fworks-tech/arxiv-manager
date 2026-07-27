"""Tests for scheduler/queue.py — DB-backed job queue."""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, SQLModel, create_engine

from arxiv_manager.scheduler.models import ScheduledTask
from arxiv_manager.scheduler.queue import (
    enqueue,
    dequeue,
    complete_job,
    fail_job,
    cancel_job,
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
