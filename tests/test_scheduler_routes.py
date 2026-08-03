"""Tests for scheduler/routes.py — HTTP API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def scheduler_client(monkeypatch, tmp_path):
    """TestClient with mocked scheduler queue."""
    from fastapi import FastAPI

    # Use file-based DB for queue persistence across requests
    db_path = tmp_path / "scheduler_test.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(engine)

    def _fake_session():
        return Session(engine)

    import arxiv_manager.scheduler.queue as q_mod

    monkeypatch.setattr(q_mod, "get_session", _fake_session)

    from arxiv_manager.scheduler.routes import router

    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        yield client


class TestEnqueueEndpoint:
    def test_enqueue(self, scheduler_client):
        resp = scheduler_client.post(
            "/api/scheduler/enqueue",
            json={
                "type": "generate_qa",
                "payload": {"image_path": "/tmp/test.png"},
                "priority": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_enqueue_missing_type(self, scheduler_client):
        resp = scheduler_client.post("/api/scheduler/enqueue", json={})
        assert resp.status_code == 400
        assert "missing" in resp.json()["error"].lower()

    def test_enqueue_defaults(self, scheduler_client):
        resp = scheduler_client.post(
            "/api/scheduler/enqueue",
            json={
                "type": "validate_batch",
            },
        )
        assert resp.status_code == 200


class TestJobStatusEndpoint:
    def test_get_status(self, scheduler_client):
        resp = scheduler_client.post(
            "/api/scheduler/enqueue",
            json={
                "type": "generate_qa",
            },
        )
        job_id = resp.json()["job_id"]

        resp = scheduler_client.get(f"/api/scheduler/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "generate_qa"
        assert data["status"] == "queued"

    def test_get_status_not_found(self, scheduler_client):
        resp = scheduler_client.get("/api/scheduler/status/9999")
        assert resp.status_code == 404

    def test_status_after_completion(self, scheduler_client):

        import arxiv_manager.scheduler.queue as q_mod

        resp = scheduler_client.post(
            "/api/scheduler/enqueue",
            json={
                "type": "generate_qa",
            },
        )
        job_id = resp.json()["job_id"]

        # Simulate completion via queue module
        q_mod.complete_job(job_id, {"question": "test"})

        resp = scheduler_client.get(f"/api/scheduler/status/{job_id}")
        assert resp.json()["status"] == "done"


class TestCancelEndpoint:
    def test_cancel(self, scheduler_client):
        resp = scheduler_client.post(
            "/api/scheduler/enqueue",
            json={
                "type": "generate_qa",
            },
        )
        job_id = resp.json()["job_id"]

        resp = scheduler_client.post(f"/api/scheduler/cancel/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_not_found(self, scheduler_client):
        resp = scheduler_client.post("/api/scheduler/cancel/9999")
        assert resp.status_code == 404


class TestQueueEndpoint:
    def test_queue_list(self, scheduler_client):
        scheduler_client.post("/api/scheduler/enqueue", json={"type": "rag_index"})
        scheduler_client.post("/api/scheduler/enqueue", json={"type": "generate_qa"})

        resp = scheduler_client.get("/api/scheduler/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["jobs"]) == 2

    def test_queue_empty(self, scheduler_client):
        resp = scheduler_client.get("/api/scheduler/queue")
        assert len(resp.json()["jobs"]) == 0

    def test_worker_status(self, scheduler_client):
        resp = scheduler_client.get("/api/scheduler/worker")
        assert resp.status_code == 200
        data = resp.json()
        assert "alive" in data
        assert "workers" in data
        assert "count" in data
