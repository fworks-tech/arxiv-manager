"""FastAPI route handlers for the scheduler API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from .manager import start_worker, stop_worker, worker_is_alive, get_worker_pid
from .queue import (
    enqueue,
    cancel_job,
    get_job_status,
    list_queue,
    queue_depth,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.post("/enqueue")
def api_enqueue(body: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a new job.

    Body:
        type: str (required) - "generate_qa" | "validate_batch" | "rag_index"
        payload: dict (optional) - kwargs for the job handler
        priority: int (optional, default 0)
        max_attempts: int (optional, default 3)
    """
    job_type = body.get("type", "")
    if not job_type:
        raise HTTPException(status_code=400, detail="type is required")

    job = enqueue(
        job_type=job_type,
        payload=body.get("payload"),
        priority=body.get("priority", 0),
        max_attempts=body.get("max_attempts", 3),
    )
    return {"job_id": job.id, "status": job.status}


@router.get("/status/{job_id}")
def api_job_status(job_id: int) -> dict[str, Any]:
    """Get the status of a job by ID."""
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.post("/cancel/{job_id}")
def api_cancel_job(job_id: int) -> dict[str, Any]:
    """Cancel a queued or running job."""
    job = cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job.id, "status": job.status}


@router.get("/queue")
def api_queue(limit: int = 50) -> dict[str, Any]:
    """List recent jobs and current queue depth."""
    return {
        "depth": queue_depth(),
        "jobs": list_queue(limit=limit),
    }


@router.get("/worker")
def api_worker_status() -> dict[str, Any]:
    """Get the worker subprocess status."""
    alive = worker_is_alive()
    return {
        "alive": alive,
        "pid": get_worker_pid() if alive else None,
    }


@router.post("/worker/start")
def api_start_worker() -> dict[str, Any]:
    """Start the worker subprocess."""
    pid = start_worker()
    return {"pid": pid, "status": "started"}


@router.post("/worker/stop")
def api_stop_worker() -> dict[str, Any]:
    """Stop the worker subprocess."""
    stopped = stop_worker()
    return {"stopped": stopped}
