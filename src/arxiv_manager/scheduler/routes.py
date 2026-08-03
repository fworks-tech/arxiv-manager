"""Scheduler API endpoints for the worker pool."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.post("/enqueue")
def api_enqueue(data: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a new job."""
    from .queue import enqueue

    job_type = data.get("type", "")
    if not job_type:
        return JSONResponse({"error": "missing 'type'"}, status_code=400)

    job = enqueue(
        job_type=job_type,
        payload=data.get("payload"),
        priority=data.get("priority", 0),
    )
    return {"ok": True, "job_id": job.id}


@router.get("/status/{job_id}")
def api_job_status(job_id: int) -> dict[str, Any]:
    """Get the status of a job."""
    from .queue import get_job_status

    status = get_job_status(job_id)
    if status is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return {"ok": True, **status}


@router.post("/cancel/{job_id}")
def api_cancel_job(job_id: int) -> dict[str, Any]:
    """Cancel a queued job (drop from queue)."""
    from .queue import cancel_job

    job = cancel_job(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return {"ok": True, "status": job.status}


@router.post("/abort/{job_id}")
def api_abort_job(job_id: int) -> dict[str, Any]:
    """Abort a running job (worker checks abort sentinel mid-execution)."""
    from .queue import abort_job

    job = abort_job(job_id)
    if job is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return {"ok": True, "status": job.status}


@router.get("/queue")
def api_list_queue(limit: int = Query(50)) -> dict[str, Any]:
    """List recent jobs."""
    from .queue import list_queue

    return {"ok": True, "jobs": list_queue(limit)}


@router.get("/depth")
def api_queue_depth() -> dict[str, Any]:
    """Return the number of queued jobs."""
    from .queue import queue_depth

    return {"ok": True, "depth": queue_depth()}


@router.get("/worker")
def api_worker_status() -> dict[str, Any]:
    """Return the status of the worker pool."""
    from .manager import worker_pool_status

    workers = worker_pool_status()
    return {
        "ok": True,
        "alive": any(w.get("alive") for w in workers),
        "workers": workers,
        "count": len([w for w in workers if w.get("alive")]),
    }


@router.get("/pool")
def api_pool_status() -> dict[str, Any]:
    """Return the worker pool status (alias for /worker)."""
    from .manager import worker_pool_status

    workers = worker_pool_status()
    return {
        "ok": True,
        "workers": workers,
        "count": len([w for w in workers if w.get("alive")]),
    }


@router.post("/pool/start")
def api_start_pool(count: int = Query(5)) -> dict[str, Any]:
    """Start the worker pool with N workers."""
    from .manager import start_worker_pool

    count = max(1, min(10, count))
    pids = start_worker_pool(count)
    return {"ok": True, "workers": len(pids), "pids": pids}


@router.post("/pool/stop")
def api_stop_pool() -> dict[str, Any]:
    """Stop all workers in the pool."""
    from .manager import stop_worker_pool

    stopped = stop_worker_pool()
    return {"ok": True, "stopped": stopped}


@router.get("/worker-log")
def api_worker_log(lines: int = Query(50)) -> dict[str, Any]:
    """Return the last N lines from the worker log file."""
    from ..storage import STORAGE_DIR

    log_path = STORAGE_DIR / "_scheduler_worker.log"
    if not log_path.exists():
        return {"ok": True, "lines": []}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        recent = all_lines[-lines:]
        return {"ok": True, "lines": [line.rstrip("\n") for line in recent]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
