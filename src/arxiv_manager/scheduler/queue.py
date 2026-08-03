"""DB-backed FIFO job queue with priority ordering."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlmodel import desc, select

from ..db import get_session
from .models import ScheduledTask

logger = logging.getLogger(__name__)


def enqueue(
    job_type: str,
    payload: dict[str, Any] | None = None,
    priority: int = 0,
    max_attempts: int = 3,
) -> ScheduledTask:
    """Enqueue a new job.

    Args:
        job_type: The type of job ("generate_qa", "validate_batch", "rag_index").
        payload: Keyword arguments for the job handler.
        priority: Higher values = more urgent (default 0).
        max_attempts: Max retry attempts before marking failed.

    Returns:
        The created ScheduledTask record.
    """
    session = get_session()
    try:
        task = ScheduledTask(
            type=job_type,
            payload=json.dumps(payload or {}),
            priority=priority,
            max_attempts=max_attempts,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        logger.info("scheduler: enqueued %s job %d (priority=%d)", job_type, task.id, priority)
        return task
    finally:
        session.close()


def dequeue(worker_id: int = 0) -> ScheduledTask | None:
    """Pick the highest-priority queued job and mark it running.

    Uses an atomic UPDATE ... WHERE status='queued' claim so concurrent
    workers never execute the same job twice. Returns the job or None if
    the queue is empty.
    """
    from sqlmodel import update

    session = get_session()
    try:
        while True:
            task = session.exec(
                select(ScheduledTask)
                .where(ScheduledTask.status == "queued")
                .order_by(desc(ScheduledTask.priority), ScheduledTask.created_at)
                .limit(1)
            ).first()

            if task is None:
                return None

            # Atomic claim: only succeeds if still queued
            result = session.exec(
                update(ScheduledTask)
                .where(ScheduledTask.id == task.id)
                .where(ScheduledTask.status == "queued")
                .values(status="running", started_at=datetime.now(), worker_id=worker_id)
            )
            session.commit()
            if result.rowcount != 1:
                session.expire_all()  # lost the race; re-scan
                continue
            session.refresh(task)
            return task
    finally:
        session.close()


def complete_job(job_id: int, result: dict[str, Any] | None = None) -> ScheduledTask | None:
    """Mark a job as done.

    Args:
        job_id: The job ID.
        result: Optional result dict (will be JSON-serialized).
    """
    session = get_session()
    try:
        task = session.get(ScheduledTask, job_id)
        if task is None:
            return None

        task.status = "done"
        task.completed_at = datetime.now()
        task.result = json.dumps(result) if result else ""
        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    finally:
        session.close()


def fail_job(job_id: int, error: str) -> ScheduledTask | None:
    """Mark a job as failed (or retry if attempts remain).

    Increments attempts. If max_attempts reached, marks 'failed'.
    Otherwise re-queues the job.
    """
    session = get_session()
    try:
        task = session.get(ScheduledTask, job_id)
        if task is None:
            return None

        task.attempts += 1
        if task.attempts >= task.max_attempts:
            task.status = "failed"
            task.completed_at = datetime.now()
            task.result = json.dumps({"error": error})
            logger.warning("scheduler: job %d failed after %d attempts", job_id, task.attempts)
        else:
            task.status = "queued"
            task.started_at = None
            logger.info("scheduler: job %d retry %d/%d", job_id, task.attempts, task.max_attempts)

        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    finally:
        session.close()


def cancel_job(job_id: int) -> ScheduledTask | None:
    """Cancel a queued job (drop from queue)."""
    session = get_session()
    try:
        task = session.get(ScheduledTask, job_id)
        if task is None or task.status == "done":
            return task

        task.status = "cancelled"
        task.completed_at = datetime.now()
        session.add(task)
        session.commit()
        session.refresh(task)
        logger.info("scheduler: cancelled job %d", job_id)
        return task
    finally:
        session.close()


def abort_job(job_id: int) -> ScheduledTask | None:
    """Abort a running job by setting status and writing an abort sentinel.

    The worker checks for this sentinel during execution and raises
    JobCancelled to stop the current LLM call.
    """
    session = get_session()
    try:
        task = session.get(ScheduledTask, job_id)
        if task is None:
            return None
        if task.status != "running":
            return task

        task.status = "cancelled"
        task.completed_at = datetime.now()
        task.result = json.dumps({"error": "aborted by user"})
        session.add(task)
        session.commit()
        session.refresh(task)

        # Write abort sentinel so the worker can check mid-execution
        from ..storage import STORAGE_DIR

        sentinel = STORAGE_DIR / f"_abort_{job_id}"
        sentinel.write_text(f"aborted at {datetime.now().isoformat()}")
        logger.info("scheduler: aborted running job %d (worker_id=%d)", job_id, task.worker_id)
        return task
    finally:
        session.close()


def check_abort_sentinel(job_id: int) -> bool:
    """Check if an abort sentinel exists for a running job."""
    from ..storage import STORAGE_DIR

    sentinel = STORAGE_DIR / f"_abort_{job_id}"
    return sentinel.exists()


def cleanup_abort_sentinel(job_id: int) -> None:
    """Remove the abort sentinel after processing."""
    from ..storage import STORAGE_DIR

    sentinel = STORAGE_DIR / f"_abort_{job_id}"
    try:
        sentinel.unlink(missing_ok=True)
    except OSError:
        pass


def queue_position(job_id: int) -> tuple[int, int]:
    """Return (position, total_queued) for a queued job.

    Position is 1-indexed (1 = next to run).
    Returns (0, 0) if the job is not queued.
    """
    session = get_session()
    try:
        task = session.get(ScheduledTask, job_id)
        if task is None or task.status != "queued":
            return (0, 0)

        ahead = session.exec(
            select(ScheduledTask)
            .where(ScheduledTask.status == "queued")
            .where(ScheduledTask.id < job_id)
        ).all()
        total = session.exec(
            select(ScheduledTask)
            .where(ScheduledTask.status == "queued")
        ).all()
        return (len(ahead) + 1, len(total))
    finally:
        session.close()


def find_job_for_task(task_id: int, statuses: list[str] | None = None) -> ScheduledTask | None:
    """Find the latest regeneration job for a task with given statuses."""
    if statuses is None:
        statuses = ["queued", "running"]

    session = get_session()
    try:
        task_id_pattern = f'%"task_id": {task_id}'
        jobs = session.exec(
            select(ScheduledTask)
            .where(ScheduledTask.type == "regenerate_task")
            .where(ScheduledTask.status.in_(statuses))
            .where(
                ScheduledTask.payload.like(task_id_pattern + ",%")
                | ScheduledTask.payload.like(task_id_pattern + "}%")
            )
            .order_by(ScheduledTask.id.desc())
            .limit(5)
        ).all()
        return jobs[0] if jobs else None
    finally:
        session.close()


def get_job_status(job_id: int) -> dict[str, Any] | None:
    """Get the current status of a job."""
    session = get_session()
    try:
        task = session.get(ScheduledTask, job_id)
        if task is None:
            return None
        return {
            "id": task.id,
            "type": task.type,
            "status": task.status,
            "priority": task.priority,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "attempts": task.attempts,
            "max_attempts": task.max_attempts,
            "result": task.result,
            "worker_id": task.worker_id,
        }
    finally:
        session.close()


def list_queue(limit: int = 50) -> list[dict[str, Any]]:
    """List recent jobs, newest first."""
    session = get_session()
    try:
        tasks = session.exec(select(ScheduledTask).order_by(desc(ScheduledTask.created_at)).limit(limit)).all()
        return [
            {
                "id": t.id,
                "type": t.type,
                "status": t.status,
                "priority": t.priority,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "attempts": t.attempts,
                "max_attempts": t.max_attempts,
                "worker_id": t.worker_id,
            }
            for t in tasks
        ]
    finally:
        session.close()


def queue_depth() -> int:
    """Return the number of queued (not yet started) jobs."""
    from sqlmodel import func

    session = get_session()
    try:
        result = session.exec(select(func.count(ScheduledTask.id)).where(ScheduledTask.status == "queued")).one()
        return result
    finally:
        session.close()
