"""DB-backed task scheduler with subprocess worker pool.

Provides an async job queue using the existing SQLite database.
Jobs are enqueued via API, picked up by a pool of subprocess workers,
and their status can be polled via API.

No external dependencies (no Redis, no Celery).
"""

from __future__ import annotations

from .manager import start_worker, start_worker_pool, stop_worker_pool, worker_pool_status
from .models import ScheduledTask
from .queue import (
    abort_job,
    cancel_job,
    check_abort_sentinel,
    cleanup_abort_sentinel,
    dequeue,
    enqueue,
    find_job_for_task,
    get_job_status,
    list_queue,
    queue_position,
)

__all__ = [
    "ScheduledTask",
    "abort_job",
    "cancel_job",
    "check_abort_sentinel",
    "cleanup_abort_sentinel",
    "dequeue",
    "enqueue",
    "find_job_for_task",
    "get_job_status",
    "list_queue",
    "queue_position",
    "start_worker",
    "start_worker_pool",
    "stop_worker_pool",
    "worker_pool_status",
]
