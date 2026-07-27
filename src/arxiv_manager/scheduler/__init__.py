"""DB-backed task scheduler with subprocess worker.

Provides an async job queue using the existing SQLite database.
Jobs are enqueued via API, picked up by a subprocess worker,
and their status can be polled via API.

No external dependencies (no Redis, no Celery).
"""

from __future__ import annotations

from .manager import start_worker, stop_worker, worker_is_alive
from .models import ScheduledTask
from .queue import cancel_job, dequeue, enqueue, get_job_status, list_queue

__all__ = [
    "ScheduledTask",
    "enqueue",
    "dequeue",
    "cancel_job",
    "get_job_status",
    "list_queue",
    "start_worker",
    "stop_worker",
    "worker_is_alive",
]
