"""Standalone worker process for the DB-backed task scheduler pool.

Runs as a subprocess, polls the scheduled_tasks table, and executes
jobs. Checks for a sentinel file to know when to shut down gracefully.

Each worker has a unique ID (0-N) for pool management and abort support.

Usage:
    python -m arxiv_manager.scheduler.worker --sentinel /path/to/sentinel --worker-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from ..db import init_db
from .models import ScheduledTask
from .queue import check_abort_sentinel, cleanup_abort_sentinel, complete_job, dequeue, fail_job

logger = logging.getLogger("arxiv_manager.scheduler.worker")

# Sentinel: worker exits when this file is deleted
_SENTINEL_PATH: Path | None = None
# Worker identity
_WORKER_ID: int = 0


class JobCancelledError(Exception):
    """Raised when a running job is aborted via sentinel."""


def _setup_logging() -> None:
    """Configure logging for the worker subprocess with log rotation."""
    from logging.handlers import RotatingFileHandler

    from ..storage import STORAGE_DIR

    log_path = STORAGE_DIR / "_scheduler_worker.log"
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Rotating file handler: 5MB max, keep 3 backups
    file_handler = RotatingFileHandler(
        str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s [worker-%(name)s] %(levelname)s: %(message)s"))

    # Also log to stderr for subprocess capture
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter("%(asctime)s [worker-%(name)s] %(levelname)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)


def _sentinel_exists() -> bool:
    """Check if the sentinel file still exists."""
    global _SENTINEL_PATH
    if _SENTINEL_PATH is None:
        return True
    return _SENTINEL_PATH.exists()


def _execute_job(task: ScheduledTask) -> None:
    """Execute a single job based on its type."""
    payload = json.loads(task.payload) if task.payload else {}

    if task.type == "generate_qa":
        _execute_generate_qa(task.id, payload)
    elif task.type == "regenerate_task":
        _execute_regenerate_task(task.id, payload)
    elif task.type == "validate_batch":
        _execute_validate_batch(task.id, payload)
    elif task.type == "rag_index":
        _execute_rag_index(task.id, payload)
    else:
        fail_job(task.id, f"Unknown job type: {task.type}")


def _execute_regenerate_task(job_id: int, payload: dict) -> None:
    """Regenerate a task's Q&A via the full gate chain (worker-callable)."""
    try:
        # Check abort sentinel before starting
        if check_abort_sentinel(job_id):
            cleanup_abort_sentinel(job_id)
            fail_job(job_id, "aborted by user")
            return

        from ..agents.orchestrator import run_regeneration

        task_id = payload.get("task_id")
        difficulty = payload.get("difficulty", "challenging")
        if not task_id:
            fail_job(job_id, "regenerate_task: missing task_id in payload")
            return
        result = run_regeneration(task_id, difficulty, source_route="scheduler_worker")

        # Check abort after completion
        if check_abort_sentinel(job_id):
            cleanup_abort_sentinel(job_id)
            fail_job(job_id, "aborted by user")
            return

        if result and result.get("ok"):
            complete_job(job_id, result)
        else:
            error = (result or {}).get("error", "Regeneration failed")
            logger.warning("worker-%d: regenerate_task job %d failed: %s", _WORKER_ID, job_id, error)
            fail_job(job_id, error)
    except JobCancelledError:
        cleanup_abort_sentinel(job_id)
        fail_job(job_id, "aborted by user")
    except Exception as exc:
        logger.exception("worker-%d: regenerate_task job %d failed", _WORKER_ID, job_id)
        fail_job(job_id, str(exc))


def _execute_generate_qa(job_id: int, payload: dict) -> None:
    """Generate a Q&A pair via the AI drafting pipeline."""
    try:
        if check_abort_sentinel(job_id):
            cleanup_abort_sentinel(job_id)
            fail_job(job_id, "aborted by user")
            return

        from ..authoring.ai_draft.core import draft_qa

        result = draft_qa(**payload)
        if result:
            complete_job(job_id, result)
        else:
            fail_job(job_id, "Generation returned None")
    except JobCancelledError:
        cleanup_abort_sentinel(job_id)
        fail_job(job_id, "aborted by user")
    except Exception as exc:
        logger.exception("worker-%d: generate_qa job %d failed", _WORKER_ID, job_id)
        fail_job(job_id, str(exc))


def _execute_validate_batch(job_id: int, payload: dict) -> None:
    """Validate a batch of Q&A pairs."""
    try:
        from ..authoring.validator import validate_task

        tasks = payload.get("tasks", [])
        results = []
        for t in tasks:
            v = validate_task(**t)
            results.append({"is_valid": v.is_valid, "errors": v.errors})
        complete_job(job_id, {"results": results, "total": len(results)})
    except Exception as exc:
        logger.exception("worker-%d: validate_batch job %d failed", _WORKER_ID, job_id)
        fail_job(job_id, str(exc))


def _execute_rag_index(job_id: int, payload: dict) -> None:
    """Index figures in the RAG vector store."""
    try:
        from ..sourcing.indexers import index_figures

        count = index_figures(
            paper_ids=payload.get("paper_ids"),
            force=payload.get("force", False),
        )
        complete_job(job_id, {"indexed": count})
    except Exception as exc:
        logger.exception("worker-%d: rag_index job %d failed", _WORKER_ID, job_id)
        fail_job(job_id, str(exc))


def _main_loop(poll_interval: float = 1.0) -> None:
    """Main worker loop: poll for jobs and execute them."""
    logger.info("worker-%d: started (poll_interval=%.1fs)", _WORKER_ID, poll_interval)

    while _sentinel_exists():
        # Write heartbeat so watchdog knows we're alive
        try:
            from ..scheduler.manager import write_heartbeat
            write_heartbeat(_WORKER_ID)
        except Exception:
            pass

        task = dequeue(worker_id=_WORKER_ID)
        if task is None:
            time.sleep(poll_interval)
            continue

        logger.info("worker-%d: executing job %d (type=%s)", _WORKER_ID, task.id, task.type)
        _execute_job(task)

    logger.info("worker-%d: sentinel removed, shutting down", _WORKER_ID)


def _pid_file_path() -> Path:
    """Path to the worker PID file (used for orphan detection)."""
    from ..storage import STORAGE_DIR

    return STORAGE_DIR / f"_scheduler_worker_{_WORKER_ID}.pid"


def _write_pid_file() -> None:
    """Write this worker's PID so the manager can detect orphans after restart."""
    try:
        _pid_file_path().write_text(str(os.getpid()))
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("worker-%d: could not write pid file: %s", _WORKER_ID, exc)


def _remove_pid_file() -> None:
    """Remove the PID file on clean shutdown."""
    try:
        pid_file = _pid_file_path()
        if pid_file.exists():
            pid_file.unlink()
    except Exception:  # pragma: no cover - best-effort
        pass


def _requeue_stale_running_jobs() -> None:
    """Reset jobs stuck in 'running' that belong to dead workers.

    Only resets jobs where the worker_id is NOT alive — jobs being
    processed by other live workers are left alone.
    """
    from sqlmodel import select, update

    from ..db import get_session
    from .models import ScheduledTask

    session = get_session()
    try:
        # Find all running jobs
        running = session.exec(
            select(ScheduledTask).where(ScheduledTask.status == "running")
        ).all()

        # Check which worker_ids are alive
        from .manager import _orphan_worker_pids

        alive_pids = _orphan_worker_pids()
        # Also check in-memory (we are the worker, so we're alive)
        alive_pids[_WORKER_ID] = os.getpid()

        stale_count = 0
        for job in running:
            # If the worker that owns this job is not alive, requeue it
            if job.worker_id not in alive_pids:
                session.exec(
                    update(ScheduledTask)
                    .where(ScheduledTask.id == job.id)
                    .values(status="queued", worker_id=0)
                )
                stale_count += 1

        session.commit()
        if stale_count:
            logger.warning("worker-%d: requeued %d stale running job(s)", _WORKER_ID, stale_count)
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("worker-%d: requeue stale jobs failed: %s", _WORKER_ID, exc)
    finally:
        session.close()


def main() -> None:
    """Worker entry point."""
    global _SENTINEL_PATH, _WORKER_ID

    parser = argparse.ArgumentParser(description="Scheduler worker subprocess")
    parser.add_argument("--sentinel", type=str, required=True, help="Path to sentinel file (worker exits when deleted)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between queue polls")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker pool slot ID (0-N)")
    args = parser.parse_args()

    _SENTINEL_PATH = Path(args.sentinel)
    _WORKER_ID = args.worker_id

    _setup_logging()
    logger.info("worker-%d: init DB", _WORKER_ID)
    init_db()

    # Register agent pipeline so orchestrator uses the registry
    try:
        from ..agents.registry import register_all_agents
        register_all_agents()
        logger.info("worker-%d: agents registered", _WORKER_ID)
    except Exception as exc:
        logger.warning("worker-%d: agent registration failed (will use fallback): %s", _WORKER_ID, exc)

    _write_pid_file()
    _requeue_stale_running_jobs()
    try:
        _main_loop(args.poll_interval)
    finally:
        _remove_pid_file()
        # Clean up heartbeat file
        try:
            from ..scheduler.manager import cleanup_heartbeat
            cleanup_heartbeat(_WORKER_ID)
        except Exception:
            pass


if __name__ == "__main__":
    main()
