"""Standalone worker process for the DB-backed task scheduler.

Runs as a subprocess, polls the scheduled_tasks table, and executes
jobs. Checks for a sentinel file to know when to shut down gracefully.

Usage:
    python -m arxiv_manager.scheduler.worker --sentinel /path/to/sentinel
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
from .queue import complete_job, dequeue, fail_job

logger = logging.getLogger("arxiv_manager.scheduler.worker")

# Sentinel: worker exits when this file is deleted
_SENTINEL_PATH: Path | None = None


def _setup_logging() -> None:
    """Configure logging for the worker subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [worker] %(levelname)s: %(message)s",
    )


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
        from ..web.routes.task_routes import run_regeneration

        task_id = payload.get("task_id")
        difficulty = payload.get("difficulty", "challenging")
        if not task_id:
            fail_job(job_id, "regenerate_task: missing task_id in payload")
            return
        result = run_regeneration(task_id, difficulty, source_route="scheduler_worker")
        if result and result.get("ok"):
            complete_job(job_id, result)
        else:
            error = (result or {}).get("error", "Regeneration failed")
            logger.warning("worker: regenerate_task job %d failed: %s", job_id, error)
            fail_job(job_id, error)
    except Exception as exc:
        logger.exception("worker: regenerate_task job %d failed", job_id)
        fail_job(job_id, str(exc))


def _execute_generate_qa(job_id: int, payload: dict) -> None:
    """Generate a Q&A pair via the AI drafting pipeline."""
    try:
        from ..authoring.ai_draft.core import draft_qa

        result = draft_qa(**payload)
        if result:
            complete_job(job_id, result)
        else:
            fail_job(job_id, "Generation returned None")
    except Exception as exc:
        logger.exception("worker: generate_qa job %d failed", job_id)
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
        logger.exception("worker: validate_batch job %d failed", job_id)
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
        logger.exception("worker: rag_index job %d failed", job_id)
        fail_job(job_id, str(exc))


def _main_loop(poll_interval: float = 1.0) -> None:
    """Main worker loop: poll for jobs and execute them."""
    logger.info("worker: started (poll_interval=%.1fs)", poll_interval)

    while _sentinel_exists():
        task = dequeue()
        if task is None:
            time.sleep(poll_interval)
            continue

        logger.info("worker: executing job %d (type=%s)", task.id, task.type)
        _execute_job(task)

    logger.info("worker: sentinel removed, shutting down")


_PID_FILE_NAME = "_scheduler_worker.pid"


def _pid_file_path() -> Path:
    """Path to the worker PID file (used for orphan detection)."""
    from ..storage import STORAGE_DIR

    return STORAGE_DIR / _PID_FILE_NAME


def _write_pid_file() -> None:
    """Write this worker's PID so the manager can detect orphans after restart."""
    try:
        _pid_file_path().write_text(str(os.getpid()))
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("worker: could not write pid file: %s", exc)


def _remove_pid_file() -> None:
    """Remove the PID file on clean shutdown."""
    try:
        pid_file = _pid_file_path()
        if pid_file.exists():
            pid_file.unlink()
    except Exception:  # pragma: no cover - best-effort
        pass


def _requeue_stale_running_jobs() -> None:
    """Reset jobs stuck in 'running' back to 'queued'.

    A job in 'running' with no live worker (server restarted, worker killed)
    would otherwise sit forever. Safe with the atomic dequeue + PID-file
    orphan detection: only one worker is ever alive, so any 'running' job
    belongs to a dead predecessor. Started_at is preserved for audit.
    """
    from sqlmodel import update

    from ..db import get_session
    from .models import ScheduledTask

    session = get_session()
    try:
        result = session.exec(
            update(ScheduledTask)
            .where(ScheduledTask.status == "running")
            .values(status="queued")
        )
        session.commit()
        if result.rowcount:
            logger.warning("worker: requeued %d stale running job(s)", result.rowcount)
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("worker: requeue stale jobs failed: %s", exc)
    finally:
        session.close()


def main() -> None:
    """Worker entry point."""
    parser = argparse.ArgumentParser(description="Scheduler worker subprocess")
    parser.add_argument("--sentinel", type=str, required=True, help="Path to sentinel file (worker exits when deleted)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between queue polls")
    args = parser.parse_args()

    global _SENTINEL_PATH
    _SENTINEL_PATH = Path(args.sentinel)

    _setup_logging()
    logger.info("worker: init DB")
    init_db()
    _write_pid_file()
    _requeue_stale_running_jobs()
    try:
        _main_loop(args.poll_interval)
    finally:
        _remove_pid_file()


if __name__ == "__main__":
    main()
