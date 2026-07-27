"""Subprocess lifecycle management for the scheduler worker.

Uses a sentinel file for graceful shutdown (Windows-compatible).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROCESS: subprocess.Popen | None = None
_SENTINEL_PATH: Path | None = None


def _create_sentinel() -> Path:
    """Create a unique sentinel file for worker lifecycle."""
    f = tempfile.NamedTemporaryFile(prefix="scheduler_", suffix=".sentinel", delete=False)
    f.write(b"running")
    f.close()
    path = Path(f.name)
    logger.debug("manager: sentinel at %s", path)
    return path


def start_worker(poll_interval: float = 1.0) -> int | None:
    """Start the scheduler worker as a subprocess.

    Creates a sentinel file that the worker monitors. The worker
    exits gracefully when the sentinel is deleted.

    Args:
        poll_interval: Seconds between queue polls.

    Returns:
        PID of the worker process, or None if already running.
    """
    global _PROCESS, _SENTINEL_PATH

    if _PROCESS is not None and _PROCESS.poll() is None:
        logger.info("manager: worker already running (PID %d)", _PROCESS.pid)
        return _PROCESS.pid

    sentinel = _create_sentinel()
    _SENTINEL_PATH = sentinel

    _PROCESS = subprocess.Popen(
        [
            sys.executable, "-m", "arxiv_manager.scheduler.worker",
            "--sentinel", str(sentinel),
            "--poll-interval", str(poll_interval),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    pid = _PROCESS.pid
    logger.info("manager: worker started (PID %d, sentinel=%s)", pid, sentinel)
    return pid


def stop_worker(timeout: float = 5.0) -> bool:
    """Stop the scheduler worker gracefully.

    Deletes the sentinel file and waits for the process to exit.
    Terminates forcefully if timeout is exceeded.

    Args:
        timeout: Seconds to wait for graceful shutdown.

    Returns:
        True if worker stopped, False if it was not running.
    """
    global _PROCESS, _SENTINEL_PATH

    if _PROCESS is None or _PROCESS.poll() is not None:
        _PROCESS = None
        return False

    # Signal graceful shutdown by deleting sentinel
    if _SENTINEL_PATH and _SENTINEL_PATH.exists():
        _SENTINEL_PATH.unlink()
        logger.info("manager: sentinel deleted, waiting for worker to exit")

    # Wait for graceful exit
    try:
        _PROCESS.wait(timeout=timeout)
        logger.info("manager: worker exited gracefully (PID %d)", _PROCESS.pid)
    except subprocess.TimeoutExpired:
        logger.warning("manager: worker did not exit within %.1fs, terminating", timeout)
        _PROCESS.terminate()
        try:
            _PROCESS.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _PROCESS.kill()

    _PROCESS = None
    return True


def worker_is_alive() -> bool:
    """Check if the worker process is running."""
    global _PROCESS
    if _PROCESS is None:
        return False
    return _PROCESS.poll() is None


def get_worker_pid() -> int | None:
    """Return the worker PID, or None if not running."""
    global _PROCESS
    if _PROCESS is None:
        return None
    if _PROCESS.poll() is None:
        return _PROCESS.pid
    return None
