"""Subprocess lifecycle management for the scheduler worker.

Uses a sentinel file for graceful shutdown (Windows-compatible).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_PROCESS: subprocess.Popen | None = None
_SENTINEL_PATH: Path | None = None
_WORKER_LOG_HANDLE = None  # kept open for the worker's lifetime


def _create_sentinel() -> Path:
    """Create a unique sentinel file for worker lifecycle."""
    f = tempfile.NamedTemporaryFile(prefix="scheduler_", suffix=".sentinel", delete=False)
    f.write(b"running")
    f.close()
    path = Path(f.name)
    logger.debug("manager: sentinel at %s", path)
    return path


def _pid_file_path() -> Path:
    """Path to the worker PID file (worker writes it; manager reads it)."""
    from ..storage import STORAGE_DIR

    return STORAGE_DIR / "_scheduler_worker.pid"


def _process_is_alive(pid: int) -> bool:
    """Cross-platform process-liveness check.

    os.kill(pid, 0) works on POSIX but raises WinError 87 on Windows for
    signal 0, so use a ctypes OpenProcess probe there.
    """
    if os.name == "nt":
        import ctypes

        query_limited = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
        handle = ctypes.windll.kernel32.OpenProcess(query_limited, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _orphan_worker_pid() -> int | None:
    """Return the PID of a surviving worker from the PID file, if any.

    After a server restart the in-memory _PROCESS handle is gone, but the
    worker subprocess may still be alive (it only exits when its sentinel is
    deleted). Without this check the manager would spawn a second worker.
    """
    try:
        pid_file = _pid_file_path()
        if not pid_file.exists():
            return None
        pid = int(pid_file.read_text().strip())
        if pid <= 0:
            return None
        if _process_is_alive(pid):
            return pid
        return None
    except (ValueError, OSError):
        return None


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

    # A worker from a previous server process may still be alive — don't
    # spawn a duplicate.
    orphan = _orphan_worker_pid()
    if orphan is not None:
        logger.info("manager: adopting orphan worker PID %d", orphan)
        return orphan

    sentinel = _create_sentinel()
    _SENTINEL_PATH = sentinel

    # The package lives in src/; the server may be launched from the repo
    # root (e.g. `python -m uvicorn src.arxiv_manager.web.app:create_app`),
    # where `python -m arxiv_manager...` fails with ModuleNotFoundError and
    # the worker dies instantly. Spawn from src/ and pin PYTHONPATH so the
    # worker subprocess can import the package regardless of the server CWD
    # or launch mode.
    src_dir = Path(__file__).resolve().parent.parent.parent
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing if existing else "")

    # Worker stdout/stderr go to a log file, NOT PIPE: the manager never
    # drains PIPE buffers, so a verbose job (RAG prewarm prints hundreds of
    # lines) fills the ~64KB pipe and the worker blocks forever on its next
    # write, leaving jobs stuck in 'running'.
    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "storage"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "_scheduler_worker.log"
    log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 — held for worker lifetime

    _PROCESS = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "arxiv_manager.scheduler.worker",
            "--sentinel",
            str(sentinel),
            "--poll-interval",
            str(poll_interval),
        ],
        cwd=str(src_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _WORKER_LOG_HANDLE = log_handle  # noqa: N806 — module-level global

    pid = _PROCESS.pid
    logger.info("manager: worker started (PID %d, sentinel=%s)", pid, sentinel)
    return pid


def _terminate_process(pid: int) -> None:
    """Force-terminate a process cross-platform."""
    if os.name == "nt":
        import ctypes

        process_terminate = 0x0001  # PROCESS_TERMINATE
        handle = ctypes.windll.kernel32.OpenProcess(process_terminate, False, pid)
        if handle:
            ctypes.windll.kernel32.TerminateProcess(handle, 1)
            ctypes.windll.kernel32.CloseHandle(handle)
        return
    try:
        os.kill(pid, 9)
    except OSError:
        pass


def stop_worker(timeout: float = 5.0) -> bool:
    """Stop the scheduler worker gracefully.

    Deletes the sentinel file and waits for the process to exit.
    Terminates forcefully if timeout is exceeded.

    Args:
        timeout: Seconds to wait for graceful shutdown.

    Returns:
        True if worker stopped, False if it was not running.
    """
    global _PROCESS, _SENTINEL_PATH, _WORKER_LOG_HANDLE

    orphan = _orphan_worker_pid()
    if _PROCESS is None or _PROCESS.poll() is not None:
        # No in-memory handle, but a worker from a previous server process may
        # survive. Its sentinel path is lost, so terminate it directly.
        if orphan is not None and orphan != (_PROCESS.pid if _PROCESS else None):
            _terminate_process(orphan)
            logger.info("manager: terminated orphan worker PID %d", orphan)
            try:
                _pid_file_path().unlink()
            except OSError:
                pass
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

    if _WORKER_LOG_HANDLE is not None:
        try:
            _WORKER_LOG_HANDLE.close()
        except Exception:
            pass
        _WORKER_LOG_HANDLE = None
    _PROCESS = None
    return True


def worker_is_alive() -> bool:
    """Check if the worker process is running (handle or orphan PID file)."""
    global _PROCESS
    if _PROCESS is not None and _PROCESS.poll() is None:
        return True
    return _orphan_worker_pid() is not None


def get_worker_pid() -> int | None:
    """Return the worker PID, or None if not running."""
    global _PROCESS
    if _PROCESS is not None and _PROCESS.poll() is None:
        return _PROCESS.pid
    return _orphan_worker_pid()
