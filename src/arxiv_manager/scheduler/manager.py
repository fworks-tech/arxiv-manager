"""Subprocess lifecycle management for the scheduler worker pool.

Uses sentinel files for graceful shutdown (Windows-compatible).
Each worker in the pool has its own PID file, sentinel, and log handle.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _WorkerSlot:
    """Tracks one worker subprocess in the pool."""

    id: int
    process: subprocess.Popen | None = None
    sentinel: Path | None = None
    log_handle: object | None = None


_WORKERS: dict[int, _WorkerSlot] = {}
_DEFAULT_POOL_SIZE = 5


def _storage_dir() -> Path:
    from ..storage import STORAGE_DIR

    return STORAGE_DIR


def _create_sentinel() -> Path:
    """Create a unique sentinel file for worker lifecycle."""
    f = tempfile.NamedTemporaryFile(prefix="scheduler_", suffix=".sentinel", delete=False)
    f.write(b"running")
    f.close()
    path = Path(f.name)
    logger.debug("manager: sentinel at %s", path)
    return path


def _pid_file_path(worker_id: int) -> Path:
    """Path to the worker PID file for a specific pool slot."""
    return _storage_dir() / f"_scheduler_worker_{worker_id}.pid"


def _process_is_alive(pid: int) -> bool:
    """Cross-platform process-liveness check."""
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


def _orphan_worker_pids() -> dict[int, int]:
    """Return {worker_id: pid} for surviving workers from PID files."""
    result = {}
    storage = _storage_dir()
    for pid_file in storage.glob("_scheduler_worker_*.pid"):
        try:
            name = pid_file.stem  # e.g. _scheduler_worker_0
            wid = int(name.split("_")[-1])
            pid = int(pid_file.read_text().strip())
            if pid > 0 and _process_is_alive(pid):
                result[wid] = pid
        except (ValueError, IndexError, OSError):
            pass
    return result


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


def _spawn_worker(worker_id: int, poll_interval: float = 1.0) -> _WorkerSlot:
    """Spawn a single worker subprocess."""
    sentinel = _create_sentinel()
    src_dir = Path(__file__).resolve().parent.parent.parent
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing if existing else "")

    log_dir = _storage_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "_scheduler_worker.log"
    log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "arxiv_manager.scheduler.worker",
            "--sentinel",
            str(sentinel),
            "--poll-interval",
            str(poll_interval),
            "--worker-id",
            str(worker_id),
        ],
        cwd=str(src_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    slot = _WorkerSlot(id=worker_id, process=process, sentinel=sentinel, log_handle=log_handle)
    logger.info("manager: worker %d started (PID %d, sentinel=%s)", worker_id, process.pid, sentinel)
    return slot


def start_worker_pool(count: int = _DEFAULT_POOL_SIZE, poll_interval: float = 1.0) -> list[int]:
    """Start N workers as a pool. Adopts orphans where possible.

    Returns list of PIDs for all running workers.
    """
    # Adopt orphans first
    orphans = _orphan_worker_pids()
    for wid, pid in orphans.items():
        if wid not in _WORKERS:
            _WORKERS[wid] = _WorkerSlot(id=wid)
            logger.info("manager: adopting orphan worker %d (PID %d)", wid, pid)

    # Spawn missing workers
    for wid in range(count):
        if wid in _WORKERS:
            slot = _WORKERS[wid]
            if slot.process and slot.process.poll() is None:
                continue  # already running
        slot = _spawn_worker(wid, poll_interval)
        _WORKERS[wid] = slot

    return [slot.process.pid for slot in _WORKERS.values() if slot.process and slot.process.poll() is None]


def stop_worker_pool(timeout: float = 5.0) -> int:
    """Stop all workers gracefully. Returns number of workers stopped."""
    stopped = 0

    for wid, slot in list(_WORKERS.items()):
        # Delete sentinel for graceful shutdown
        if slot.sentinel and slot.sentinel.exists():
            slot.sentinel.unlink()
            logger.info("manager: worker %d sentinel deleted", wid)

        if slot.process is None or slot.process.poll() is not None:
            # Dead handle — check PID file for orphan
            pid = _read_pid_file(wid)
            if pid and _process_is_alive(pid):
                _terminate_process(pid)
                logger.info("manager: terminated orphan worker %d (PID %d)", wid, pid)
            _remove_pid_file(wid)
            stopped += 1
            continue

        # Wait for graceful exit
        try:
            slot.process.wait(timeout=timeout)
            logger.info("manager: worker %d exited gracefully (PID %d)", wid, slot.process.pid)
        except subprocess.TimeoutExpired:
            logger.warning("manager: worker %d did not exit within %.1fs, terminating", wid, timeout)
            slot.process.terminate()
            try:
                slot.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                slot.process.kill()

        if slot.log_handle:
            try:
                slot.log_handle.close()
            except Exception:
                pass
        _remove_pid_file(wid)
        stopped += 1

    _WORKERS.clear()
    return stopped


def worker_pool_status() -> list[dict]:
    """Return status of each worker in the pool."""
    status = []
    # Include live workers from handles
    for wid in sorted(_WORKERS.keys()):
        slot = _WORKERS[wid]
        if slot.process and slot.process.poll() is None:
            status.append({"id": wid, "pid": slot.process.pid, "alive": True})
        else:
            # Check PID file for orphan
            pid = _read_pid_file(wid)
            alive = pid is not None and _process_is_alive(pid)
            status.append({"id": wid, "pid": pid, "alive": alive})
    # Include orphans not in _WORKERS
    orphans = _orphan_worker_pids()
    for wid, pid in orphans.items():
        if wid not in _WORKERS:
            status.append({"id": wid, "pid": pid, "alive": True, "orphan": True})
    return status


def worker_is_alive() -> bool:
    """Check if ANY worker in the pool is running."""
    for slot in _WORKERS.values():
        if slot.process and slot.process.poll() is None:
            return True
    return len(_orphan_worker_pids()) > 0


def get_worker_pid() -> int | None:
    """Return any live worker PID, or None."""
    for slot in _WORKERS.values():
        if slot.process and slot.process.poll() is None:
            return slot.process.pid
    orphans = _orphan_worker_pids()
    if orphans:
        return next(iter(orphans.values()))
    return None


def _read_pid_file(worker_id: int) -> int | None:
    """Read a worker's PID file, return PID or None."""
    try:
        pid_file = _pid_file_path(worker_id)
        if not pid_file.exists():
            return None
        pid = int(pid_file.read_text().strip())
        return pid if pid > 0 else None
    except (ValueError, OSError):
        return None


def _remove_pid_file(worker_id: int) -> None:
    """Remove a worker's PID file."""
    try:
        _pid_file_path(worker_id).unlink(missing_ok=True)
    except OSError:
        pass


# Backward-compatible aliases
def start_worker(poll_interval: float = 1.0) -> int | None:
    """Start a single worker (backward compat)."""
    if not _WORKERS:
        start_worker_pool(count=1, poll_interval=poll_interval)
    return get_worker_pid()


def stop_worker(timeout: float = 5.0) -> bool:
    """Stop all workers (backward compat)."""
    return stop_worker_pool(timeout) > 0
