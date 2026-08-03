"""Tests for scheduler/manager.py — subprocess lifecycle management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arxiv_manager.scheduler.manager import (
    get_worker_pid,
    start_worker,
    stop_worker,
    worker_is_alive,
)


@pytest.fixture(autouse=True)
def _reset_globals(tmp_path, monkeypatch):
    """Reset manager globals before each test to prevent bleed.

    Also points STORAGE_DIR at a temp dir so the real storage's worker PID
    file (written by a live worker) can't be mistaken for an orphan.
    """
    import arxiv_manager.scheduler.manager as mgr
    from arxiv_manager import storage as st_mod

    monkeypatch.setattr(st_mod, "STORAGE_DIR", tmp_path)

    mgr._WORKERS.clear()


class TestStartWorker:
    @patch("arxiv_manager.scheduler.manager.subprocess.Popen")
    @patch("arxiv_manager.scheduler.manager._create_sentinel")
    def test_start_worker_returns_pid(self, mock_sentinel, mock_popen):
        mock_sentinel.return_value = MagicMock()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        pid = start_worker()
        assert pid == 12345

    @patch("arxiv_manager.scheduler.manager.subprocess.Popen")
    @patch("arxiv_manager.scheduler.manager._create_sentinel")
    def test_start_worker_twice(self, mock_sentinel, mock_popen):
        mock_sentinel.return_value = MagicMock()
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # Start first worker via direct call
        start_worker()
        # Second call should return existing PID (pool already has a worker)
        pid2 = start_worker()
        assert pid2 == 12345


class TestStopWorker:
    @patch("arxiv_manager.scheduler.manager.subprocess.Popen")
    @patch("arxiv_manager.scheduler.manager._create_sentinel")
    def test_stop_worker_graceful(self, mock_sentinel, mock_popen):
        sentinel_file = MagicMock()
        sentinel_file.exists.return_value = True
        mock_sentinel.return_value = sentinel_file

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        start_worker()
        result = stop_worker(timeout=1.0)
        assert result is True
        sentinel_file.unlink.assert_called_once()

    def test_stop_when_not_running(self):
        import arxiv_manager.scheduler.manager as mgr

        mgr._WORKERS.clear()
        assert stop_worker() is False


class TestWorkerIsAlive:
    def test_no_process(self):
        import arxiv_manager.scheduler.manager as mgr

        mgr._WORKERS.clear()
        assert worker_is_alive() is False

    @patch("arxiv_manager.scheduler.manager._orphan_worker_pids", return_value={})
    def test_process_running(self, _mock_orphans):
        import arxiv_manager.scheduler.manager as mgr

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        mgr._WORKERS[0] = mgr._WorkerSlot(id=0, process=mock_process)
        assert worker_is_alive() is True

    @patch("arxiv_manager.scheduler.manager._orphan_worker_pids", return_value={})
    def test_process_exited(self, _mock_orphans):
        import arxiv_manager.scheduler.manager as mgr

        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_process.pid = 12345
        mgr._WORKERS[0] = mgr._WorkerSlot(id=0, process=mock_process)
        assert worker_is_alive() is False

    def test_get_pid_no_process(self):
        import arxiv_manager.scheduler.manager as mgr

        mgr._WORKERS.clear()
        assert get_worker_pid() is None
