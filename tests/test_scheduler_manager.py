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
def _reset_globals():
    """Reset manager globals before each test to prevent bleed."""
    import arxiv_manager.scheduler.manager as mgr

    mgr._PROCESS = None
    mgr._SENTINEL_PATH = None


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
        # Second call should return existing PID
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
        # Clean global state from previous tests
        import arxiv_manager.scheduler.manager as mgr

        mgr._PROCESS = None
        assert stop_worker() is False


class TestWorkerIsAlive:
    @patch("arxiv_manager.scheduler.manager._PROCESS", None)
    def test_no_process(self):
        assert worker_is_alive() is False

    @patch("arxiv_manager.scheduler.manager._PROCESS")
    def test_process_running(self, mock_process):
        mock_process.poll.return_value = None
        assert worker_is_alive() is True

    @patch("arxiv_manager.scheduler.manager._PROCESS")
    def test_process_exited(self, mock_process):
        mock_process.poll.return_value = 0
        assert worker_is_alive() is False

    @patch("arxiv_manager.scheduler.manager._PROCESS", None)
    def test_get_pid_no_process(self):
        assert get_worker_pid() is None
