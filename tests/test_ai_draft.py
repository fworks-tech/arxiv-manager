"""Tests for AI draft module — unit tests without external API calls."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from arxiv_manager.authoring.ai_draft import (
    _get_api_key,
    _log_draft,
    draft_qa,
    draft_qa_consensus,
    verify_draft,
)


def test_get_api_key_opencode(monkeypatch):
    """_get_api_key returns OPENCODE_API_KEY value."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-opencode-key")
    assert _get_api_key("opencode") == "test-opencode-key"


def test_get_api_key_openai(monkeypatch):
    """_get_api_key returns OPENAI_API_KEY value."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    assert _get_api_key("openai") == "test-openai-key"


def test_get_api_key_anthropic(monkeypatch):
    """_get_api_key returns ANTHROPIC_API_KEY value."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    assert _get_api_key("anthropic") == "test-anthropic-key"


def test_get_api_key_missing(monkeypatch):
    """_get_api_key returns None for unset keys."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _get_api_key("opencode") is None


def test_get_api_key_unknown_provider():
    """_get_api_key returns None for unknown providers."""
    assert _get_api_key("unknown") is None


def test_log_draft_writes_jsonl(tmp_path):
    """_log_draft appends a JSONL record to storage/_draft_telemetry.jsonl."""
    import arxiv_manager.authoring.ai_draft as draft_mod
    original_path = draft_mod._TELEMETRY_PATH
    test_path = tmp_path / "_draft_telemetry.jsonl"
    draft_mod._TELEMETRY_PATH = test_path
    try:
        _log_draft(
            model="test-model",
            ok=True,
            elapsed=1.5,
            difficulty="challenging",
            figure_type="chart_graph_text",
            figure_path="/tmp/test.png",
        )
        assert test_path.exists()
        lines = test_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["model"] == "test-model"
        assert record["ok"] is True
        assert record["elapsed_s"] == 1.5
        assert record["difficulty"] == "challenging"
    finally:
        draft_mod._TELEMETRY_PATH = original_path


def test_log_draft_handles_error(tmp_path):
    """_log_draft truncates error to 100 chars."""
    import arxiv_manager.authoring.ai_draft as draft_mod
    original_path = draft_mod._TELEMETRY_PATH
    test_path = tmp_path / "_draft_telemetry.jsonl"
    draft_mod._TELEMETRY_PATH = test_path
    try:
        long_error = "x" * 200
        _log_draft(
            model="test", ok=False, elapsed=2.0,
            difficulty="easy", figure_type="", figure_path="/tmp/t.png",
            error=long_error,
        )
        record = json.loads(test_path.read_text().strip())
        assert len(record["error"]) == 100
    finally:
        draft_mod._TELEMETRY_PATH = original_path


def test_draft_qa_no_key(sample_image_chart_path, mock_no_api_key):
    """draft_qa returns None when no API key is set."""
    result = draft_qa(
        image_path=sample_image_chart_path,
        provider="opencode",
        api_key=None,
    )
    assert result is None


def test_draft_qa_consensus_no_key(sample_image_chart_path, mock_no_api_key):
    """draft_qa_consensus returns None when no API key is set."""
    result = draft_qa_consensus(
        image_path=sample_image_chart_path,
        n_attempts=1,
        verify=False,
        provider="opencode",
        api_key=None,
    )
    assert result is None


def test_verify_draft_no_key(sample_image_chart_path, mock_no_api_key):
    """verify_draft returns the original draft unchanged when no API key."""
    original = {"question": "Q?", "answer": "A", "answer_format": "word", "task_type": "chart"}
    result = verify_draft(
        image_path=sample_image_chart_path,
        draft=original,
        provider="opencode",
        api_key=None,
    )
    # Falls back to the original draft (no verification possible)
    assert result == original
