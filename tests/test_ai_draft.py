"""Tests for AI draft module — unit tests without external API calls."""

import json

from arxiv_manager.authoring._draft_telemetry import log_draft
from arxiv_manager.authoring.ai_draft import (
    _get_api_key,
    draft_qa,
    draft_qa_consensus,
    verify_draft,
)


def test_get_api_key_returns_env_value(monkeypatch):
    """_get_api_key returns OPENCODE_API_KEY value when set."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-opencode-key")
    assert _get_api_key() == "test-opencode-key"


def test_get_api_key_returns_none_when_missing(monkeypatch):
    """_get_api_key returns None when env var is unset."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    assert _get_api_key() is None


def test_log_draft_writes_jsonl(tmp_path):
    """log_draft appends a JSONL record to storage/_draft_telemetry.jsonl."""
    import arxiv_manager.authoring._draft_telemetry as tel_mod
    original_path = tel_mod._TELEMETRY_PATH
    test_path = tmp_path / "_draft_telemetry.jsonl"
    tel_mod._TELEMETRY_PATH = test_path
    try:
        log_draft(
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
        tel_mod._TELEMETRY_PATH = original_path


def test_log_draft_handles_error(tmp_path):
    """log_draft truncates error to 100 chars."""
    import arxiv_manager.authoring._draft_telemetry as tel_mod
    original_path = tel_mod._TELEMETRY_PATH
    test_path = tmp_path / "_draft_telemetry.jsonl"
    tel_mod._TELEMETRY_PATH = test_path
    try:
        long_error = "x" * 200
        log_draft(
            model="test", ok=False, elapsed=2.0,
            difficulty="easy", figure_type="", figure_path="/tmp/t.png",
            error=long_error,
        )
        record = json.loads(test_path.read_text().strip())
        assert len(record["error"]) == 100
    finally:
        tel_mod._TELEMETRY_PATH = original_path


def test_draft_qa_no_key(sample_image_chart_path, mock_no_api_key):
    """draft_qa returns None when no API key is set."""
    result = draft_qa(
        image_path=sample_image_chart_path,
        api_key=None,
    )
    assert result is None


def test_draft_qa_consensus_no_key(sample_image_chart_path, mock_no_api_key):
    """draft_qa_consensus returns None when no API key is set."""
    result = draft_qa_consensus(
        image_path=sample_image_chart_path,
        n_attempts=1,
        verify=False,
        api_key=None,
    )
    assert result is None


def test_verify_draft_no_key(sample_image_chart_path, mock_no_api_key):
    """verify_draft returns the original draft unchanged when no API key."""
    original = {"question": "Q?", "answer": "A", "answer_format": "word", "task_type": "chart"}
    result = verify_draft(
        image_path=sample_image_chart_path,
        draft=original,
        api_key=None,
    )
    # Falls back to the original draft (no verification possible)
    assert result == original


def test_verify_draft_returns_original_dict(sample_image_chart_path, mock_no_api_key):
    """verify_draft returns the same dict object (not a copy) when no API key."""
    original = {"question": "Q?", "answer": "A", "answer_format": "word", "task_type": "chart"}
    result = verify_draft(
        image_path=sample_image_chart_path,
        draft=original,
        api_key=None,
    )
    assert result is original


def test_draft_qa_figure_id_passthrough(sample_image_chart_path, mock_api_key, monkeypatch):
    """draft_qa accepts figure_id without error."""
    from arxiv_manager.authoring.ai_draft import draft_qa

    def fake_call(*a, **kw):
        return {"question": "Q?", "answer": "5", "answer_format": "number", "task_type": "chart"}

    import arxiv_manager.authoring.ai_draft._api_client as api_mod
    import arxiv_manager.authoring.ai_draft.core as core_mod
    monkeypatch.setattr(api_mod, "_call_opencode", fake_call)
    monkeypatch.setattr(core_mod, "_call_opencode", fake_call)

    result = draft_qa(
        image_path=sample_image_chart_path,
        api_key="test-key",
        difficulty="easy",
        figure_id=42,
    )
    assert result is not None
    assert result["question"] == "Q?"
    assert result["answer"] == "5"


def test_draft_qa_includes_prompt_version_in_dict(sample_image_chart_path, mock_api_key, monkeypatch):
    """draft_qa returns _prompt_version_id and _prompt_text_hash in the dict."""
    from arxiv_manager.authoring.ai_draft import draft_qa

    def fake_call(*a, **kw):
        return {"question": "Q?", "answer": "5", "answer_format": "number", "task_type": "chart"}

    import arxiv_manager.authoring.ai_draft._api_client as api_mod
    import arxiv_manager.authoring.ai_draft.core as core_mod
    monkeypatch.setattr(api_mod, "_call_opencode", fake_call)
    monkeypatch.setattr(core_mod, "_call_opencode", fake_call)

    result = draft_qa(
        image_path=sample_image_chart_path,
        api_key="test-key",
        difficulty="easy",
    )
    assert result is not None
    assert "_prompt_version_id" in result
    assert "_prompt_text_hash" in result
    assert result["_prompt_version_id"].startswith("EASY_PROMPT@")


def test_draft_qa_prompt_template_selection(sample_image_chart_path, mock_api_key, monkeypatch):
    """draft_qa selects correct prompt template per difficulty."""
    from arxiv_manager.authoring.ai_draft import draft_qa

    captured = []

    def fake_call(*a, **kw):
        captured.append(True)
        return {"question": "Q?", "answer": "5", "answer_format": "number", "task_type": "chart"}

    import arxiv_manager.authoring.ai_draft._api_client as api_mod
    import arxiv_manager.authoring.ai_draft.core as core_mod
    monkeypatch.setattr(api_mod, "_call_opencode", fake_call)
    monkeypatch.setattr(core_mod, "_call_opencode", fake_call)

    result = draft_qa(
        image_path=sample_image_chart_path,
        api_key="test-key",
        difficulty="hardest",
        figure_type="chart_graph_text",
    )
    assert result is not None
    assert len(captured) == 1
    assert "_prompt_version_id" in result
    assert result["_prompt_version_id"].startswith("HARDEST_PROMPT@")


def test_import_from_new_sub_modules():
    """All public symbols importable from their new sub-module locations."""
    from arxiv_manager.authoring.ai_draft._api_client import _call_opencode, _get_api_key
    from arxiv_manager.authoring.ai_draft._response_parser import (
        _extract_reasoning,
        _parse_critique_response,
        _parse_llm_response,
    )
    from arxiv_manager.authoring.ai_draft._verifier import verify_draft
    from arxiv_manager.authoring.ai_draft.composition import draft_qa_consensus, draft_with_self_critique
    from arxiv_manager.authoring.ai_draft.core import draft_qa
    # Verify all callable
    assert callable(_extract_reasoning)
    assert callable(_parse_llm_response)
    assert callable(_parse_critique_response)
    assert callable(_call_opencode)
    assert callable(_get_api_key)
    assert callable(draft_qa)
    assert callable(draft_qa_consensus)
    assert callable(draft_with_self_critique)
    assert callable(verify_draft)


def test_parse_critique_response_valid_critique():
    """_parse_critique_response accepts the critique format."""
    from arxiv_manager.authoring.ai_draft import _parse_critique_response
    text = '{"score": 3, "rewrite_question": "Rewrite Q?", "rewrite_answer": "7"}'
    result = _parse_critique_response(text)
    assert result is not None
    assert result["score"] == 3
    assert result["rewrite_question"] == "Rewrite Q?"


def test_parse_critique_response_none_for_empty():
    """_parse_critique_response returns None for empty input."""
    from arxiv_manager.authoring.ai_draft import _parse_critique_response
    assert _parse_critique_response("") is None
