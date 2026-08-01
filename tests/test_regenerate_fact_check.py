"""Tests for the regenerate fact-check retry loop in _do_regenerate."""

from types import SimpleNamespace

import pytest


def _draft(question, answer):
    return {
        "question": question,
        "answer": answer,
        "answer_format": "number",
        "task_type": "chart",
        "_usage": {},
    }


def _valid_validation():
    return SimpleNamespace(quality_score=95.0, is_valid=True, errors=[], warnings=[])


def _fact_pass():
    return {"claims": [], "unsupported": [], "verdict": "pass", "checked": True}


def _fact_fail(claims):
    return {"claims": [], "unsupported": claims, "verdict": "fail", "checked": True}


@pytest.fixture(autouse=True)
def _patch_deps(monkeypatch):
    import arxiv_manager.authoring.ai_draft._fact_checker as fc_mod
    from arxiv_manager.web.routes import task_routes as tr

    monkeypatch.setattr(tr, "validate_task", lambda *a, **kw: _valid_validation())
    monkeypatch.setattr(tr, "_is_format_only_error", lambda ctx: False)
    monkeypatch.setattr(tr, "draft_with_self_critique", lambda **kw: None)
    monkeypatch.setattr(fc_mod, "fact_check_draft", None)
    yield


def _run(monkeypatch, difficulty="hardest", fact_results=None, drafts=None):
    """Drive _do_regenerate with scripted draft + fact-check responses."""
    import arxiv_manager.authoring.ai_draft._fact_checker as fc_mod
    from arxiv_manager.web.routes import task_routes as tr

    draft_calls = []

    def fake_draft_qa(**kw):
        draft_calls.append(kw)
        return drafts.pop(0) if drafts else _draft("Q?", "1")

    monkeypatch.setattr(tr, "draft_qa", fake_draft_qa)
    monkeypatch.setattr(fc_mod, "fact_check_draft", lambda question, image_path, api_key, difficulty="": fact_results.pop(0))

    return tr._do_regenerate(
        "img.jpg",
        "key",
        difficulty,
        "chart_graph_text",
        0.5,
        "prev?",
        figure_id=1,
        task_id=44,
    ), draft_calls


def test_fact_check_failure_feeds_retry_with_claims(monkeypatch):
    """A fact-check failure must retry with the claims as feedback, then pass."""
    result, calls = _run(
        monkeypatch,
        fact_results=[_fact_fail(["two text overlap panels"]), _fact_pass()],
        drafts=[_draft("Q1?", "1"), _draft("Q2?", "2")],
    )
    assert result is not None
    assert result["question"] == "Q2?"
    assert result["_fact_check_failed"] is False
    assert result["_fact_check_errors"] == []
    assert len(calls) == 2
    assert any("Fact check failed: two text overlap panels" in c.get("feedback", "") for c in calls)


def test_all_retries_fact_failed_returns_best_with_flag(monkeypatch):
    """When every retry fails fact-check, the last draft keeps the failure flag."""
    result, calls = _run(
        monkeypatch,
        fact_results=[_fact_fail(["two text overlap panels"]), _fact_fail(["two text overlap panels"]), _fact_fail(["two text overlap panels"])],
        drafts=[_draft("Q1?", "1"), _draft("Q2?", "2")],
    )
    assert result is not None
    assert result["_fact_check_failed"] is True
    assert result["_fact_check_errors"] == ["two text overlap panels"]


def test_fact_check_skipped_for_easy(monkeypatch):
    """Easy drafts skip the fact-check call entirely."""
    import arxiv_manager.authoring.ai_draft._fact_checker as fc_mod
    from arxiv_manager.web.routes import task_routes as tr

    calls = {"fc": 0}

    def fake_draft_qa(**kw):
        return _draft("Q?", "1")

    monkeypatch.setattr(tr, "draft_qa", fake_draft_qa)
    monkeypatch.setattr(
        fc_mod, "fact_check_draft",
        lambda *a, **kw: calls.__setitem__("fc", calls["fc"] + 1) or _fact_pass(),
    )

    result = tr._do_regenerate(
        "img.jpg", "key", "easy", "chart_graph_text", 0.5, "prev?", figure_id=1, task_id=44
    )
    assert result is not None
    assert calls["fc"] == 0


def test_fact_check_fail_open_when_checker_unavailable(monkeypatch):
    """A checker tooling failure (checked=False) must not block the draft."""
    import arxiv_manager.authoring.ai_draft._fact_checker as fc_mod
    from arxiv_manager.web.routes import task_routes as tr

    def fake_draft_qa(**kw):
        return _draft("Q?", "1")

    monkeypatch.setattr(tr, "draft_qa", fake_draft_qa)
    monkeypatch.setattr(fc_mod, "fact_check_draft", lambda *a, **kw: {"claims": [], "unsupported": [], "verdict": "pass", "checked": False})

    result = tr._do_regenerate(
        "img.jpg", "key", "hardest", "chart_graph_text", 0.5, "prev?", figure_id=1, task_id=44
    )
    assert result is not None
    assert result["_fact_check_failed"] is False
