"""Tests for self-critique composition flow — feedback retry on invalid drafts."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _FakeImg:
    def __init__(self):
        self.mode = "RGB"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def convert(self, mode):
        return self

    def thumbnail(self, size):
        pass

    def save(self, buf, format=None, quality=None, optimize=None):
        buf.write(b"jpeg")


def _draft(question="Q?", answer="1", valid=False, quality=50.0):
    """Build a draft dict shaped like draft_qa output."""
    return {
        "question": question,
        "answer": answer,
        "answer_format": "number",
        "task_type": "chart",
        "_validation_is_valid": valid,
        "_validation_quality": quality,
        "_validation_errors": [] if valid else ["Question too simple for Challenging difficulty"],
        "_validation_warnings": [],
        "_model": "minimax-m3",
    }


def _validation_result(errors=("Question too simple for Challenging difficulty",)):
    """Stub validate_task result with errors."""
    return SimpleNamespace(
        quality_score=50.0,
        is_valid=not errors,
        errors=list(errors),
        warnings=[],
    )


def _critique_response(score=2, rewrite_q="Harder Q?", rewrite_a="7"):
    """Stub _call_opencode for the critique round."""
    return {
        "score": score,
        "rewrite_question": rewrite_q,
        "rewrite_answer": rewrite_a,
        "_usage": {},
    }


@pytest.fixture(autouse=True)
def _patch_pil():
    with patch("PIL.Image.open", return_value=_FakeImg()):
        yield


def test_rewrite_invalid_returns_valid_original():
    """When rewrite fails validation but original is valid, original is returned."""
    from arxiv_manager.authoring import validator
    from arxiv_manager.authoring.ai_draft import composition

    with (
        patch.object(composition, "draft_qa", return_value=_draft(valid=True)) as dqa,
        patch.object(composition, "_call_opencode", return_value=_critique_response()),
        patch.object(validator, "validate_task", return_value=_validation_result()),
    ):
        result = composition.draft_with_self_critique(
            "img.jpg", max_rounds=1, api_key="key", difficulty="challenging"
        )

    assert result is not None
    assert result["question"] == "Q?"
    assert result["_validation_is_valid"] is True
    dqa.assert_called_once()


def test_rewrite_and_original_invalid_retries_with_feedback():
    """When both rewrite and original fail, feedback retries run; first valid wins."""
    from arxiv_manager.authoring import validator
    from arxiv_manager.authoring.ai_draft import composition

    retry_returns = [
        _draft("Retry1?", "2", valid=False),
        _draft("Retry2?", "3", valid=True),
    ]

    def fake_draft_qa(**kwargs):
        if kwargs.get("feedback"):
            return retry_returns.pop(0)
        return _draft(valid=False)

    with (
        patch.object(composition, "draft_qa", side_effect=fake_draft_qa) as dqa,
        patch.object(composition, "_call_opencode", return_value=_critique_response()),
        patch.object(validator, "validate_task", return_value=_validation_result()),
    ):
        result = composition.draft_with_self_critique(
            "img.jpg", max_rounds=1, max_feedback_retries=2, api_key="key", difficulty="challenging"
        )

    assert result is not None
    assert result["question"] == "Retry2?"
    feedbacks = [c.kwargs.get("feedback") for c in dqa.call_args_list if c.kwargs.get("feedback")]
    assert feedbacks[0] == "Errors to fix: Question too simple for Challenging difficulty"


def test_all_retries_invalid_returns_best_candidate():
    """When all feedback retries fail, the best-scoring candidate is returned."""
    from arxiv_manager.authoring import validator
    from arxiv_manager.authoring.ai_draft import composition

    retry_returns = [_draft("Retry1?", "2", valid=False, quality=60.0)]

    def fake_draft_qa(**kwargs):
        if kwargs.get("feedback"):
            if retry_returns:
                return retry_returns.pop(0)
            return _draft("Retry2?", "9", valid=False, quality=10.0)
        return _draft(valid=False, quality=50.0)

    with (
        patch.object(composition, "draft_qa", side_effect=fake_draft_qa),
        patch.object(composition, "_call_opencode", return_value=_critique_response()),
        patch.object(validator, "validate_task", return_value=_validation_result()),
    ):
        result = composition.draft_with_self_critique(
            "img.jpg", max_rounds=1, max_feedback_retries=2, api_key="key", difficulty="challenging"
        )

    assert result is not None
    assert result["question"] == "Retry1?"
    assert result["_validation_is_valid"] is False
