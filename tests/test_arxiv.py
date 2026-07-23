"""Tests for arXiv search: query expansion."""

from arxiv_manager.sourcing.arxiv import _expand_terms


def test_expand_detection():
    """detection expands to detection, detecting, detector, yolo, faster r-cnn."""
    result = _expand_terms(["detection"])
    assert len(result) >= 3
    assert "detection" in result
    assert "detecting" in result
    assert "detector" in result


def test_expand_unknown():
    """Unknown term passes through unchanged."""
    result = _expand_terms(["quantum"])
    assert result == ["quantum"]


def test_expand_multiple_terms():
    """Multiple terms are both expanded; no duplicates."""
    result = _expand_terms(["detection", "optical"])
    assert "detection" in result
    assert "detector" in result
    assert "optical" in result
    assert "photon" in result
    # No duplicates
    assert len(result) == len(set(result))


def test_expand_capped_per_term():
    """Each term is capped at 6 expansions."""
    result = _expand_terms(["detection", "segmentation", "neural network"])
    assert len(result) >= 5
    assert len(result) <= 18  # 6 per term max


def test_expand_case_insensitive():
    """Case mismatch still expands."""
    result = _expand_terms(["Detection"])
    assert "detection" in result or "detecting" in result
    assert len(result) >= 3
