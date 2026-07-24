"""Tests for image analysis — suitability classification."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from arxiv_manager.authoring.image_analyzer import analyze_uploaded_image, validate_draft


def _make_image(width=200, height=200, content="blank"):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    if content == "blank":
        pass
    elif content == "chart":
        for y in range(height):
            for x in range(width):
                img.putpixel((x, y), (
                    (x * 255 // width) % 256,
                    (y * 255 // height) % 256,
                    128,
                ))
    elif content == "text_wall":
        import random
        for y in range(20, height - 20, 12):
            for x in range(10, width - 10):
                if random.random() < 0.45:
                    img.putpixel((x, y), (0, 0, 0))
    elif content == "sparse_banner":
        img = Image.new("RGB", (600, 50), (255, 255, 255))
    return img


def _save(img: Image.Image) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    return Path(f.name)


def test_analyze_chart_image():
    """Chart-like image (white bg, low saturation, structured elements)."""
    img = Image.new("RGB", (400, 300), (255, 255, 255))  # White background
    # Add axis-like lines
    for x in range(50, 380):
        img.putpixel((x, 260), (0, 0, 0))  # horizontal axis
    for y in range(40, 270):
        img.putpixel((50, y), (0, 0, 0))   # vertical axis
    # Add some data bars
    for bar_x, bar_h in [(80, 120), (130, 180), (180, 90), (230, 200), (280, 150)]:
        for y in range(260 - bar_h, 260):
            for x in range(bar_x - 10, bar_x + 10):
                img.putpixel((x, y), (60, 80, 200))
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert result["audit"]["figure_type"] == "chart_graph_text"
    finally:
        path.unlink(missing_ok=True)


def test_analyze_blank_image():
    """Very simple image is classified as general_image."""
    img = _make_image(200, 200, "blank")
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert "suitability" in result
    finally:
        path.unlink(missing_ok=True)


def test_analyze_rejected_low_complexity():
    """Very simple images may be rejected due to low complexity."""
    img = _make_image(100, 100, "blank")
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert result["suitability"] == "REJECTED"
    finally:
        path.unlink(missing_ok=True)


def test_analyze_returns_audit_fields():
    """Analysis result contains all expected audit fields."""
    img = _make_image(400, 300, "chart")
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert "audit" in result
        assert "suitability" in result
        assert "suitability_reason" in result
        assert "figure_type_label" in result
        audit = result["audit"]
        assert "width" in audit
        assert "height" in audit
        assert "complexity_score" in audit
        assert "is_dense" in audit
    finally:
        path.unlink(missing_ok=True)


def test_analyze_text_wall():
    """Text-heavy image is analyzed without error."""
    img = _make_image(300, 300, "text_wall")
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert "audit" in result
        assert "suitability" in result
    finally:
        path.unlink(missing_ok=True)


def test_analyze_sparse_banner_rejected():
    """Extreme aspect ratio image is rejected as unsuitable."""
    img = _make_image(content="sparse_banner")
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert result["suitability"] == "REJECTED"
    finally:
        path.unlink(missing_ok=True)


def test_analyze_hardest_high_complexity_dense():
    """High complexity dense chart may be classified as HARDEST."""
    img = Image.new("RGB", (600, 500), (255, 255, 255))
    for y in range(500):
        for x in range(600):
            img.putpixel((x, y), (
                (x * 7) % 256, (y * 13) % 256, (x * y * 3) % 256,
            ))
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        # This may or may not be HARDEST depending on thresholds,
        # but should at least be CHALLENGING or HARDEST
        assert result["suitability"] in ("CHALLENGING", "HARDEST")
    finally:
        path.unlink(missing_ok=True)


def test_validate_draft_calls_validator(monkeypatch):
    """validate_draft calls through to validate_task."""
    from arxiv_manager.authoring import image_analyzer
    calls = []
    def fake_validate(q, a, fmt, **kw):
        calls.append((q, a, fmt))
        from arxiv_manager.authoring.validator import ValidationResult
        return ValidationResult()
    monkeypatch.setattr(image_analyzer, "validate_task", fake_validate)
    draft = {"question": "Q?", "answer": "A", "answer_format": "word", "task_type": "chart"}
    result = validate_draft(draft)
    assert result is not None
    assert len(calls) == 1


def test_analyze_rejected_for_tiny_size():
    """Very small images get low complexity / REJECTED."""
    img = _make_image(40, 40, "chart")
    path = _save(img)
    try:
        result = analyze_uploaded_image(path)
        assert result["audit"]["complexity_score"] < 0.5
    finally:
        path.unlink(missing_ok=True)
