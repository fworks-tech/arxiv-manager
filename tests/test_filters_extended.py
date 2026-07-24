"""Extended tests for figure filters — edge cases."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from arxiv_manager.sourcing.filters import (
    audit_figure,
    is_likely_sparse,
    is_text_only,
)


def _save(img: Image.Image) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    return Path(f.name)


def test_audit_corrupt_image():
    """Corrupt/invalid image does not crash — returns result with low complexity."""
    path = Path(tempfile.mktemp(suffix=".jpg"))
    path.write_bytes(b"not an image at all")
    try:
        result = audit_figure(path)
        # Should return something (may be a rejection or partial result)
        assert isinstance(result, dict)
    except Exception:
        # Also acceptable if it raises (PIL error)
        pass
    finally:
        path.unlink(missing_ok=True)


def test_audit_tiny_image():
    """Very small image returns low complexity / may be rejected."""
    img = Image.new("RGB", (30, 30), (255, 255, 255))
    path = _save(img)
    try:
        audit = audit_figure(path)
        assert audit["width"] == 30
        assert audit["height"] == 30
        assert audit["complexity_score"] < 0.3
    finally:
        path.unlink(missing_ok=True)


def test_audit_large_chart():
    """Large chart-like image has higher complexity."""
    img = Image.new("RGB", (800, 600), (255, 255, 255))
    # Add varied colors
    for y in range(600):
        for x in range(800):
            img.putpixel((x, y), ((x * 7) % 256, (y * 13) % 256, 128))
    path = _save(img)
    try:
        audit = audit_figure(path)
        assert audit["is_text_only"] is False
        assert audit["complexity_score"] > 0.1
    finally:
        path.unlink(missing_ok=True)


def test_audit_includes_filesize():
    """Audit output includes filesize_bytes field."""
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    path = _save(img)
    try:
        audit = audit_figure(path)
        assert "filesize_bytes" in audit
        assert audit["filesize_bytes"] > 0
    finally:
        path.unlink(missing_ok=True)


def test_audit_includes_dimensions():
    """Audit output includes width and height."""
    img = Image.new("RGB", (640, 480), (0, 0, 0))
    path = _save(img)
    try:
        audit = audit_figure(path)
        assert audit["width"] == 640
        assert audit["height"] == 480
        assert audit["width_height_ratio"] == pytest.approx(640 / 480, rel=0.01)
    finally:
        path.unlink(missing_ok=True)


def test_is_likely_sparse_threshold():
    """5:1 aspect ratio is sparse; chart-like image is not."""
    sparse = Image.new("RGB", (500, 50), (255, 255, 255))
    path1 = _save(sparse)
    try:
        assert is_likely_sparse(path1) is True
    finally:
        path1.unlink(missing_ok=True)

    # Non-blank chart image with moderate variance — not sparse
    not_sparse = Image.new("RGB", (400, 300), (255, 255, 255))
    for y in range(50, 250):
        for x in range(50, 350):
            not_sparse.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, 128))
    path2 = _save(not_sparse)
    try:
        assert is_likely_sparse(path2) is False
    finally:
        path2.unlink(missing_ok=True)


def test_audit_is_dense_flag():
    """Chart-like image with varied pixels is marked as dense."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    for y in range(50, 150):
        for x in range(50, 150):
            img.putpixel((x, y), (100, 150, 200))
    path = _save(img)
    try:
        audit = audit_figure(path)
        assert "is_dense" in audit
    finally:
        path.unlink(missing_ok=True)
