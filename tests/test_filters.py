"""Tests for figure filters: is_text_only, is_likely_sparse."""

from pathlib import Path
from PIL import Image
import io
import random
import tempfile

from arxiv_manager.sourcing.filters import is_text_only, is_likely_sparse, audit_figure


def _make_image(width=200, height=200, content="blank"):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    if content == "blank":
        pass
    elif content == "sparse_banner":
        img = Image.new("RGB", (600, 50), (255, 255, 255))
    elif content == "large_chart":
        # Varied colored pixels like a chart
        for y in range(height):
            for x in range(width):
                img.putpixel((x, y), (
                    (x * 255 // width) % 256,
                    (y * 255 // height) % 256,
                    128,
                ))
    elif content == "text_wall":
        # Dense dark pixels in horizontal bands simulating text paragraphs
        for y in range(20, height - 20, 12):
            for x in range(10, width - 10):
                if random.random() < 0.45:
                    img.putpixel((x, y), (0, 0, 0))
    return img


def _save_temp(img) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(f.name)
    return Path(f.name)


# ─── is_text_only ──────────────────────────────────────────────────


def test_text_only_blank_is_text():
    """Blank image (mostly white) detected as text-only."""
    img = _make_image(content="blank")
    path = _save_temp(img)
    try:
        assert is_text_only(path) is True
    finally:
        path.unlink(missing_ok=True)


def test_text_only_chart_is_not_text():
    """Chart with varied color is NOT text-only."""
    img = _make_image(width=300, height=300, content="large_chart")
    path = _save_temp(img)
    try:
        assert is_text_only(path) is False
    finally:
        path.unlink(missing_ok=True)


# ─── is_likely_sparse ──────────────────────────────────────────────


def test_sparse_banner():
    """Aspect ratio > 5:1 is sparse."""
    img = _make_image(content="sparse_banner")
    path = _save_temp(img)
    try:
        assert is_likely_sparse(path) is True
    finally:
        path.unlink(missing_ok=True)


def test_sparse_allows_chart():
    """Normal chart is not sparse."""
    img = _make_image(width=300, height=300, content="large_chart")
    path = _save_temp(img)
    try:
        assert is_likely_sparse(path) is False
    finally:
        path.unlink(missing_ok=True)


# ─── audit_figure integration ──────────────────────────────────────


def test_audit_includes_fields():
    """audit_figure output includes is_text_only and is_suitable."""
    img = _make_image(width=500, height=400, content="large_chart")
    path = _save_temp(img)
    try:
        audit = audit_figure(path)
        assert "is_text_only" in audit
        assert "is_suitable" in audit
        assert "is_dense" in audit
        assert "is_likely_sparse" in audit
        assert "filesize_bytes" in audit
        assert audit["filesize_bytes"] > 0
    finally:
        path.unlink(missing_ok=True)
