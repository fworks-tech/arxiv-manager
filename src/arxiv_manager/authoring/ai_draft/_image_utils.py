"""Shared image encoding utilities for LLM consumption."""

from __future__ import annotations

import base64 as _b64
import io as _io
from pathlib import Path

from PIL import Image as PILImage


def encode_image_for_llm(
    image_path: Path,
    thumbnail_size: tuple[int, int] = (1024, 1024),
    jpeg_quality: int = 85,
) -> tuple[str, str]:
    """Encode image to base64 JPEG for LLM consumption.

    Args:
        image_path: Path to the image file.
        thumbnail_size: Maximum dimensions for thumbnail (width, height).
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (base64_string, media_type).
    """
    with PILImage.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(thumbnail_size)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return _b64.b64encode(buf.getvalue()).decode(), "image/jpeg"
