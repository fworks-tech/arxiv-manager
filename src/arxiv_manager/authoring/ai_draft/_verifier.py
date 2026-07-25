"""Draft verification — asks the model to check its own answer."""
from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

from .._draft_config import CONFIG
from .._draft_telemetry import log_draft
from ._api_client import _call_opencode, _get_api_key
from .._draft_prompts import VERIFY_PROMPT

logger = logging.getLogger(__name__)


def verify_draft(
    image_path: str | Path,
    draft: dict,
    api_key: str | None = None,
    model: str | None = None,
    media_type: str = "image/jpeg",
) -> dict | None:
    """Verify a draft by asking the model to check its own answer.

    If verification fails, returns the original draft.
    """
    from PIL import Image

    if not api_key:
        api_key = _get_api_key()
    if not api_key:
        return draft

    prompt = VERIFY_PROMPT.text.format(
        question=draft.get("question", ""),
        answer=draft.get("answer", ""),
    )

    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(CONFIG.thumbnail_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    media_type = "image/jpeg"

    start = time.time()
    try:
        verified = _call_opencode(
            api_key, prompt, b64, model=model,
            difficulty="", retries=1, media_type=media_type,
        )
    except Exception:
        verified = None

    elapsed = time.time() - start
    if verified and all(k in verified for k in ("question", "answer", "answer_format", "task_type")):
        log_draft(
            model=model or CONFIG.default_model, ok=True,
            elapsed=elapsed, difficulty="verify",
            figure_type="", figure_path=str(image_path), error="",
        )
        return verified
    log_draft(
        model=model or CONFIG.default_model, ok=False,
        elapsed=elapsed, difficulty="verify",
        figure_type="", figure_path=str(image_path),
        error="verify_failed_kept_original",
    )
    return draft
