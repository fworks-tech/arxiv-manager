"""Draft telemetry logging to JSONL."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_TELEMETRY_PATH: Path | None = None


def _get_telemetry_path() -> Path:
    """Lazy-initialize and return the telemetry file path."""
    global _TELEMETRY_PATH
    if _TELEMETRY_PATH is None:
        from ..storage import STORAGE_DIR
        _TELEMETRY_PATH = STORAGE_DIR / "_draft_telemetry.jsonl"
    return _TELEMETRY_PATH


def log_draft(
    model: str,
    ok: bool,
    elapsed: float,
    difficulty: str,
    figure_type: str,
    figure_path: str,
    error: str = "",
):
    """Append a draft attempt to the telemetry log (JSONL)."""
    record = {
        "ts": datetime.now().isoformat(),
        "model": model,
        "ok": ok,
        "elapsed_s": round(elapsed, 1),
        "difficulty": difficulty,
        "figure_type": figure_type,
        "figure_path": figure_path,
    }
    if error:
        record["error"] = error[:100]
    try:
        with open(str(_get_telemetry_path()), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
