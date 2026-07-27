"""Draft telemetry logging to JSONL and generation history to DB."""

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
    except Exception as e:
        logger.warning("log_draft: write failed: %s", e)


def log_generation_attempt(
    *,
    figure_id: int | None = None,
    task_id: int | None = None,
    parent_attempt_id: int | None = None,
    attempt_number: int = 0,
    generation_type: str = "",
    source_route: str = "",
    prompt_template_name: str = "",
    prompt_text: str = "",
    prompt_text_hash: str = "",
    prompt_version_id: str = "",
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    previous_question: str = "",
    feedback_text: str = "",
    model_name: str = "",
    max_tokens: int = 0,
    timeout_s: int = 0,
    raw_response: str = "",
    reasoning_trace: str = "",
    generated_question: str = "",
    generated_answer: str = "",
    generated_answer_format: str = "",
    generated_task_type: str = "",
    validation_quality: float = 0.0,
    validation_is_valid: bool = False,
    validation_errors: str = "",
    validation_warnings: str = "",
    critique_score: int = 0,
    critique_rewrite_question: str = "",
    critique_rewrite_answer: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    success: bool = False,
    error_message: str = "",
    elapsed_ms: int = 0,
):
    """Persist a generation attempt to the database for traceability and learning.

    Accepts all fields as keyword arguments. Only figure_id is truly required
    at the DB schema level; everything else is optional.
    """
    if figure_id is None and task_id is None:
        logger.info("log_generation_attempt: no figure_id or task_id (draft on unproposed upload) — skipping DB write")
        return

    from ..db import get_session
    from ..models import GenerationAttempt

    session = get_session()
    try:
        record = GenerationAttempt(
            figure_id=figure_id,
            task_id=task_id,
            parent_attempt_id=parent_attempt_id,
            attempt_number=attempt_number,
            generation_type=generation_type,
            source_route=source_route,
            prompt_template_name=prompt_template_name,
            prompt_text=prompt_text[:10000],
            prompt_text_hash=prompt_text_hash,
            prompt_version_id=prompt_version_id,
            difficulty=difficulty,
            figure_type=figure_type,
            complexity_score=complexity_score,
            previous_question=previous_question[:2000],
            feedback_text=feedback_text[:2000],
            model_name=model_name,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            raw_response=raw_response[:10000],
            reasoning_trace=reasoning_trace[:5000],
            generated_question=generated_question,
            generated_answer=generated_answer,
            generated_answer_format=generated_answer_format,
            generated_task_type=generated_task_type,
            validation_quality=validation_quality,
            validation_is_valid=validation_is_valid,
            validation_errors=validation_errors[:2000],
            validation_warnings=validation_warnings[:2000],
            critique_score=critique_score,
            critique_rewrite_question=critique_rewrite_question[:2000],
            critique_rewrite_answer=critique_rewrite_answer[:2000],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            success=success,
            error_message=error_message[:500],
            elapsed_ms=elapsed_ms,
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.warning("log_generation_attempt: failed to persist: %s", e)
    finally:
        session.close()
