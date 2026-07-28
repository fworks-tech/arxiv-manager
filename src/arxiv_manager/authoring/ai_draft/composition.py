"""Multi-attempt consensus and self-critique drafting flows."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from .._draft_config import CONFIG
from .._draft_prompts import SELF_CRITIQUE_PROMPT
from ._api_client import _call_opencode, _get_api_key
from ._response_parser import _parse_critique_response
from .core import draft_qa

logger = logging.getLogger(__name__)


def draft_qa_consensus(
    image_path: str | Path,
    n_attempts: int = 3,
    verify: bool = True,
    api_key: str | None = None,
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    caption: str = "",
    **kwargs,
) -> dict | None:
    """Draft Q&A with multi-attempt consensus + optional verification."""
    if not api_key:
        api_key = _get_api_key()
    if not api_key:
        return None

    attempts: list[tuple[dict, float]] = []
    last_feedback = ""

    for i in range(n_attempts):
        draft = draft_qa(
            image_path=image_path,
            api_key=api_key,
            difficulty=difficulty,
            figure_type=figure_type,
            complexity_score=complexity_score,
            caption=caption,
            feedback=last_feedback,
            **kwargs,
        )
        if not draft:
            continue

        from .validator import validate_task as _validate

        v = _validate(
            draft["question"],
            draft["answer"],
            draft.get("answer_format", "word"),
        )
        score = v.quality_score + (50 if v.is_valid else 0) + (10 if v.quality_score >= 80 else 0)
        attempts.append((draft, score))

        if v.errors or v.warnings:
            parts = []
            if v.errors:
                parts.append("Errors to fix: " + "; ".join(v.errors[:3]))
            if v.warnings:
                parts.append("Warnings to address: " + "; ".join(v.warnings[:3]))
            last_feedback = " | ".join(parts)
        else:
            last_feedback = ""

    if not attempts:
        return None

    best = max(attempts, key=lambda x: x[1])[0]

    if verify:
        from ._verifier import verify_draft

        verified = verify_draft(image_path, best, api_key=api_key)
        if verified:
            from .validator import validate_task as _validate

            v_verified = _validate(
                verified["question"],
                verified["answer"],
                verified.get("answer_format", "word"),
            )
            if v_verified.is_valid:
                return verified

    return best


def draft_with_self_critique(
    image_path: str | Path,
    max_rounds: int = 2,
    model: str | None = None,
    api_key: str | None = None,
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    caption: str = "",
    previous_question: str = "",
    validation_context: str = "",
    figure_id: int | None = None,
    task_id: int | None = None,
) -> dict | None:
    """Draft a Q&A pair and self-critique the question's difficulty."""
    logger.info("self_critique entry difficulty=%s figure_type=%s max_rounds=%d", difficulty, figure_type, max_rounds)

    if not api_key:
        api_key = _get_api_key()
    if not api_key:
        logger.warning("self_critique: no api key set")
        return None

    from PIL import Image

    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(CONFIG.thumbnail_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    draft = draft_qa(
        image_path=image_path,
        model=model,
        api_key=api_key,
        difficulty=difficulty,
        figure_type=figure_type,
        complexity_score=complexity_score,
        caption=caption,
        previous_question=previous_question,
        validation_context=validation_context,
        figure_id=figure_id,
        task_id=task_id,
    )
    if draft is None:
        logger.warning("self_critique: initial draft failed")
        return None

    for round_idx in range(max_rounds):
        prompt = SELF_CRITIQUE_PROMPT.text.format(
            question=draft["question"],
            answer=draft["answer"],
        )

        try:
            critique = _call_opencode(
                api_key,
                prompt,
                b64,
                model,
                retries=2,
                difficulty=difficulty,
                media_type="image/jpeg",
                parser=_parse_critique_response,
            )
        except Exception as e:
            logger.warning("self_critique: model call failed round=%d err=%s", round_idx, str(e)[:100])
            break

        if not critique:
            break

        score = critique.get("score", 0)
        rewrite_q = critique.get("rewrite_question", "").strip()
        rewrite_a = critique.get("rewrite_answer", "").strip()
        logger.info("self_critique round=%d score=%d", round_idx, score)

        if score >= 4 or not rewrite_q or not rewrite_a:
            break

        rewrite_format = critique.get("answer_format", draft.get("answer_format", "word"))
        rewrite_type = critique.get("task_type", draft.get("task_type", "chart"))
        draft = {
            "question": rewrite_q,
            "answer": rewrite_a,
            "answer_format": rewrite_format,
            "task_type": rewrite_type,
            "_usage": draft.get("_usage") or critique.get("_usage"),
            "_raw_response": draft.get("_raw_response", ""),
            "_reasoning_trace": draft.get("_reasoning_trace", ""),
            "_model": draft.get("_model", ""),
            "_prompt_version_id": draft.get("_prompt_version_id", ""),
            "_prompt_text_hash": draft.get("_prompt_text_hash", ""),
        }
        logger.info("self_critique: applied rewrite round=%d", round_idx)

    return draft
