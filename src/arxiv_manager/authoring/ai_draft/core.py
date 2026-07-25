"""Core drafting function — prompt building, history injection, guardrails."""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from pathlib import Path

from .._draft_config import CONFIG
from .._draft_prompts import (
    CHALLENGING_PROMPT, DRAFT_PROMPT, EASY_PROMPT, HARDEST_PROMPT,
    REGEN_PROMPT, SPATIAL_CHALLENGING_PROMPT, SPATIAL_DRAFT_PROMPT,
    SPATIAL_HARDEST_PROMPT, SPATIAL_REGEN_PROMPT,
)
from .._draft_telemetry import log_draft
from ._api_client import _call_opencode, _get_api_key

logger = logging.getLogger(__name__)


def draft_qa(
    image_path: str | Path,
    paper_title: str = "",
    caption: str = "",
    task_type_hint: str = "",
    model: str | None = None,
    api_key: str | None = None,
    feedback: str = "",
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    previous_question: str = "",
    validation_context: str = "",
    figure_id: int | None = None,
    use_rag: bool = True,
) -> dict | None:
    """Draft a Q&A pair from an image using an LLM."""
    logger.info("draft_qa entry image=%s difficulty=%s figure_type=%s complexity=%.3f",
                image_path, difficulty, figure_type, complexity_score)

    if not api_key:
        api_key = _get_api_key()
    if not api_key:
        logger.warning("draft_qa: no api key set")
        return None

    from PIL import Image

    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(CONFIG.thumbnail_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_media_type = "image/jpeg"

    is_spatial = figure_type == "general_image"
    if difficulty == "hardest":
        raw_template = SPATIAL_HARDEST_PROMPT if is_spatial else HARDEST_PROMPT
        prompt = raw_template.text
    elif difficulty == "challenging":
        raw_template = SPATIAL_CHALLENGING_PROMPT if is_spatial else CHALLENGING_PROMPT
        prompt = raw_template.text
    elif difficulty == "easy":
        raw_template = SPATIAL_DRAFT_PROMPT if is_spatial else EASY_PROMPT
        prompt = raw_template.text
    elif feedback:
        raw_template = SPATIAL_REGEN_PROMPT if is_spatial else REGEN_PROMPT
        prompt = raw_template.text.format(feedback=feedback)
    else:
        raw_template = SPATIAL_DRAFT_PROMPT if is_spatial else DRAFT_PROMPT
        prompt = raw_template.text
    if caption:
        prompt += f"\nCaption: {caption}"
    if figure_type:
        prompt += f"\nFigure type: {figure_type} (chart_graph_text = scientific chart/plot/diagram; general_image = photo/scene)"
    if complexity_score > 0:
        prompt += f"\nFigure complexity: {complexity_score:.2f}/1.0 (higher = more complex, candidate for hard multi-step counting)"
    if task_type_hint:
        prompt += f"\nType: {task_type_hint}"

    from .._history_context import inject_history_into_prompt, select_best_model
    if model is None:
        model = select_best_model(
            figure_type=figure_type,
            difficulty=difficulty,
            default_model=CONFIG.default_model,
            allowed_models=CONFIG.vision_models,
        )
    prompt = inject_history_into_prompt(
        prompt,
        figure_id=figure_id,
        figure_type=figure_type,
        difficulty=difficulty,
        complexity_score=complexity_score,
        task_type=task_type_hint,
        previous_question=previous_question,
        validation_context=validation_context,
    )

    # 3. RAG context injection (Phase 3, lazy singleton)
    # Skip RAG on retry (feedback set) to give the model a simpler prompt
    if use_rag and not feedback and figure_id is not None:
        try:
            from ...services.rag_pipeline import get_pipeline as _get_rag
            rag_ctx = _get_rag().get_context(
                query=prompt,
                figure_id=figure_id,
                figure_type=figure_type,
                difficulty=difficulty,
            )
            if rag_ctx["context_str"]:
                # Truncate RAG context to at most 2000 chars to avoid prompt overflow
                ctx = rag_ctx["context_str"]
                if len(ctx) > 2000:
                    ctx = ctx[:1997] + "..."
                prompt += "\n\n" + ctx
        except Exception as e:
            logger.warning("draft_qa: RAG context injection failed: %s", e)

    prompt_text_hash = hashlib.sha256(prompt.encode()).hexdigest()[:20]
    prompt_version_id = f"{raw_template.name}@{prompt_text_hash[:12]}"

    model_id = model or CONFIG.default_model
    start = time.time()
    result: dict | None = None
    try:
        result = _call_opencode(api_key, prompt, b64, model, difficulty=difficulty, media_type=image_media_type)
        ok = result is not None
    except ValueError:
        raise
    except Exception as e:
        ok = False
        result = None
        _err = str(e)[:100]

    if result is not None:
        if not result.get("question", "").strip() or not result.get("answer", "").strip():
            logger.warning("draft_qa: empty Q&A after API call (question=%r answer=%r raw=%.300s)",
                           result.get("question", ""), result.get("answer", ""),
                           result.get("_raw_response", "")[:300])
            result = None
            ok = False

    if result is not None:
        result["_prompt_version_id"] = prompt_version_id
        result["_prompt_text_hash"] = prompt_text_hash

        # Skip guardrail checks on retry (feedback set) to prevent
        # unbounded mutual recursion: draft_qa -> run_guardrails -> _auto_retry -> draft_qa
        if not feedback:
            from .._guardrails import run_guardrails
            from ..validator import validate_task as _validate

            v = _validate(
                result.get("question", ""),
                result.get("answer", ""),
                result.get("answer_format", "word"),
                figure_type=figure_type,
                task_type=result.get("task_type", "chart"),
            )
            guardrail_context = {
                "validation_result": {
                    "quality_score": v.quality_score,
                    "errors": v.errors,
                    "warnings": v.warnings,
                },
                "min_quality": 30,
                "api_key": api_key,
                "difficulty": difficulty,
                "figure_type": figure_type,
                "complexity_score": complexity_score,
                "previous_question": previous_question,
                "figure_id": figure_id,
                "model": model,
            }
            result = run_guardrails(
                result,
                guardrail_context,
                api_key=api_key,
                image_path=str(image_path),
                max_retries=1,
                draft_qa_callback=draft_qa,
            )
        if result is not None:
            v2 = _validate(
                result.get("question", ""),
                result.get("answer", ""),
                result.get("answer_format", "word"),
                figure_type=figure_type,
                task_type=result.get("task_type", "chart"),
            )
            result["_validation_quality"] = v2.quality_score
            result["_validation_is_valid"] = v2.is_valid
            result["_validation_errors"] = v2.errors
            result["_validation_warnings"] = v2.warnings

    elapsed = time.time() - start
    error_msg = _err if '_err' in dir() else ""
    log_draft(
        model=model_id, ok=ok, elapsed=elapsed,
        difficulty=difficulty, figure_type=figure_type or "",
        figure_path=str(image_path), error=error_msg,
    )
    return result
