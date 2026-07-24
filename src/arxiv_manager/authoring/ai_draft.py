"""AI-assisted Q&A drafting using LLM."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path

from ._draft_config import CONFIG
from ._draft_prompts import (
    CHALLENGING_PROMPT,
    DRAFT_PROMPT,
    EASY_PROMPT,
    HARDEST_PROMPT,
    REGEN_PROMPT,
    SELF_CRITIQUE_PROMPT,
    SPATIAL_CHALLENGING_PROMPT,
    SPATIAL_DRAFT_PROMPT,
    SPATIAL_HARDEST_PROMPT,
    SPATIAL_REGEN_PROMPT,
    VERIFY_PROMPT,
)
from ._draft_telemetry import log_draft

logger = logging.getLogger(__name__)


def draft_qa(
    image_path: str | Path,
    paper_title: str = "",
    caption: str = "",
    task_type_hint: str = "",
    provider: str = "opencode",
    model: str | None = None,
    api_key: str | None = None,
    feedback: str = "",
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    previous_question: str = "",
) -> dict | None:
    """Draft a Q&A pair from an image using an LLM."""
    logger.info("draft_qa entry image=%s provider=%s difficulty=%s figure_type=%s complexity=%.3f",
                image_path, provider, difficulty, figure_type, complexity_score)

    if not api_key:
        api_key = _get_api_key(provider)
    if not api_key:
        logger.warning("draft_qa: no api key for provider=%s", provider)
        return None

    from PIL import Image

    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(CONFIG.thumbnail_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_media_type = "image/jpeg"

    is_spatial = figure_type == "general_image"
    if difficulty == "hardest":
        prompt = SPATIAL_HARDEST_PROMPT if is_spatial else HARDEST_PROMPT
    elif difficulty == "challenging":
        prompt = SPATIAL_CHALLENGING_PROMPT if is_spatial else CHALLENGING_PROMPT
    elif difficulty == "easy":
        prompt = SPATIAL_DRAFT_PROMPT if is_spatial else EASY_PROMPT
    elif feedback:
        prompt = (SPATIAL_REGEN_PROMPT if is_spatial else REGEN_PROMPT).format(feedback=feedback)
    else:
        prompt = SPATIAL_DRAFT_PROMPT if is_spatial else DRAFT_PROMPT
    if caption:
        prompt += f"\nCaption: {caption}"
    if figure_type:
        prompt += f"\nFigure type: {figure_type} (chart_graph_text = scientific chart/plot/diagram; general_image = photo/scene)"
    if complexity_score > 0:
        prompt += f"\nFigure complexity: {complexity_score:.2f}/1.0 (higher = more complex, candidate for hard multi-step counting)"
    if task_type_hint:
        prompt += f"\nType: {task_type_hint}"
    if previous_question:
        prompt += f"\n\nThe previous question for this image was: {previous_question}\nGenerate a SUBSTANTIALLY DIFFERENT question — different strategy, different data references, different answer. Do NOT reuse the same pattern (e.g., if previous was ratio of minima, use threshold counting or cross-panel sum instead)."

    model_id = model or CONFIG.default_model
    start = time.time()
    result: dict | None = None
    try:
        time.sleep(1)
        result = _call_opencode(api_key, prompt, b64, model, difficulty=difficulty, media_type=image_media_type)
        ok = result is not None
    except Exception as e:
        ok = False
        result = None
        error = str(e)[:100]

    elapsed = time.time() - start
    error_msg = error if 'error' in dir() else ""
    log_draft(
        model=model_id, ok=ok, elapsed=elapsed,
        difficulty=difficulty, figure_type=figure_type or "",
        figure_path=str(image_path), error=error_msg,
    )
    return result


def _get_api_key(provider: str) -> str | None:
    """Get API key from environment."""
    return os.environ.get("OPENCODE_API_KEY") if provider == "opencode" else None


def _call_opencode(api_key: str, prompt: str, b64_image: str, model: str | None = None, retries: int | None = None, difficulty: str = "", media_type: str = "image/jpeg") -> dict | None:
    """Call OpenCode Go API (OpenAI-compatible) with image.

    Default model: minimax-m3 (selected after A/B test on fresh figures).
    A/B results (5 fresh figures, 2026-07-23):
      - kimi-k2.7-code: 86/100 avg quality, 32.9s avg, 3/5 valid
      - minimax-m3:     98/100 avg quality, 23.0s avg, 5/5 valid (after <think> fix)
    minimax-m3 is ~30% faster, 14% higher quality.

    Vision models tested on OpenCode Go:
    - minimax-m3: Default. Emits <think> blocks (auto-stripped). ✅
    - kimi-k2.7-code: Works but slower + lower quality on this task. ✅
    - mimo-v2.5: Returns empty content, only reasoning ❌
    - glm-5.2: No vision support ❌

    Note: kimi-k2.7-code uses extensive thinking tokens. For "hardest" difficulty,
    uses higher max_tokens (32000) to allow reasoning + output.
    """
    import httpx

    model_id = model or CONFIG.default_model
    retries = retries or CONFIG.retries
    is_hard = difficulty in ("hardest", "challenging")
    cfg = CONFIG.get_model_config(model_id)
    max_tokens = cfg.max_tokens_hard if is_hard else cfg.max_tokens_easy
    timeout = cfg.timeout_hard if is_hard else cfg.timeout_easy

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)
        try:
            resp = httpx.post(
                CONFIG.api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                    "url": f"data:{media_type};base64,{b64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if content.strip():
                parsed = _parse_llm_response(content)
                if parsed:
                    return parsed
                logger.warning("_call_opencode: parsing returned None despite content (len=%d, preview=%.150s)",
                               len(content), content[:150])
                continue
        except Exception:
            if attempt == retries - 1:
                raise

    return None


def _parse_llm_response(text: str | None) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks and <think> tags.

    Some models (e.g. minimax-m3) wrap reasoning in <think>...</think> blocks
    inside the content field. We strip those before attempting JSON parse.

    Accepts partial JSON — only question + answer are truly required;
    answer_format defaults to "number", task_type defaults to "chart".
    """
    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _parse_candidate(candidate: str) -> dict | None:
        try:
            data = json.loads(candidate)
            if "question" not in data or "answer" not in data:
                return None
            data.setdefault("answer_format", "number")
            data.setdefault("task_type", "chart")
            return data
        except json.JSONDecodeError:
            return None

    data = _parse_candidate(text)
    if data:
        return data

    start = text.find('{')
    if start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    if '"question"' in candidate:
                        data = _parse_candidate(candidate)
                        if data:
                            return data
                    break

    logger.warning("_parse_llm_response: could not parse text (len=%d, preview=%.200s)", len(text), text[:200])
    return None


def verify_draft(
    image_path: str | Path,
    draft: dict,
    provider: str = "opencode",
    api_key: str | None = None,
    model: str | None = None,
    media_type: str = "image/jpeg",
) -> dict | None:
    """Verify a draft by asking the model to check its own answer.

    If verification fails (no parse, invalid), returns the original draft.
    Logs the verification result to telemetry.
    """
    from PIL import Image

    if not api_key:
        api_key = _get_api_key(provider)
    if not api_key:
        return draft

    prompt = VERIFY_PROMPT.format(
        question=draft.get("question", ""),
        answer=draft.get("answer", ""),
    )

    img = Image.open(image_path)
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
            figure_type="", figure_path=str(image_path),
            error="",
        )
        return verified
    log_draft(
        model=model or CONFIG.default_model, ok=False,
        elapsed=elapsed, difficulty="verify",
        figure_type="", figure_path=str(image_path),
        error="verify_failed_kept_original",
    )
    return draft


def draft_qa_consensus(
    image_path: str | Path,
    n_attempts: int = 3,
    verify: bool = True,
    provider: str = "opencode",
    api_key: str | None = None,
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    caption: str = "",
    **kwargs,
) -> dict | None:
    """Draft Q&A with multi-attempt consensus + optional verification.

    1. Generates n_attempts drafts (all use same prompt/difficulty).
    2. Each draft is scored by valid JSON + validator quality score.
    3. Best draft is selected.
    4. If verify=True, a verification pass checks the answer against image.
    5. Returns best verified draft, or None if all attempts failed.
    """
    if not api_key:
        api_key = _get_api_key(provider)
    if not api_key:
        return None

    attempts: list[tuple[dict, float]] = []
    last_feedback = ""

    for i in range(n_attempts):
        draft = draft_qa(
            image_path=image_path,
            provider=provider,
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
            draft["question"], draft["answer"], draft.get("answer_format", "word"),
        )
        score = (
            v.quality_score
            + (50 if v.is_valid else 0)
            + (10 if v.quality_score >= 80 else 0)
        )
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
        verified = verify_draft(
            image_path, best,
            provider=provider, api_key=api_key,
        )
        if verified:
            from .validator import validate_task as _validate
            v_verified = _validate(
                verified["question"], verified["answer"],
                verified.get("answer_format", "word"),
            )
            if v_verified.is_valid:
                return verified

    return best


def draft_with_self_critique(
    image_path: str | Path,
    max_rounds: int = 2,
    provider: str = "opencode",
    model: str | None = None,
    api_key: str | None = None,
    difficulty: str = "",
    figure_type: str = "",
    complexity_score: float = 0.0,
    caption: str = "",
    previous_question: str = "",
) -> dict | None:
    """Draft a Q&A pair and self-critique the question's difficulty.

    Flow:
    1. Generate initial draft via draft_qa().
    2. Call model again with the draft + image: rate 1-5 (would Qwen fail?).
    3. If score < 4, use the model's rewrite.
    4. Repeat up to max_rounds.
    """
    logger.info("self_critique entry difficulty=%s figure_type=%s max_rounds=%d", difficulty, figure_type, max_rounds)

    if not api_key:
        api_key = _get_api_key(provider)
    if not api_key:
        logger.warning("self_critique: no api key for provider=%s", provider)
        return None

    from PIL import Image
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(CONFIG.thumbnail_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    draft = draft_qa(
        image_path=image_path,
        provider=provider,
        model=model,
        api_key=api_key,
        difficulty=difficulty,
        figure_type=figure_type,
        complexity_score=complexity_score,
        caption=caption,
        previous_question=previous_question,
    )
    if not draft:
        logger.warning("self_critique: initial draft failed")
        return None

    for round_idx in range(max_rounds):
        prompt = SELF_CRITIQUE_PROMPT.format(
            question=draft["question"],
            answer=draft["answer"],
        )

        try:
            critique = _call_opencode(
                api_key, prompt, b64, model, retries=2, difficulty=difficulty,
                media_type="image/jpeg",
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
        }
        logger.info("self_critique: applied rewrite round=%d", round_idx)

    return draft
