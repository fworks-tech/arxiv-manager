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

logger = logging.getLogger(__name__)


DRAFT_PROMPT = """Create a hard visual-reasoning question for this image that requires multiple steps of reasoning.

Authoring principles (from QA handbook):
- Prefer REASONING over recognition: compose multiple series, filter by attribute, compare across panels
- Add ONE intermediate reasoning step: combine two values and compute, or locate then compare
- Pin the answer down with constraints: specify which series, panel, axis, subset
- Make the answer exact-matchable: smallest unit, no restating, no units unless required
- Question must require the image — a smart person cannot answer from text alone
- The answer must be objective — two reasonable people give the same answer

Rules: English, 1 sentence (2 max for format spec), must need the image, no yes/no, no "how does" / "what trend" (explanation banned), no "none" / "cannot be determined" answer, answer is 1 word or 1 number. NO option restriction like "Out of the 3...".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""


EASY_PROMPT = """Create a straightforward visual-reasoning question for this image that is easy to answer.

The question should:
- Focus on a single clear element (count, color, shape, label, position)
- Require looking at the image, but only a simple observation
- Have an obvious, unambiguous answer of 1 word or 1 number
- Be answerable by anyone who can see the image — no multi-step reasoning
- Avoid counting large numbers (>10) or complex comparisons

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend".
Answer is 1 word or 1 number.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""


REGEN_PROMPT = """Create a hard visual-reasoning question for this image.
The previous attempt had validation errors — fix ALL of them:

{feedback}

Authoring principles (from QA handbook):
- Prefer REASONING over recognition: compose multiple series, filter by attribute, compare across panels
- Add ONE intermediate reasoning step: combine two values and compute, or locate then compare
- Pin the answer down with constraints
- Make the answer exact-matchable: smallest unit
- Question must require the image
- Answer must be objective

Rules: English, 1 sentence, must need the image, no yes/no, answer is 1 word or number. No option restriction. No "how does" / "what trend" / no "none" / "cannot be determined".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""


SPATIAL_REGEN_PROMPT = """Create a hard spatial-reasoning question for this natural image.
The previous attempt had validation errors — fix ALL of them:

{feedback}

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Authoring principles (from QA handbook §4 - Spatial Reasoning):
- Objects must be clearly visible and easy to name ("red mug", "white chair")
- Prefer real 3D layout cues: depth, occlusion, object size differences, foreground/background
- Spatial ambiguity should be LOW — two people must give the same answer

Use these spatial question types:
1. Viewer-centered left/right: "From the viewer's perspective, which object is immediately to the left of the laptop?"
2. Depth/distance: "Which object appears closest to the camera?"
3. Relative height: "Which object is positioned highest in the image?"
4. Containment/support: "What object is sitting on top of the microwave?"
5. Between/surrounded by: "Which object is between the sofa and the coffee table?"
6. Occlusion/in front: "Which object is partially blocking the view of the cabinet?"

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend". Answer is 1 word (object name) or number. No trick answers.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""


HARDEST_PROMPT = """Create an EXTREMELY HARD visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on.

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

Use these strategies:
1. Count many items (10+) — Qwen's counting degrades past ~7-9 items
2. Multi-step counting: "Count all [type A] and [type B], then give their sum"
3. Exclusion counting: "Excluding [subset], how many [elements] remain?"
4. Spatial classification: "How many [elements] touch/connect to [specific component]?"
5. Path tracing: "Trace path from [A] through [B] to [C]. How many [elements] does it cross?"

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- No yes/no, no "how does" / "what trend" / "explain"
- No "none" / "cannot be determined" answers
- No option restriction like "Out of the 3..."
- No domain jargon (no sp3, p-value, EBITDA, etc.)
- Answer is 1 word or 1 number (smallest possible unit)
- The answer must be an UNUSUAL number (not 2, 3, 4, 5) to avoid Qwen guessing

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""


CHALLENGING_PROMPT = """Create a CHALLENGING visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on but Gemini is likely to PASS.

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

PROVEN example (optical computing):
"Count these three types of elements in the diagram, excluding anything inside the blue dashed boxes: (1) black rectangular mirrors, (2) standalone gray ellipse lenses, and (3) groups of colored filter bars. Sum all three counts." → 18 (6 mirrors + 8 lenses + 4 filter groups) [Challenging: Qwen 0/4, Gemini 4/4]

Use these PROVEN Challenging strategies:
1. Multi-type count + sum: "Count these three types of elements, then give their sum: (1) [type A], (2) [type B], (3) [type C]"
2. Multi-type count + sum + exclusion: "Count [type A], [type B], and [type C] excluding anything inside [region]. Sum all three counts."
3. Subjective classification + count: "Count [elements] that contain color (not gray or blank). How many are there?"
4. Spatial targeting + count: "Count arrows entering red block X, blue block Y, and green block Z combined."
5. Threshold filter + count: "Count [elements] with [attribute] greater than [value] across all panels."
6. Cross-attribute filter + arithmetic: "How many more [type A] than [type B] are visible in the image?"
7. Spatial + count: "How many of the visible [objects] are in the back row vs. the front row?"

Even if the image looks simple or has few elements, FORCE a multi-step question — combine attributes, apply filters, or do arithmetic on counts. A simple question defeats the purpose. The author explicitly chose Challenging.

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- No yes/no, no "how does" / "what trend" / "explain"
- No "none" / "cannot be determined" answers
- No option restriction like "Out of the 3..."
- No domain jargon (no sp3, p-value, EBITDA, etc.)
- Answer is 1 word or 1 number (smallest possible unit)
- The answer must be an UNUSUAL number (not 2, 3, 4, 5) to avoid Qwen guessing
- The question must be CLEAR enough that a careful Gemini pass can solve it
- Add ONE intermediate reasoning step: combine two values, or locate then compare
- Pin the answer down with constraints: specify which series, panel, axis, subset

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""


# --- Figure-type specific prompts (Tier 2a) ---
# Used when figure_type = "general_image" (spatial scenes, photos).

SPATIAL_DRAFT_PROMPT = """Create a hard spatial-reasoning question for this natural image that requires looking at 3D layout, object positions, depth, or spatial relationships.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Authoring principles (from QA handbook §4 - Spatial Reasoning):
- Objects must be clearly visible and easy to name ("red mug", "white chair")
- Prefer real 3D layout cues: depth, occlusion, object size differences, foreground/background
- Non-trivial viewpoints: low-angle, top-down, egocentric, aisle view
- Spatial ambiguity should be LOW — two people must give the same answer

Use these spatial question types:
1. Viewer-centered left/right: "From the viewer's perspective, which object is immediately to the left of the laptop?"
2. Depth/distance: "Which object appears closest to the camera?"
3. Relative height: "Which object is positioned highest in the image?"
4. Containment/support: "What object is sitting on top of the microwave?"
5. Between/surrounded by: "Which object is between the sofa and the coffee table?"
6. Occlusion/in front: "Which object is partially blocking the view of the cabinet?"
7. Perspective switching: "Imagine you are sitting where the person is and facing the same direction. Which object would be on your right?"
8. 3D orientation: "Which vehicle is facing toward the camera?"
9. Navigation-style: "Starting from the doorway and moving toward the sink, which object would be on your right?"

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend". Answer is 1 word (object name) or number. No trick answers ("none" / "cannot be determined"). NO option restriction like "Out of the 3...".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""


SPATIAL_CHALLENGING_PROMPT = """Create a CHALLENGING spatial-reasoning question for this natural image that Qwen 3.6-35B-A3B will FAIL on but Gemini is likely to PASS.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Qwen's known weaknesses:
- ODInW13 (object detection/counting): 50.8 — weak at detecting many objects
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel spatial tasks
- RefSpatialBench: 64.3 — moderate at referenced spatial relations

Use these PROVEN spatial strategies for Challenging:
1. Multi-object exclusion: "Which is the third object to the right of X, excluding objects smaller than Y?"
2. Depth + attribute: "Which object furthest from the camera has the brightest color?"
3. Perspective switching + occlusion: "From the seated person's view, which object is partially blocked by the lamp?"
4. Multi-step containment: "What object sits on the surface that is between the plant and the window?"
5. Relative positioning with count: "Count how many objects are between the white chair and the blue table."

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- Objects must be clearly nameable ("red mug", "white chair")
- Spatial ambiguity must be LOW — one correct answer only
- No yes/no, no "how does" / "what trend" / "explain"
- No trick answers ("none" / "cannot be determined")
- Answer is 1 word (object name) or number
- The question must be CLEAR enough that a careful Gemini pass can solve it
- Add ONE intermediate step: locate then compare, or filter then name
- Pin the answer down with constraints

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""


SPATIAL_HARDEST_PROMPT = """Create an EXTREMELY HARD spatial-reasoning question for this natural image that Qwen 3.6-35B-A3B will FAIL on.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Qwen's known weaknesses:
- ODInW13: 50.8 — weak at detecting many objects
- ZEROBench_sub: 34.4 — weak at novel spatial tasks

Strategies:
1. Count objects across depth planes: "How many objects are closer to the camera than the red chair?"
2. Spatial paths: "Starting from the doorway, which object would you pass third before reaching the sink?"
3. Occlusion counting: "How many objects are partially hidden by the table?"
4. Multi-step filtering: "Among objects on the top shelf, which is to the left of the blue vase and darker than the gray box?"

Rules: English, 1 sentence (2 max for format spec), must need the image. No yes/no. Answer is 1 word or number.
No trick answers. Answer must be UNUSUAL number (not 2, 3, 4, 5).
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""


_TELEMETRY_PATH = None


def _log_draft(
    model: str,
    ok: bool,
    elapsed: float,
    difficulty: str,
    figure_type: str,
    figure_path: str,
    error: str = "",
):
    """Append a draft attempt to the telemetry log (JSONL)."""
    global _TELEMETRY_PATH
    if _TELEMETRY_PATH is None:
        from ..storage import STORAGE_DIR
        _TELEMETRY_PATH = STORAGE_DIR / "_draft_telemetry.jsonl"
    import json
    from datetime import datetime
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
        with open(str(_TELEMETRY_PATH), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


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
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
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

    model_id = model or "minimax-m3"
    start = time.time()
    result: dict | None = None
    try:
        if provider == "opencode":
            time.sleep(1)
            result = _call_opencode(api_key, prompt, b64, model, difficulty=difficulty, media_type=image_media_type)
        elif provider == "openai":
            result = _call_openai(api_key, prompt, b64, model, media_type=image_media_type)
        elif provider == "anthropic":
            result = _call_anthropic(api_key, prompt, b64, model, media_type=image_media_type)
        ok = result is not None
    except Exception as e:
        ok = False
        result = None
        error = str(e)[:100]

    elapsed = time.time() - start
    error_msg = error if 'error' in dir() else ""
    _log_draft(
        model=model_id, ok=ok, elapsed=elapsed,
        difficulty=difficulty, figure_type=figure_type or "",
        figure_path=str(image_path), error=error_msg,
    )
    return result


def _get_api_key(provider: str) -> str | None:
    """Get API key from environment."""
    env_map = {
        "opencode": "OPENCODE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    return os.environ.get(env_map.get(provider, ""))


def _call_opencode(api_key: str, prompt: str, b64_image: str, model: str | None = None, retries: int = 3, difficulty: str = "", media_type: str = "image/jpeg") -> dict | None:
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

    model_id = model or "minimax-m3"
    # Per-model token/timeout tuning
    is_hard = difficulty in ("hardest", "challenging")
    if "kimi" in model_id.lower():
        max_tokens = 32000 if is_hard else 4000
        timeout = 300 if is_hard else 180
    elif "minimax" in model_id.lower() or "m3" in model_id.lower():
        max_tokens = 16000 if is_hard else 4000
        timeout = 240 if is_hard else 120
    else:
        max_tokens = 8000 if is_hard else 4000
        timeout = 180 if is_hard else 120

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)  # 2s, 4s backoff
        try:
            resp = httpx.post(
                "https://opencode.ai/zen/go/v1/chat/completions",
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
                return _parse_llm_response(content)
        except Exception:
            if attempt == retries - 1:
                raise

    return None


def _call_openai(api_key: str, prompt: str, b64_image: str, model: str | None = None, retries: int = 3, media_type: str = "image/jpeg") -> dict | None:
    """Call OpenAI API with image (with retry)."""
    import httpx

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model or "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{b64_image}",
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 4000,
                },
                timeout=90,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            result = _parse_llm_response(text)
            if result:
                return result
        except Exception:
            if attempt == retries - 1:
                raise
    return None


def _call_anthropic(api_key: str, prompt: str, b64_image: str, model: str | None = None, retries: int = 3, media_type: str = "image/jpeg") -> dict | None:
    """Call Anthropic API with image (with retry)."""
    import httpx

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model or "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_image,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            result = _parse_llm_response(text)
            if result:
                return result
        except Exception:
            if attempt == retries - 1:
                raise
    return None


def _parse_llm_response(text: str | None) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks and <think> tags.

    Some models (e.g. minimax-m3) wrap reasoning in <think>...</think> blocks
    inside the content field. We strip those before attempting JSON parse.
    """
    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Strip <think>...</think> blocks (some models emit reasoning in content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try parsing as-is
    try:
        data = json.loads(text)
        required = ["question", "answer", "answer_format", "task_type"]
        if all(k in data for k in required):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with balanced braces (handles nested objects)
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
                        try:
                            data = json.loads(candidate)
                            required = ["question", "answer", "answer_format", "task_type"]
                            if all(k in data for k in required):
                                return data
                        except json.JSONDecodeError:
                            pass
                    break

    return None  # end of _parse_llm_response


# ─── Verification pass (Tier 1) ────────────────────────────────────


VERIFY_PROMPT = """Look at this image carefully. The question and answer below were
drafted by AI. Verify if the answer is CORRECT based on the image, and fix
if wrong. Reply with corrected JSON only (same keys).

Draft question: {question}
Draft answer: {answer}"""


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
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    media_type = "image/jpeg"

    start = time.time()
    try:
        if provider == "opencode":
            verified = _call_opencode(
                api_key, prompt, b64, model=model,
                difficulty="", retries=1, media_type=media_type,
            )
        else:
            verified = None
    except Exception:
        verified = None

    elapsed = time.time() - start
    if verified and all(k in verified for k in ("question", "answer", "answer_format", "task_type")):
        _log_draft(
            model=model or "minimax-m3", ok=True,
            elapsed=elapsed, difficulty="verify",
            figure_type="", figure_path=str(image_path),
            error="",
        )
        return verified
    _log_draft(
        model=model or "minimax-m3", ok=False,
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

        # Build feedback for next iteration
        if v.errors or v.warnings:
            parts = []
            if v.errors:
                parts.append("Errors to fix: " + "; ".join(v.errors[:3]))
            if v.warnings:
                parts.append("Warnings to address: " + "; ".join(v.warnings[:3]))
            last_feedback = " | ".join(parts)
        else:
            last_feedback = ""  # Reset if last attempt was valid

    if not attempts:
        return None

    # Pick best by score (highest wins; ties → last seen, which incorporates feedback)
    best = max(attempts, key=lambda x: x[1])[0]

    # Verification pass
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
