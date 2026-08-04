"""Answer determinism checking — independent sampled reads must agree with the golden answer.

A question/answer pair is trustworthy only if multiple independent reads of the
image converge on the golden answer. Text validation and fact-checking cannot
prove determinism; running the question through the vision model several times
with sampling is the strongest available signal that the answer is objectively
derivable ("two reasonable people give the same answer").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .._draft_config import CONFIG
from .._draft_prompts import CHECK_ANSWER_PROMPT
from ._api_client import _call_opencode
from ._image_utils import encode_image_for_llm

logger = logging.getLogger(__name__)

_VERIFY_SEMANTIC_PROMPT = (
    'Compare these two short answers to the same visual question. '
    'Return ONLY valid JSON: {{"match": true or false, "explanation": "one line"}}\n'
    "Answer 1: {a}\nAnswer 2: {b}"
)


def normalize_number(value: str) -> float | None:
    """Extract a numeric value from an answer string (handles %, commas, units)."""
    if not value:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value).strip().replace(",", ""))
    if not cleaned or cleaned in (".", "-"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _semantic_equivalent(answer_a: str, answer_b: str, api_key: str) -> bool:
    """Ask the text model whether two short answers mean the same thing."""
    prompt = _VERIFY_SEMANTIC_PROMPT.format(a=answer_a, b=answer_b)

    def _parse_match(content: str, raw_text: str = "") -> dict | None:
        if not content:
            return None
        text = content.strip()
        if text.startswith("```"):
            import re as _re

            text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = _re.sub(r"\n?```\s*$", "", text)
        import json as _json

        try:
            data = _json.loads(text)
            if "match" in data:
                return data
        except _json.JSONDecodeError:
            pass
        m = re.search(r'"match"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m:
            return {"match": m.group(1).lower() == "true"}
        return None

    try:
        result = _call_opencode(
            api_key,
            prompt,
            "",
            model=CONFIG.text_model,
            retries=1,
            parser=_parse_match,
        )
        if result and "match" in result:
            return bool(result["match"])
    except Exception as e:
        logger.warning("determinism: semantic equivalence call failed: %s", str(e)[:120])
    return False


def matches_golden(model_answer: str, golden: str, answer_format: str = "number", api_key: str = "") -> bool:
    """Compare a model answer against the golden answer.

    Numeric answers compare by normalized value with a small relative
    tolerance (formatting-insensitive: "20%", "20.0", "20" all match).
    Word/phrase answers compare normalized-exact, falling back to semantic
    equivalence via the text model when the formats differ.
    """
    model_answer = str(model_answer or "").strip()
    golden = str(golden or "").strip()
    if not model_answer or not golden:
        return False
    if answer_format == "number":
        a = normalize_number(model_answer)
        b = normalize_number(golden)
        if a is None or b is None:
            return False
        return abs(a - b) <= max(0.01, abs(b) * 0.005)
    if model_answer.lower() == golden.lower():
        return True
    if api_key:
        return _semantic_equivalent(model_answer, golden, api_key)
    return False


def _parse_answer(content: str | None, raw_text: str = "") -> dict | None:
    """Parse CHECK_ANSWER output: {"reasoning": "...", "answer": "..."}."""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        import re as _re

        text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)
    import json as _json

    try:
        data = _json.loads(text)
        if "answer" in data and data["answer"]:
            return data
    except _json.JSONDecodeError:
        pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = _json.loads(text[start : end + 1])
                        if "answer" in data and data["answer"]:
                            return data
                    except _json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def check_determinism_for_qa(
    question: str,
    golden: str,
    answer_format: str,
    image_path: str | Path,
    api_key: str,
    runs: int = 3,
    difficulty: str = "challenging",
    model: str | None = None,
) -> dict:
    """Run the question against the vision model `runs` times and compare answers.

    For "hardest" difficulty, uses Qwen by default to verify that Qwen specifically
    fails. For other difficulties, uses the default model (minimax-m3).

    Returns:
        {
            "deterministic": bool,          # all sampled runs matched the golden
            "runs": [{"answer": str|None, "reasoning": str|None, "match": bool}],
            "diverging": [str],             # answers that did not match the golden
            "checked": bool,                # False when no run produced an answer (fail-open)
        }
    """
    b64, image_media_type = encode_image_for_llm(image_path, CONFIG.thumbnail_size, CONFIG.jpeg_quality)
    prompt = CHECK_ANSWER_PROMPT.text.format(question=question)

    # For hardest difficulty, use Qwen to verify Qwen specifically fails
    if model is None:
        if difficulty == "hardest":
            model = "openrouter/qwen/qwen3.6-35b-a3b"
        else:
            model = CONFIG.default_model

    runs_out = []
    diverging: list[str] = []
    answered = 0
    for i in range(runs):
        try:
            result = _call_opencode(
                api_key,
                prompt,
                b64,
                model=model,
                retries=1,
                difficulty=difficulty,
                media_type=image_media_type,
                parser=_parse_answer,
            )
        except Exception as e:
            logger.warning("determinism: run %d call failed: %s", i + 1, str(e)[:120])
            result = None
        answer = str(result.get("answer", "")).strip() if result else ""
        reasoning = str(result.get("reasoning", "")).strip() if result else ""
        match = bool(answer) and matches_golden(answer, golden, answer_format, api_key)
        if answer:
            answered += 1
            if not match:
                diverging.append(answer)
        runs_out.append({"answer": answer, "reasoning": reasoning, "match": match})

    checked = answered > 0
    deterministic = checked and not diverging
    if not deterministic and checked:
        logger.warning(
            "determinism: %d/%d runs diverged from golden=%s answers=%s",
            len(diverging),
            runs,
            golden,
            diverging[:3],
        )
    return {
        "deterministic": deterministic,
        "runs": runs_out,
        "diverging": diverging,
        "checked": checked,
    }
