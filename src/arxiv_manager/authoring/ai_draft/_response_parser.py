"""Response parsing for LLM outputs — think-block extraction and JSON parsing."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def _extract_reasoning(text: str) -> tuple[str, str]:
    """Extract <think> reasoning blocks from model output.

    Returns (cleaned_text, reasoning_trace).
    """
    reasoning_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    reasoning = "\n".join(part.strip() for part in reasoning_parts).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned, reasoning


def _parse_llm_response(text: str | None, raw_text: str = "") -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks and <think> tags.

    Returns _reasoning_trace and _raw_response in the result dict.
    """
    if not text:
        return None

    text = text.strip()
    err_lower = text.lower()
    if "does not support image" in err_lower or "cannot read" in err_lower:
        logger.warning("_parse_llm_response: model does not support image input: %.200s", text[:200])
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    cleaned, reasoning = _extract_reasoning(text)
    if not cleaned.strip():
        cleaned = text
    text = cleaned

    def _parse_candidate(candidate: str) -> dict | None:
        try:
            data = json.loads(candidate)
            if "question" not in data or "answer" not in data:
                return None
            q_val = data.get("question")
            a_val = data.get("answer")
            if q_val is None or (isinstance(q_val, str) and not q_val.strip()):
                logger.warning("_parse_llm_response: empty question (raw=%.300s)", (raw_text or text)[:300])
                return None
            if a_val is None or (isinstance(a_val, str) and not a_val.strip()):
                logger.warning("_parse_llm_response: empty answer (raw=%.300s)", (raw_text or text)[:300])
                return None
            data["question"] = str(q_val)
            data["answer"] = str(a_val)
            data.setdefault("answer_format", "number")
            data.setdefault("task_type", "chart")
            data["_reasoning_trace"] = reasoning
            data["_raw_response"] = raw_text or text
            return data
        except json.JSONDecodeError:
            return None

    data = _parse_candidate(text)
    if data:
        return data

    start = text.find("{")
    if start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    if '"question"' in candidate:
                        data = _parse_candidate(candidate)
                        if data:
                            return data
                    break

    logger.warning("_parse_llm_response: could not parse (len=%d, preview=%.200s)", len(text), text[:200])
    return None


def _parse_critique_response(text: str | None, raw_text: str = "") -> dict | None:
    """Parse a self-critique response expecting score + rewrite.

    Expected format: {"score": 1-5, "rewrite_question": "...", "rewrite_answer": "..."}
    """
    if not text:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    cleaned, reasoning = _extract_reasoning(text)
    if not cleaned.strip():
        cleaned = text

    def _parse(candidate: str) -> dict | None:
        try:
            data = json.loads(candidate)
            if "score" not in data:
                return None
            data["_reasoning_trace"] = reasoning
            data["_raw_response"] = raw_text or cleaned
            return data
        except json.JSONDecodeError:
            return None

    data = _parse(cleaned)
    if data:
        return data

    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for end in range(start, len(cleaned)):
            if cleaned[end] == "{":
                depth += 1
            elif cleaned[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : end + 1]
                    if '"score"' in candidate:
                        data = _parse(candidate)
                        if data:
                            return data
                    break

    logger.warning("_parse_critique_response: could not parse (len=%d, preview=%.200s)", len(text), text[:200])
    return None
