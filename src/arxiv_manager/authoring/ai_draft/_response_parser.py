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

    # Handle incomplete <think> blocks (missing closing tag or truncated response)
    if not reasoning and cleaned.startswith("<think>"):
        # No complete think block found, but text starts with <think>
        # Try to extract everything up to the end as reasoning
        think_end = cleaned.find("</think>")
        if think_end == -1:
            # No closing tag - entire text is reasoning, no JSON
            reasoning = cleaned[7:].strip()  # Remove "<think>" prefix
            cleaned = ""
        else:
            # Found closing tag
            reasoning = cleaned[7:think_end].strip()
            cleaned = cleaned[think_end + 8:].strip()

    return cleaned, reasoning


def _extract_json_from_text(source: str, key_hint: str, reasoning: str = "") -> dict | None:
    """Search a text string for a balanced JSON object containing key_hint.

    Returns parsed dict or None.
    """
    for txt in (source, reasoning):
        if not txt:
            continue
        start = 0
        while True:
            start = txt.find("{", start)
            if start < 0:
                break
            depth = 0
            for end in range(start, len(txt)):
                if txt[end] == "{":
                    depth += 1
                elif txt[end] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = txt[start: end + 1]
                        if key_hint in candidate:
                            try:
                                data = json.loads(candidate)
                                key_name = key_hint.strip('"\'')
                                if key_name in data:
                                    return data
                            except json.JSONDecodeError:
                                pass
                        break
            start = end + 1
    return None


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

    # Search cleaned text for JSON with "question" key
    found = _extract_json_from_text(text, '"question"', reasoning)
    if found and "question" in found and "answer" in found:
        found["_reasoning_trace"] = reasoning
        found["_raw_response"] = raw_text or text
        found.setdefault("answer_format", "number")
        found.setdefault("task_type", "chart")
        return found

    # For very large responses (>10KB), the JSON answer is typically at the end
    # after a long reasoning block. Try extracting from the last 16KB.
    if len(text) > 10240:
        tail = text[-16384:]
        found_tail = _extract_json_from_text(tail, '"question"', "")
        if found_tail and "question" in found_tail and "answer" in found_tail:
            found_tail["_reasoning_trace"] = reasoning
            found_tail["_raw_response"] = raw_text or text
            found_tail.setdefault("answer_format", "number")
            found_tail.setdefault("task_type", "chart")
            logger.info("_parse_llm_response: recovered from tail (total=%d tail=%d)", len(text), len(tail))
            return found_tail

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

    # Search cleaned text and reasoning for JSON with "score" key
    found = _extract_json_from_text(cleaned, '"score"', reasoning)
    if found and "score" in found:
        found["_reasoning_trace"] = reasoning
        found["_raw_response"] = raw_text or cleaned
        return found

    logger.warning("_parse_critique_response: could not parse (len=%d, preview=%.200s)", len(text), text[:200])
    return None
