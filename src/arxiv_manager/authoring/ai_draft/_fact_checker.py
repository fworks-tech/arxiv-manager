"""Adversarial premise fact-checking for generated Q&A drafts.

Asks a vision model to enumerate every factual claim a question makes
about the image and verify each one is actually visible. Catches the
class of "internally inconsistent" questions where the premise is
visually false (e.g. "the two 'text overlap' panels" when only one
panel has that label) — something text-only validation cannot see.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .._draft_config import CONFIG
from .._draft_prompts import FACT_CHECK_PROMPT
from ._api_client import _call_opencode
from ._image_utils import encode_image_for_llm

logger = logging.getLogger(__name__)

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
UNVERIFIABLE = "UNVERIFIABLE"


def _parse_fact_check(content: str | None, raw_text: str = "") -> dict | None:
    """Parse the fact-check JSON response.

    Expects {"claims": [{"claim": ..., "verdict": ..., "evidence": ...}], "verdict": "pass|fail"}.
    Falls back to scanning for a balanced JSON object with a "claims" key.
    """
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        import re as _re

        text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)

    import json as _json

    def _try_parse(candidate: str) -> dict | None:
        try:
            data = _json.loads(candidate)
            if "verdict" in data and "claims" in data:
                return data
        except _json.JSONDecodeError:
            return None
        return None

    data = _try_parse(text)
    if data:
        return data

    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    data = _try_parse(text[start : end + 1])
                    if data:
                        return data
                    break
        start = text.find("{", start + 1)
    logger.warning("_parse_fact_check: could not parse (len=%d, preview=%.200s)", len(text), text[:200])
    return None


def _normalize_verdict(verdict) -> str:
    """Normalize a model verdict string into one of the three constants."""
    v = str(verdict or "").strip().upper().replace(" ", "_")
    if v in (SUPPORTED, NOT_SUPPORTED, UNVERIFIABLE):
        return v
    if v.startswith("NOT_"):
        return NOT_SUPPORTED
    if v.startswith("UNVERIFI") or v.startswith("CANNOT") or v.startswith("CAN'T"):
        return UNVERIFIABLE
    return SUPPORTED


def fact_check_draft(
    question: str,
    image_path: str | Path,
    api_key: str,
    model: str | None = None,
    difficulty: str = "",
) -> dict:
    """Verify every factual claim in a question against the image.

    Returns:
        {
            "claims": [{"claim": str, "verdict": str, "evidence": str}],
            "unsupported": [str],          # NOT_SUPPORTED or UNVERIFIABLE claims
            "verdict": "pass" | "fail",
            "checked": bool,               # False when the check itself failed (fail-open)
        }

    Fail-open: if the checker call itself fails (network/parse), verdict is
    "pass" with checked=False so regeneration is not blocked by tooling
    errors — the checker is conservative only when it actually runs.
    """
    if not question or not image_path:
        return {"claims": [], "unsupported": [], "verdict": "pass", "checked": False}

    b64, image_media_type = encode_image_for_llm(image_path, CONFIG.thumbnail_size, CONFIG.jpeg_quality)
    prompt = FACT_CHECK_PROMPT.text.format(question=question)

    try:
        result = _call_opencode(
            api_key,
            prompt,
            b64,
            model=model or CONFIG.default_model,
            retries=1,
            difficulty=difficulty,
            media_type=image_media_type,
            parser=_parse_fact_check,
        )
    except Exception as e:  # fail-open on tooling errors
        logger.warning("fact_check_draft: check call failed, failing open: %s", str(e)[:150])
        return {"claims": [], "unsupported": [], "verdict": "pass", "checked": False}

    if not result or "claims" not in result:
        logger.warning("fact_check_draft: no usable check result, failing open")
        return {"claims": [], "unsupported": [], "verdict": "pass", "checked": False}

    claims = []
    unsupported = []
    for c in result.get("claims", []) or []:
        claim = str(c.get("claim", "")).strip()
        if not claim:
            continue
        verdict = _normalize_verdict(c.get("verdict"))
        evidence = str(c.get("evidence", "")).strip()
        claims.append({"claim": claim, "verdict": verdict, "evidence": evidence})
        if verdict != SUPPORTED:
            unsupported.append(claim)

    verdict = "fail" if unsupported else "pass"
    if verdict == "fail":
        logger.warning(
            "fact_check_draft: %d unsupported claim(s): %s",
            len(unsupported),
            "; ".join(unsupported[:3]),
        )
    return {"claims": claims, "unsupported": unsupported, "verdict": verdict, "checked": True}
