"""Composable guardrails for Q&A generation quality control.

Each guardrail is a function: (draft, context) -> (passed: bool, reason: str)
The run_guardrails orchestrator runs all checks and triggers auto-retry with feedback.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def check_answer_plausible(draft: dict, context: dict) -> tuple[bool, str]:
    """Reject empty, trivial, or impossible answers."""
    answer = (draft.get("answer") or "").strip().lower()
    if not answer:
        return False, "Answer is empty"
    if answer in ("none", "n/a", "na", "cannot be determined", "unknown", "null", "undefined"):
        return False, f"Answer is trivial ('{answer}')"
    return True, ""


def check_extreme_answer(draft: dict, context: dict) -> tuple[bool, str]:
    """Reject answers that are statistically improbable (LLM hallucination markers)."""
    answer = (draft.get("answer") or "").strip()
    if not answer:
        return True, ""
    answer_format = draft.get("answer_format", "")
    if answer_format in ("number", "integer", "percent"):
        try:
            val = float(answer.replace(",", "").replace("%", ""))
            if val == 0 and answer_format != "percent":
                return False, "Answer is zero (likely hallucination)"
            if val > 1_000_000:
                return False, f"Answer is extremely large ({val:,.0f})"
            if val < -1_000_000:
                return False, f"Answer is extremely negative ({val:,.0f})"
        except ValueError:
            pass
    return True, ""


def check_answer_format_match(draft: dict, context: dict) -> tuple[bool, str]:
    """Verify answer matches the declared format."""
    answer = (draft.get("answer") or "").strip()
    fmt = draft.get("answer_format", "")
    if not answer or not fmt:
        return True, ""

    if fmt == "number":
        if not re.match(r"^-?\d+(\.\d+)?$", answer.replace(",", "")):
            return False, f"Answer '{answer}' does not match format '{fmt}'"
    elif fmt == "integer":
        if not re.match(r"^-?\d+$", answer.replace(",", "")):
            return False, f"Answer '{answer}' does not match format '{fmt}'"
    elif fmt == "percent":
        if not re.match(r"^-?\d+(\.\d+)?%?$", answer.replace(",", "")):
            return False, f"Answer '{answer}' does not match format '{fmt}'"
    elif fmt == "word":
        if len(answer.split()) > 4:
            return False, f"Answer '{answer}' is too long for format 'word' ({len(answer.split())} words)"
    elif fmt == "phrase":
        if len(answer.split()) > 6:
            return False, f"Answer '{answer}' is too long for format 'phrase' ({len(answer.split())} words)"
    return True, ""


def check_diversity(draft: dict, context: dict) -> tuple[bool, str]:
    """Check if question is too similar to previous (when previous_question is provided)."""
    prev = context.get("previous_question", "")
    if not prev:
        return True, ""
    q = (draft.get("question") or "").strip().lower()
    p = prev.strip().lower()
    if q == p:
        return False, "Question is identical to previous question"
    # Simple word-overlap check for near-duplicates
    q_words = set(q.split())
    p_words = set(p.split())
    if len(q_words) > 0 and len(p_words) > 0:
        overlap = len(q_words & p_words) / max(len(q_words), len(p_words))
        if overlap > 0.85:
            return False, f"Question has {overlap:.0%} word overlap with previous question"
    return True, ""


def quality_threshold(draft: dict, context: dict) -> tuple[bool, str]:
    """Flag drafts below quality threshold for auto-retry."""
    min_quality = context.get("min_quality", 40)
    validation = context.get("validation_result", {})
    quality = validation.get("quality_score", 0) if isinstance(validation, dict) else 0
    if quality > 0 and quality < min_quality:
        return False, f"Quality score {quality:.0f} is below threshold {min_quality}"
    return True, ""


GUARDRAILS = [
    check_answer_plausible,
    check_extreme_answer,
    check_answer_format_match,
    check_diversity,
]


def _run_guardrail_checks(draft: dict, context: dict) -> list[str]:
    """Run all guardrail checks and return list of failure reasons."""
    failed: list[str] = []
    for check_fn in GUARDRAILS:
        passed, reason = check_fn(draft, context)
        if not passed:
            failed.append(reason)
    return failed


def _auto_retry(draft, context, api_key, image_path, feedback, draft_qa_callback):
    """Retry generation with guardrail feedback as prompt context."""
    if not api_key or not image_path:
        return None
    if draft_qa_callback is not None:
        return draft_qa_callback(
            image_path=image_path,
            api_key=api_key,
            feedback=feedback,
            difficulty=context.get("difficulty", ""),
            figure_type=context.get("figure_type", ""),
            complexity_score=context.get("complexity_score", 0.0),
            previous_question=context.get("previous_question", ""),
            figure_id=context.get("figure_id"),
            model=context.get("model"),
        )
    from .ai_draft import draft_qa

    return draft_qa(
        image_path=image_path,
        api_key=api_key,
        feedback=feedback,
        difficulty=context.get("difficulty", ""),
        figure_type=context.get("figure_type", ""),
        complexity_score=context.get("complexity_score", 0.0),
        previous_question=context.get("previous_question", ""),
        figure_id=context.get("figure_id"),
        model=context.get("model"),
    )


def run_guardrails(
    draft: dict,
    context: dict,
    api_key: str | None = None,
    image_path: str = "",
    max_retries: int = 2,
    draft_qa_callback=None,
) -> dict | None:
    """Run all guardrails against a generated draft.

    If any guardrail fails, attempts to regenerate with feedback
    up to max_retries times using the REGEN_PROMPT feedback mechanism.
    Returns the verified draft or None if all retries fail.
    """
    for attempt in range(max_retries + 1):
        failed_checks = _run_guardrail_checks(draft, context)

        if not failed_checks:
            return draft

        logger.warning(
            "guardrail attempt %d/%d: %d failures: %s",
            attempt + 1,
            max_retries + 1,
            len(failed_checks),
            "; ".join(failed_checks),
        )

        if attempt >= max_retries:
            logger.warning("guardrail: max retries exhausted, returning None")
            return None

        feedback = "; ".join(failed_checks)

        retry = _auto_retry(draft, context, api_key, image_path, feedback, draft_qa_callback)
        if retry:
            draft = retry
            continue

        return None

    return None
