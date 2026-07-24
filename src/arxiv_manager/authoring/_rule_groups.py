"""Validation rule groups — each _run_* method is a group of related rules."""

from __future__ import annotations

import logging
import re

from ._validation_helpers import (
    TRICK_ANSWERS,
    MATH_HEAVY_PATTERNS,
    TEXT_HEAVY_PATTERNS,
    LONG_WINDED_INDICATORS,
    NOISE_CONDITION_PATTERNS,
    WATERMARK_HINTS,
    _answer_is_extreme,
    _answer_is_list_of_three_plus,
    _answer_seems_derivable,
    _check_grammar,
    _check_mcq_options,
    _count_sentences,
    _has_answer_in_question,
    _has_domain_jargon,
    _has_extreme_seeking,
    _has_reasoning_depth,
    _has_threshold_filter,
    _is_binary_question,
    _is_caption_solvable,
    _is_chart_math_only,
    _is_explanation_question,
    _is_generic_count_question,
    _is_number,
    _matches_chart_anti_pattern,
    _passes_one_answer_test,
    _passes_visual_dependence_test,
    _references_chart_data,
    _references_multi_panel,
    _references_visual_content,
    _restricts_options,
)

logger = logging.getLogger(__name__)


def _run_format_checks(result, q: str, a: str, answer_format: str) -> None:
    """Rules 1-5: Binary/T-F, answer format, length, trick answers, single question."""
    if _is_binary_question(q):
        result.errors.append("Binary/T-F question is not allowed")
    else:
        result.passed_checks.append("Question is not binary/T-F")

    if not answer_format:
        result.errors.append("Answer format not specified (e.g. number, word, phrase)")
    else:
        result.passed_checks.append("Answer format is specified")

    word_count = len(a.split())
    if word_count > 4:
        result.warnings.append(f"Answer has {word_count} words — prefer 1-2 words (max 4)")
    else:
        result.passed_checks.append(f"Answer is concise ({word_count} word{'s' if word_count != 1 else ''})")
    if len(a) > 50:
        result.errors.append(f"Answer too long ({len(a)} chars)")

    if a in TRICK_ANSWERS:
        result.errors.append(f"Trick answer '{a}' is not allowed")
    else:
        result.passed_checks.append("Answer is not a trick answer")

    stripped = q.rstrip(".!?")
    ending = q[len(stripped):] if len(stripped) < len(q) else ""
    if q.count("?") > 1:
        result.errors.append(f"Multiple questions detected ({q.count('?')} question marks)")
    elif not ending:
        result.errors.append("Question must end with punctuation ('?' or '.')")
    else:
        result.passed_checks.append(f"Ends with '{ending}'")


def _run_content_checks(result, q: str, a: str, answer_format: str) -> None:
    """Rules 6-11: Sentence count, option restriction, jargon, visual reference, format consistency, explanation."""
    sentences = _count_sentences(q)
    if sentences > 2:
        result.warnings.append(f"Question has {sentences} sentences — prefer 1-2")
    else:
        result.passed_checks.append("Question is concise")

    if _restricts_options(q):
        result.errors.append("Don't restrict options in question (e.g. 'Out of the 3...')")
    else:
        result.passed_checks.append("No option restriction in question")

    if _has_domain_jargon(q):
        result.warnings.append("Contains domain-specific terminology — rewrite for general audience")
    else:
        result.passed_checks.append("No domain-specific jargon")

    if not _references_visual_content(q):
        result.warnings.append("Question may not require the image to answer")
    else:
        result.passed_checks.append("Question references visual content")

    if answer_format == "number" and not _is_number(a):
        result.warnings.append(f"Answer format is 'number' but answer '{a}' doesn't look numeric")
    elif answer_format == "number":
        result.passed_checks.append("Answer matches declared format")
    if answer_format == "percent" and "%" not in a:
        result.warnings.append("Answer format is 'percent' but answer missing '%'")

    if _is_explanation_question(q):
        result.errors.append("Explanation questions ('Explain how...' / 'What trend...') are not allowed")
    else:
        result.passed_checks.append("Not an explanation question")


def _run_complexity_checks(result, q: str, figure_type: str, task_type: str) -> None:
    """Rules 12-12d: Reasoning depth, chart anti-patterns, generic count, chart math-only."""
    if _has_reasoning_depth(q):
        result.passed_checks.append("Question requires multi-step reasoning")
    else:
        result.warnings.append("Question may be too simple — consider adding comparison or ranking")

    is_chart = figure_type in ("chart_graph_text", "chart") or task_type == "chart"
    if is_chart:
        anti_pattern_hits = _matches_chart_anti_pattern(q)
        if anti_pattern_hits:
            for hit in anti_pattern_hits:
                result.errors.append(
                    f"Chart anti-pattern: '{hit}' — chart questions must reference data values, not just chart furniture (labels/ticks/colorbars)"
                )
        else:
            if _references_chart_data(q):
                result.passed_checks.append("References chart data (axis values, peaks, regions) — not just furniture")
            else:
                result.warnings.append(
                    "Chart question may not reference actual data — consider referencing axis values, peaks, regions, or cross-panel comparisons"
                )

    if _is_generic_count_question(q):
        result.errors.append(
            "Question is a generic count ('How many X in the image?') without a filter, comparison, or arithmetic — too easy for Qwen"
        )

    if is_chart and _is_chart_math_only(q):
        result.errors.append(
            "Chart question is pure math (ratio/difference of values stated in text) — the image is not required. "
            "Rewrite to ask about a SPECIFIC visual element (peak, trough, color region, data point) that requires reading the chart"
        )


def _run_handbook_basics(result, q: str, a: str, caption: str) -> None:
    """Rules 13-19: Derivability, grammar, extreme-seeking, threshold, multi-panel, caption, extreme answer."""
    if _answer_seems_derivable(q, a):
        result.passed_checks.append("Answer appears derivable from question")
    else:
        result.warnings.append("Answer may not be clearly derivable from the question")

    grammar_issues = _check_grammar(q)
    if grammar_issues:
        for issue in grammar_issues:
            result.warnings.append(issue)
    else:
        result.passed_checks.append("Basic grammar checks passed")

    if _has_extreme_seeking(q):
        result.warnings.append("Uses extreme-seeking words (highest/lowest/most) — Qwen checks these first; consider threshold filters instead")
    else:
        result.passed_checks.append("No extreme-seeking bias detected")

    if _has_threshold_filter(q):
        result.passed_checks.append("Uses threshold filters (creates genuine visual complexity)")

    if _references_multi_panel(q):
        result.passed_checks.append("References multiple panels (cross-panel reasoning)")

    if caption and _is_caption_solvable(caption, q):
        result.warnings.append("Caption is very descriptive / question asks about caption — image may not be required")
    elif caption:
        result.passed_checks.append("Caption is not overly descriptive")

    if _answer_is_extreme(a):
        result.warnings.append("Answer is an extreme value (highest/lowest) — intermediate values are harder for models")


def _run_visual_tests(result, q: str, a: str) -> None:
    """Rules 20-20b: Visual-dependence test, one-answer test, answer-in-question."""
    if not _passes_visual_dependence_test(q):
        result.errors.append("Test 1 FAILED: A smart person could answer this without the image")
    else:
        result.passed_checks.append("Passes visual-dependence test (handbook §3)")
    if not _passes_one_answer_test(q, a):
        result.warnings.append("Test 2 WARNING: Answer may be subjective / two reasonable people could give different answers")
    else:
        result.passed_checks.append("Passes one-answer test (handbook §3)")

    if _has_answer_in_question(q, a):
        result.errors.append(
            "Question provides the data needed to compute the answer in the text (visual-dependence failure). "
            "Rewrite so the image is REQUIRED — ask about a SPECIFIC visual element (peak, region, color), not a math operation on values stated in the question"
        )


def _run_handbook_errors(result, q: str, a: str, options: list[str] | None,
                          figure_type: str, task_type: str, image_path: str) -> None:
    """Rules 21-28: Math-heavy, text-only, long-winded, noise, list answer, MCQ, watermark, type mismatch."""
    if any(re.search(p, q, re.IGNORECASE) for p in MATH_HEAVY_PATTERNS):
        result.warnings.append("Question focuses on calculation rather than visuo-spatial reasoning (handbook common error)")
    else:
        result.passed_checks.append("Question is visuo-spatial, not pure calculation")

    if any(re.search(p, q, re.IGNORECASE) for p in TEXT_HEAVY_PATTERNS):
        result.errors.append("Question is text-only — does not require visual reasoning (handbook common error)")
    else:
        result.passed_checks.append("Question is not text-only")

    if any(re.search(p, q, re.IGNORECASE) for p in LONG_WINDED_INDICATORS):
        result.warnings.append("Question is long-winded/awkward — rewrite for clarity (handbook common error)")

    if any(re.search(p, q, re.IGNORECASE) for p in NOISE_CONDITION_PATTERNS):
        result.warnings.append("Question has a condition that may not materially change the answer (handbook error #7)")

    if _answer_is_list_of_three_plus(a):
        result.warnings.append("Answer is a list with more than 3 short elements — handbook §5 bans this")

    if options:
        mcq_issues = _check_mcq_options(options)
        for issue in mcq_issues:
            result.warnings.append(issue)
        if not any("MCQ" in i for i in mcq_issues):
            result.passed_checks.append(f"MCQ has {len(options)} options meeting handbook §5")

    if image_path:
        lower_path = image_path.lower()
        for hint in WATERMARK_HINTS:
            if re.search(hint, lower_path):
                result.warnings.append("Filename suggests potential watermark/copyright — verify CC0 license")

    if figure_type and task_type:
        mismatch = (
            (figure_type == "general_image" and task_type in ("chart",))
            or (figure_type == "chart_graph_text" and task_type == "spatial")
        )
        if mismatch:
            result.warnings.append(
                f"figure_type='{figure_type}' may not match task_type='{task_type}'"
            )
        else:
            result.passed_checks.append("figure_type matches task_type")
