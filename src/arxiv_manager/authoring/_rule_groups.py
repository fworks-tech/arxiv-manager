"""Validation rule groups — each _run_* method is a group of related rules."""

from __future__ import annotations

import logging
import re

from ._validation_helpers import (
    LONG_WINDED_INDICATORS,
    MATH_HEAVY_PATTERNS,
    NOISE_CONDITION_PATTERNS,
    TEXT_HEAVY_PATTERNS,
    TRICK_ANSWERS,
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
    _has_inline_choices,
    _has_reasoning_depth,
    _has_threshold_filter,
    _is_binary_answer,
    _is_binary_question,
    _is_caption_solvable,
    _is_chart_math_only,
    _is_explanation_question,
    _is_generic_count_question,
    _is_int,
    _is_number,
    _is_single_matchmaking,
    _is_two_answer_question,
    _matches_chart_anti_pattern,
    _passes_one_answer_test,
    _passes_visual_dependence_test,
    _references_chart_data,
    _references_multi_panel,
    _references_visual_content,
    _requires_arithmetic,
    _requires_calculation,
    _restricts_options,
)
from ._validation_helpers import (
    _is_manufactured_difficulty as _check_manufactured,
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
    ending = q[len(stripped) :] if len(stripped) < len(q) else ""
    if q.count("?") > 1:
        result.errors.append(f"Multiple questions detected ({q.count('?')} question marks)")
    elif not ending:
        result.errors.append("Question must end with punctuation ('?' or '.')")
    else:
        result.passed_checks.append(f"Ends with '{ending}'")


def _run_content_checks(result, q: str, a: str, answer_format: str, difficulty: str = "") -> None:
    """Rules 6-11: Sentence count, option restriction, jargon, visual reference, format consistency, explanation."""
    sentences = _count_sentences(q)
    is_challenging = difficulty in ("challenging", "hardest")
    if sentences >= 3:
        if is_challenging:
            result.errors.append(f"Question has {sentences} sentences — max 2 for Challenging difficulty. Multi-sentence questions create ambiguity and often leak hints.")
        else:
            result.warnings.append(f"Question has {sentences} sentences — prefer 1-2")
    elif sentences == 2 and is_challenging:
        result.warnings.append("Question has 2 sentences — 1 sentence preferred for clarity. Two-sentence questions often leak hints in the first sentence that trivialize the comparison.")
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
        result.errors.append(f"Answer format is 'number' but answer '{a}' doesn't look numeric")
    elif answer_format == "number":
        result.passed_checks.append("Answer matches declared format")
    if (answer_format == "integer" or answer_format == "int") and not _is_int(a):
        result.errors.append(f"Answer format is 'integer' but answer '{a}' doesn't look like an integer")
    elif answer_format in ("integer", "int"):
        result.passed_checks.append("Answer matches declared format")
    if answer_format == "percent" and "%" not in a:
        result.errors.append("Answer format is 'percent' but answer missing '%'")

    if _is_explanation_question(q):
        result.errors.append("Explanation questions ('Explain how...' / 'What trend...') are not allowed")
    else:
        result.passed_checks.append("Not an explanation question")

    if _has_inline_choices(q):
        result.errors.append(
            "Question provides inline choices ('Is it X or Y?', 'Choose between X and Y') — "
            "this creates a 50/50 binary guess. Short-answer questions must NOT provide predefined options."
        )
    else:
        result.passed_checks.append("No inline choices in question")

    if difficulty == "hardest" and _is_two_answer_question(q):
        result.errors.append(
            "Hardest questions must have a single answer. Split into two questions or combine into one operation."
        )

    if difficulty == "hardest" and not _requires_calculation(q):
        result.warnings.append(
            "Hardest questions should require calculation or multi-step reasoning. "
            "Consider adding arithmetic (percentage change, difference, rank delta)."
        )


def _run_complexity_checks(result, q: str, a: str, figure_type: str, task_type: str, difficulty: str = "") -> None:
    """Rules 12-12d: Reasoning depth, chart anti-patterns, generic count, chart math-only."""
    if _has_reasoning_depth(q):
        result.passed_checks.append("Question requires multi-step reasoning")
    else:
        if difficulty in ("challenging", "hardest"):
            result.errors.append("Question too simple for Challenging difficulty — must require multi-step reasoning, comparison, or ranking")
        else:
            result.warnings.append("Question may be too simple — consider adding comparison or ranking")

    is_chart = figure_type in ("chart_graph_text", "chart") or task_type == "chart"
    if is_chart:
        anti_pattern_hits = _matches_chart_anti_pattern(q)
        has_data_refs = _references_chart_data(q)
        if anti_pattern_hits:
            if has_data_refs:
                for hit in anti_pattern_hits:
                    result.warnings.append(f"Chart caution: '{hit}' — but question references data values (acceptable)")
            else:
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

    if _check_manufactured(q, difficulty):
        result.errors.append(
            "Question uses counting as the main difficulty source ('manufactured difficulty'). "
            "Visual reasoning (comparison, spatial relationships, classification across panels) "
            "should be the primary challenge. Counting can be the final step after meaningful visual analysis."
        )

    if difficulty in ("challenging", "hardest") and _is_binary_answer(q, a):
        result.errors.append(
            "Binary/alternate-choice answer (higher/lower, increase/decrease, up/down) — "
            "Qwen can guess correctly 50% of the time. Use ranking, ordinal delta, or a specific value instead."
        )

    if difficulty in ("challenging", "hardest") and _is_single_matchmaking(q, a):
        result.errors.append(
            "Single-criterion matchmaking ('which panel meets X') — scanning for one label is too easy. "
            "Require comparison across panels, ranking, or ordinal reasoning instead."
        )

    if is_chart and _is_chart_math_only(q):
        result.errors.append(
            "Chart question is pure math (ratio/difference of values stated in text) — the image is not required. "
            "Rewrite to ask about a SPECIFIC visual element (peak, trough, color region, data point) that requires reading the chart"
        )

    if _requires_arithmetic(q):
        result.warnings.append(
            "Question asks for multiplication/addition of values — ensure the answer correctly reflects the arithmetic, "
            "not just a simple count. A common error is asking for 'product of X and Y' but answering with X alone."
        )


def _run_handbook_basics(result, q: str, a: str, caption: str, difficulty: str = "") -> None:
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
        if difficulty in ("challenging", "hardest"):
            result.errors.append(
                "Uses extreme-seeking words (highest/lowest/most) for Challenging difficulty — "
                "Qwen checks these first; use thresholds or ordinal ranking instead"
            )
        else:
            result.warnings.append(
                "Uses extreme-seeking words (highest/lowest/most) — Qwen checks these first; consider threshold filters instead"
            )
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
        result.warnings.append(
            "Answer is an extreme value (highest/lowest) — intermediate values are harder for models"
        )


def _run_visual_tests(result, q: str, a: str) -> None:
    """Rules 20-20b: Visual-dependence test, one-answer test, answer-in-question."""
    if not _passes_visual_dependence_test(q):
        result.errors.append("Test 1 FAILED: A smart person could answer this without the image")
    else:
        result.passed_checks.append("Passes visual-dependence test (handbook §3)")
    if not _passes_one_answer_test(q, a):
        result.warnings.append(
            "Test 2 WARNING: Answer may be subjective / two reasonable people could give different answers"
        )
    else:
        result.passed_checks.append("Passes one-answer test (handbook §3)")

    if _has_answer_in_question(q, a):
        result.errors.append(
            "Question provides the data needed to compute the answer in the text (visual-dependence failure). "
            "Rewrite so the image is REQUIRED — ask about a SPECIFIC visual element (peak, region, color), not a math operation on values stated in the question"
        )


def _run_final_checks(
    result, q: str, a: str, options: list[str] | None, figure_type: str, task_type: str, image_path: str
) -> None:
    """Rules 21-28: Math-heavy, text-only, long-winded, noise, list answer, MCQ, watermark, type mismatch."""
    if any(re.search(p, q, re.IGNORECASE) for p in MATH_HEAVY_PATTERNS):
        result.warnings.append(
            "Question focuses on calculation rather than visuo-spatial reasoning (handbook common error)"
        )
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
        mismatch = (figure_type == "general_image" and task_type in ("chart",)) or (
            figure_type == "chart_graph_text" and task_type == "spatial"
        )
        if mismatch:
            result.warnings.append(f"figure_type='{figure_type}' may not match task_type='{task_type}'")
        else:
            result.passed_checks.append("figure_type matches task_type")
