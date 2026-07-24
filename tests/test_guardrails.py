"""Tests for the guardrails system — composable quality checks."""
import pytest
from arxiv_manager.authoring._guardrails import (
    check_answer_plausible,
    check_extreme_answer,
    check_answer_format_match,
    check_diversity,
    quality_threshold,
    GUARDRAILS,
)


class TestCheckAnswerPlausible:
    def test_empty_answer(self):
        passed, reason = check_answer_plausible({"answer": ""}, {})
        assert not passed
        assert "empty" in reason.lower()

    def test_trivial_answers(self):
        for trivial in ("none", "N/A", "cannot be determined", "unknown", "null"):
            passed, reason = check_answer_plausible({"answer": trivial}, {})
            assert not passed, f"{trivial} should be rejected"
            assert "trivial" in reason.lower()

    def test_valid_answer(self):
        passed, reason = check_answer_plausible({"answer": "42"}, {})
        assert passed
        assert reason == ""


class TestCheckExtremeAnswer:
    def test_zero_number(self):
        passed, reason = check_extreme_answer({"answer": "0", "answer_format": "number"}, {})
        assert not passed
        assert "zero" in reason.lower()

    def test_zero_percent_allowed(self):
        passed, reason = check_extreme_answer({"answer": "0%", "answer_format": "percent"}, {})
        assert passed

    def test_extremely_large(self):
        passed, reason = check_extreme_answer({"answer": "5000000", "answer_format": "number"}, {})
        assert not passed
        assert "large" in reason.lower()

    def test_normal_number(self):
        passed, reason = check_extreme_answer({"answer": "42", "answer_format": "number"}, {})
        assert passed

    def test_negative_extreme(self):
        passed, reason = check_extreme_answer({"answer": "-9999999", "answer_format": "number"}, {})
        assert not passed

    def test_non_numeric_format_skips(self):
        passed, reason = check_extreme_answer({"answer": "blue", "answer_format": "word"}, {})
        assert passed


class TestCheckAnswerFormatMatch:
    def test_number_format_valid(self):
        passed, reason = check_answer_format_match({"answer": "42", "answer_format": "number"}, {})
        assert passed

    def test_number_format_invalid(self):
        passed, reason = check_answer_format_match({"answer": "forty-two", "answer_format": "number"}, {})
        assert not passed

    def test_integer_format(self):
        passed, reason = check_answer_format_match({"answer": "7", "answer_format": "integer"}, {})
        assert passed

    def test_integer_format_decimal_rejected(self):
        passed, reason = check_answer_format_match({"answer": "7.5", "answer_format": "integer"}, {})
        assert not passed

    def test_percent_format(self):
        passed, reason = check_answer_format_match({"answer": "85%", "answer_format": "percent"}, {})
        assert passed

    def test_percent_format_no_symbol(self):
        passed, reason = check_answer_format_match({"answer": "85", "answer_format": "percent"}, {})
        assert passed

    def test_word_format_too_long(self):
        passed, reason = check_answer_format_match({"answer": "the quick brown fox jumps", "answer_format": "word"}, {})
        assert not passed

    def test_empty_answer_skips(self):
        passed, reason = check_answer_format_match({"answer": "", "answer_format": "number"}, {})
        assert passed


class TestCheckDiversity:
    def test_identical_questions(self):
        passed, reason = check_diversity(
            {"question": "How many bars are in the chart?"},
            {"previous_question": "How many bars are in the chart?"},
        )
        assert not passed
        assert "identical" in reason.lower()

    def test_high_word_overlap(self):
        passed, reason = check_diversity(
            {"question": "How many bars are in the chart above?"},
            {"previous_question": "How many bars are in the chart below?"},
        )
        assert not passed
        assert "overlap" in reason.lower()

    def test_fresh_question(self):
        passed, reason = check_diversity(
            {"question": "What is the peak value in panel A?"},
            {"previous_question": "How many bars are in the chart?"},
        )
        assert passed

    def test_no_previous(self):
        passed, reason = check_diversity({"question": "Q?"}, {})
        assert passed


class TestQualityThreshold:
    def test_below_threshold(self):
        passed, reason = quality_threshold(
            {},
            {"validation_result": {"quality_score": 25}, "min_quality": 40},
        )
        assert not passed
        assert "below" in reason.lower()

    def test_above_threshold(self):
        passed, reason = quality_threshold(
            {},
            {"validation_result": {"quality_score": 85}, "min_quality": 40},
        )
        assert passed

    def test_no_validation(self):
        passed, reason = quality_threshold({}, {"min_quality": 40})
        assert passed


class TestGuardrailsRegistry:
    def test_all_guardrails_registered(self):
        assert len(GUARDRAILS) == 4
        names = [fn.__name__ for fn in GUARDRAILS]
        assert "check_answer_plausible" in names
        assert "check_extreme_answer" in names
        assert "check_answer_format_match" in names
        assert "check_diversity" in names
