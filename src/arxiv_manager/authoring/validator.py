"""Handbook rule validation engine — aligned with Rhea review criteria.

Sources:
  - QA Expert Handbook
  - arxiv-manager SKILL.md (challenging-question-generator)
  - docs/qwen_weaknesses.md (Qwen 3.6-35B-A3B exploitation strategies)
  - docs/figure_suggestions.md (figure-type patterns)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ._rule_groups import (
    _run_complexity_checks,
    _run_content_checks,
    _run_format_checks,
    _run_handbook_basics,
    _run_handbook_errors,
    _run_visual_tests,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a task against handbook rules."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100, higher = better Rhea pass chance

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  ❌ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        if not lines:
            lines.append("  ✅ All checks passed")
        lines.append(f"\n  Quality score: {self.quality_score:.0f}/100")
        return "\n".join(lines)

    def rhea_summary(self) -> str:
        """Rhea-style review summary."""
        lines = []
        for check in self.passed_checks:
            lines.append(f"  ✅ {check}")
        for e in self.errors:
            lines.append(f"  ❌ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        lines.append(f"\n  Quality score: {self.quality_score:.0f}/100")
        return "\n".join(lines)


def validate_task(
    question: str,
    answer: str,
    answer_format: str = "",
    image_path: str = "",
    caption: str = "",
    options: list[str] | None = None,
    figure_type: str = "",
    task_type: str = "",
) -> ValidationResult:
    """Validate a task against all handbook rules."""
    logger.info("validate_task entry q_len=%d a_len=%d fmt=%s ftype=%s ttype=%s",
                len(question), len(answer), answer_format, figure_type, task_type)
    result = ValidationResult()
    q = question.strip()
    a = answer.strip().lower()

    _run_format_checks(result, q, a, answer_format)
    _run_content_checks(result, q, a, answer_format)
    _run_complexity_checks(result, q, figure_type, task_type)
    _run_handbook_basics(result, q, a, caption)
    _run_visual_tests(result, q, a)
    _run_handbook_errors(result, q, a, options, figure_type, task_type, image_path)

    result.quality_score = _calculate_score(result)
    logger.info("validate_task result valid=%s score=%.1f errors=%d warnings=%d",
                result.is_valid, result.quality_score, len(result.errors), len(result.warnings))
    return result


def validate_mcq(
    question: str,
    answer: str,
    options: list[str],
    answer_format: str = "word",
    image_path: str = "",
    caption: str = "",
) -> ValidationResult:
    """Convenience wrapper for MCQ validation including options check.

    Per handbook §5, MCQ must have 8+ options with same format and no
    'none' / 'cannot be determined' distractors.
    """
    return validate_task(
        question=question,
        answer=answer,
        answer_format=answer_format,
        image_path=image_path,
        caption=caption,
        options=options,
    )


def _calculate_score(result: ValidationResult) -> float:
    """Calculate quality score (0-100) based on validation results."""
    score = 100.0
    score -= len(result.errors) * 20
    score -= len(result.warnings) * 5
    good_patterns = [
        "multi-step reasoning",
        "threshold filters",
        "multiple panels",
        "not extreme-seeking",
    ]
    for check in result.passed_checks:
        for pattern in good_patterns:
            if pattern.lower() in check.lower():
                score += 5
    return max(0, min(100, score))
