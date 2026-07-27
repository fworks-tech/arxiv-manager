"""Structured history injection for smarter Q&A generation.

Queries the GenerationAttempt table to build informative context
about past attempts for a figure, and fetches few-shot examples
from successful past generations.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlmodel import desc, select

logger = logging.getLogger(__name__)


def _deduplicate_examples(rows, limit: int) -> list[dict[str, Any]]:
    """Deduplicate GenerationAttempt rows by question text and format as dicts."""
    seen_questions: set[str] = set()
    examples: list[dict[str, Any]] = []
    for r in rows:
        q = r.generated_question.strip().lower()
        if q not in seen_questions:
            seen_questions.add(q)
            examples.append(
                {
                    "question": r.generated_question,
                    "answer": r.generated_answer,
                    "answer_format": r.generated_answer_format,
                    "task_type": r.generated_task_type,
                    "quality": r.validation_quality,
                    "figure_type": r.figure_type,
                    "difficulty": r.difficulty,
                    "complexity": r.complexity_score,
                }
            )
            if len(examples) >= limit:
                break
    return examples


def get_few_shot_examples(
    figure_type: str = "",
    difficulty: str = "",
    complexity_score: float = 0.0,
    task_type: str = "",
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Fetch high-quality past generation attempts as few-shot examples.

    Matches on figure_type, difficulty, task_type and prefers attempts
    with validation_quality >= 80. Excludes user-reported issues.
    Prefers Rhea-approved examples. Falls back to relaxing filters.
    Returns empty list if no data or table doesn't exist yet.
    """
    from ..db import get_session
    from ..models import GenerationAttempt, IssueReport

    session = get_session()
    try:
        # Get IDs of reported issues to exclude
        reported_ids = set()
        try:
            reported = list(session.exec(select(IssueReport.generation_attempt_id)).all())
            reported_ids = {r for r in reported if r is not None}
        except Exception:
            pass

        query = select(GenerationAttempt).where(
            GenerationAttempt.success,
            GenerationAttempt.validation_is_valid,
            GenerationAttempt.validation_quality >= 80,
            GenerationAttempt.generated_question != "",
        )
        if reported_ids:
            query = query.where(~GenerationAttempt.id.in_(reported_ids))

        if figure_type:
            query = query.where(GenerationAttempt.figure_type == figure_type)
        if difficulty:
            query = query.where(GenerationAttempt.difficulty == difficulty)
        if task_type:
            query = query.where(GenerationAttempt.generated_task_type == task_type)

        # Order by: closest complexity match first, then Rhea-approved, then Qwen-failed/Gemini-passed
        if complexity_score > 0:
            query = query.order_by(abs(GenerationAttempt.complexity_score - complexity_score))
        else:
            query = query.order_by(desc(GenerationAttempt.validation_quality))
        if reported_ids:
            query = query.order_by(GenerationAttempt.rhea_passed.desc())
        query = query.order_by(GenerationAttempt.qwen_passes.asc())
        query = query.order_by(GenerationAttempt.gemini_passes.desc())

        rows = list(session.exec(query.limit(limit * 2)).all())

        return _deduplicate_examples(rows, limit)
    except Exception:
        logger.debug("get_few_shot_examples: no data yet (table may not exist)")
        return []
    finally:
        session.close()


def build_figure_history(figure_id: int, max_attempts: int = 5) -> str:
    """Build a structured history block for past generation attempts on a figure.

    Returns a formatted string suitable for prompt injection, or empty string
    if no history exists.
    """
    from ..db import get_session
    from ..models import GenerationAttempt, IssueReport

    session = get_session()
    try:
        rows = list(
            session.exec(
                select(GenerationAttempt)
                .where(GenerationAttempt.figure_id == figure_id)
                .order_by(desc(GenerationAttempt.created_at))
                .limit(max_attempts)
            ).all()
        )

        # Query IssueReport feedback for this figure
        seen_feedback: set[str] = set()
        reports: dict[int, list[str]] = {}
        general_reports: list[str] = []
        try:
            issue_rows = list(
                session.exec(
                    select(IssueReport).where(IssueReport.figure_id == figure_id).order_by(IssueReport.created_at.asc())
                ).all()
            )
            for r in issue_rows:
                line = f"reason={r.reason}"
                if r.description:
                    line += f" ({r.description})"
                if r.corrected_answer:
                    line += f" | CORRECTED ANSWER: {r.corrected_answer}"
                if line in seen_feedback:
                    continue
                seen_feedback.add(line)
                if r.generation_attempt_id:
                    reports.setdefault(r.generation_attempt_id, []).append(line)
                else:
                    general_reports.append(line)
        except Exception:
            pass

        if not rows and not general_reports:
            return ""

        blocks: list[str] = []
        for i, r in enumerate(rows, 1):
            parts = [f"Attempt {i}: {r.generation_type}"]
            if r.difficulty:
                parts.append(f"difficulty={r.difficulty}")
            if r.generated_question:
                parts.append(f"question={r.generated_question}")
            if r.generated_answer:
                parts.append(f"answer={r.generated_answer}")
            if r.validation_quality > 0:
                parts.append(f"quality={r.validation_quality:.0f}")
            if r.validation_errors and r.validation_errors != "[]":
                import json

                try:
                    errs = json.loads(r.validation_errors)
                    if errs:
                        parts.append(f"errors={'; '.join(errs[:3])}")
                except json.JSONDecodeError:
                    pass
            if r.critique_score > 0:
                parts.append(f"critique_score={r.critique_score}")
            if r.qwen_passes > 0 or r.gemini_passes > 0:
                parts.append(
                    f"model_runs: Qwen={r.qwen_passes}/{r.qwen_passes + r.gemini_passes} Gemini={r.gemini_passes}/{r.qwen_passes + r.gemini_passes}"
                )
            blocks.append(" | ".join(parts))
            if r.id in reports:
                for fb_line in reports[r.id]:
                    blocks.append(f"  ⚠️ USER FEEDBACK on this attempt: {fb_line}")

        for line in general_reports:
            blocks.append(f"  ⚠️ USER FEEDBACK: {line}")

        return "\n".join(blocks)
    except Exception:
        logger.debug("build_figure_history: no data yet (table may not exist)")
        return ""
    finally:
        session.close()


def select_best_model(
    figure_type: str = "",
    difficulty: str = "",
    default_model: str = "minimax-m3",
    min_attempts: int = 3,
    allowed_models: set[str] | None = None,
) -> str:
    """Pick the best-performing model for a given (figure_type, difficulty).

    Queries past GenerationAttempt records and returns the model with the
    highest average validation quality. Falls back to default_model if
    insufficient data exists. Only considers models in allowed_models if set.
    """
    from ..db import get_session
    from ..models import GenerationAttempt

    session = get_session()
    try:
        query = select(GenerationAttempt).where(
            GenerationAttempt.success,
            GenerationAttempt.validation_quality > 0,
            GenerationAttempt.model_name != "",
        )
        if figure_type:
            query = query.where(GenerationAttempt.figure_type == figure_type)
        if difficulty:
            query = query.where(GenerationAttempt.difficulty == difficulty)

        rows = list(session.exec(query).all())

        if len(rows) < min_attempts:
            return default_model

        model_scores: dict[str, list[float]] = {}
        for r in rows:
            if allowed_models and r.model_name not in allowed_models:
                continue
            model_scores.setdefault(r.model_name, []).append(r.validation_quality)

        if not model_scores:
            return default_model

        best_model = default_model
        best_avg = 0.0
        for model_name, scores in model_scores.items():
            if len(scores) >= min_attempts:
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_model = model_name

        if best_model != default_model:
            logger.info(
                "select_best_model: %s over %s (avg=%.1f, n=%d) for figure_type=%s difficulty=%s",
                best_model,
                default_model,
                best_avg,
                len(model_scores.get(best_model, [])),
                figure_type,
                difficulty,
            )

        return best_model
    except Exception:
        logger.debug("select_best_model: no data yet (table may not exist)")
        return default_model
    finally:
        session.close()


def inject_history_into_prompt(
    base_prompt: str,
    figure_id: int | None = None,
    figure_type: str = "",
    difficulty: str = "",
    complexity_score: float = 0.0,
    task_type: str = "",
    previous_question: str = "",
    validation_context: str = "",
) -> str:
    """Enrich a prompt with figure history and few-shot examples.

    Appends sections to the base prompt if data is available:
    1. Past generation history for this figure (including user issue reports)
    2. Few-shot examples from successful past generations
    3. Previous question reminder
    4. Current task validation issues to fix
    """
    parts: list[str] = []

    # 1. Figure history
    if figure_id is not None:
        history = build_figure_history(figure_id)
        if history:
            parts.append("PREVIOUS ATTEMPTS FOR THIS FIGURE (DO NOT repeat failed patterns):\n" + history)

    # 2. Few-shot examples from similar successful generations
    examples = get_few_shot_examples(
        figure_type=figure_type,
        difficulty=difficulty,
        complexity_score=complexity_score,
        task_type=task_type,
        limit=2,
    )
    if examples:
        ex_blocks: list[str] = []
        for ex in examples:
            ex_blocks.append(
                f"Q: {ex['question']}\nA: {ex['answer']} (format={ex['answer_format']}, type={ex['task_type']}, quality={ex['quality']:.0f})"
            )
        parts.append(
            "HIGH-QUALITY EXAMPLES from similar figures (emulate these patterns):\n" + "\n---\n".join(ex_blocks)
        )

    # 3. Previous question (existing mechanism, but enhanced with answer)
    if previous_question:
        parts.append(
            f"The previous question for this image was: {previous_question}\n"
            "Generate a SUBSTANTIALLY DIFFERENT question — different strategy, "
            "different data references, different answer."
        )

    # 4. Current task validation issues to fix
    if validation_context:
        parts.append("VALIDATION ISSUES TO FIX in the current task:\n" + validation_context)

    if not parts:
        return base_prompt

    return base_prompt + "\n\n" + "\n\n".join(parts)
