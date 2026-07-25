"""Structured history injection for smarter Q&A generation.

Queries the GenerationAttempt table to build informative context
about past attempts for a figure, and fetches few-shot examples
from successful past generations.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlmodel import select, desc

logger = logging.getLogger(__name__)


def _deduplicate_examples(rows, limit: int) -> list[dict[str, Any]]:
    """Deduplicate GenerationAttempt rows by question text and format as dicts."""
    seen_questions: set[str] = set()
    examples: list[dict[str, Any]] = []
    for r in rows:
        q = r.generated_question.strip().lower()
        if q not in seen_questions:
            seen_questions.add(q)
            examples.append({
                "question": r.generated_question,
                "answer": r.generated_answer,
                "answer_format": r.generated_answer_format,
                "task_type": r.generated_task_type,
                "quality": r.validation_quality,
                "figure_type": r.figure_type,
                "difficulty": r.difficulty,
                "complexity": r.complexity_score,
            })
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
    with validation_quality >= 80. Falls back to relaxing filters.
    Returns empty list if no data or table doesn't exist yet.
    """
    from ..db import get_session
    from ..models import GenerationAttempt

    session = get_session()
    try:
        query = select(GenerationAttempt).where(
            GenerationAttempt.success == True,
            GenerationAttempt.validation_is_valid == True,
            GenerationAttempt.validation_quality >= 80,
            GenerationAttempt.generated_question != "",
        )

        if figure_type:
            query = query.where(GenerationAttempt.figure_type == figure_type)
        if difficulty:
            query = query.where(GenerationAttempt.difficulty == difficulty)
        if task_type:
            query = query.where(GenerationAttempt.generated_task_type == task_type)

        # Prefer closest complexity match
        if complexity_score > 0:
            query = query.order_by(
                abs(GenerationAttempt.complexity_score - complexity_score)
            )
        else:
            query = query.order_by(desc(GenerationAttempt.validation_quality))

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
    from ..models import GenerationAttempt

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

        if not rows:
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
            blocks.append(" | ".join(parts))

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
) -> str:
    """Pick the best-performing model for a given (figure_type, difficulty).

    Queries past GenerationAttempt records and returns the model with the
    highest average validation quality. Falls back to default_model if
    insufficient data exists.
    """
    from ..db import get_session
    from ..models import GenerationAttempt

    session = get_session()
    try:
        query = select(GenerationAttempt).where(
            GenerationAttempt.success == True,
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
            model_scores.setdefault(r.model_name, []).append(r.validation_quality)

        best_model = default_model
        best_avg = 0.0
        for model_name, scores in model_scores.items():
            if len(scores) >= min_attempts:
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_model = model_name

        if best_model != default_model:
            logger.info("select_best_model: %s over %s (avg=%.1f, n=%d) for figure_type=%s difficulty=%s",
                        best_model, default_model, best_avg, len(model_scores.get(best_model, [])),
                        figure_type, difficulty)

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
) -> str:
    """Enrich a prompt with figure history and few-shot examples.

    Appends three sections to the base prompt if data is available:
    1. Past generation history for this figure
    2. Few-shot examples from successful past generations
    3. Previous question reminder (existing mechanism, enhanced)
    """
    parts: list[str] = []

    # 1. Figure history
    if figure_id is not None:
        history = build_figure_history(figure_id)
        if history:
            parts.append(
                "PREVIOUS ATTEMPTS FOR THIS FIGURE (DO NOT repeat failed patterns):\n"
                + history
            )

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
            "HIGH-QUALITY EXAMPLES from similar figures (emulate these patterns):\n"
            + "\n---\n".join(ex_blocks)
        )

    # 3. Previous question (existing mechanism, but enhanced with answer)
    if previous_question:
        parts.append(
            f"The previous question for this image was: {previous_question}\n"
            "Generate a SUBSTANTIALLY DIFFERENT question — different strategy, "
            "different data references, different answer."
        )

    if not parts:
        return base_prompt

    return base_prompt + "\n\n" + "\n\n".join(parts)
