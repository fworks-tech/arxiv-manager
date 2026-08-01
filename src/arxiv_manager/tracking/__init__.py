"""Task tracking, difficulty classification, and export."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import select

from ..authoring import log_task_event
from ..db import get_session
from ..models import Difficulty, Figure, SubmissionLog, Task, TaskStatus


def set_difficulty(
    task_id: int,
    difficulty: str,
    qwen_passes: int = 0,
    gemini_passes: int = 0,
) -> Task | None:
    """Set difficulty and model pass counts for a task."""
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return None

        task.difficulty = difficulty
        task.qwen_passes = qwen_passes
        task.gemini_passes = gemini_passes
        task.total_runs = qwen_passes + gemini_passes

        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    finally:
        session.close()


def classify_difficulty(qwen_passes: int, gemini_passes: int) -> str:
    """Classify difficulty based on model rollout results.

    - Qwen passes any → EASY
    - Qwen fails all, Gemini passes any → CHALLENGING
    - Both fail all → HARDEST
    """
    if qwen_passes > 0:
        return Difficulty.EASY.value
    if gemini_passes > 0:
        return Difficulty.CHALLENGING.value
    return Difficulty.HARDEST.value


def mark_submitted(task_id: int, platform_task_id: str = "") -> Task | None:
    """Mark a task as submitted. Captures Rhea state onto GenerationAttempt records."""
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return None

        # Capture Rhea feedback on all attached GenerationAttempt records
        from ..models import GenerationAttempt

        attempts = list(session.exec(select(GenerationAttempt).where(GenerationAttempt.task_id == task_id)).all())
        for a in attempts:
            a.rhea_passed = task.rhea_passed
            a.rhea_notes = task.rhea_notes[:500]
            a.qwen_passes = task.qwen_passes
            a.gemini_passes = task.gemini_passes
            session.add(a)

        task.status = TaskStatus.SUBMITTED.value
        task.submitted_at = datetime.now()
        task.platform_task_id = platform_task_id

        # Log submission
        log = SubmissionLog(task_id=task_id)
        session.add(log)

        session.add(task)
        session.commit()
        session.refresh(task)

        log_task_event(
            task_id,
            "submit",
            {"qwen_passes": task.qwen_passes, "gemini_passes": task.gemini_passes, "total_runs": task.total_runs},
        )
        return task
    finally:
        session.close()


def mark_status(task_id: int, status: str, notes: str = "") -> Task | None:
    """Update task status (approved, rework, etc.)."""
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return None

        task.status = status

        # Update latest submission log
        log = session.exec(
            select(SubmissionLog).where(SubmissionLog.task_id == task_id).order_by(SubmissionLog.submitted_at.desc())
        ).first()
        if log:
            log.review_status = status
            log.reviewer_notes = notes
            session.add(log)

        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    finally:
        session.close()


REALM_VERDICTS = ("too_easy", "too_hard", "approved")
_DIFFICULTY_DOWNGRADE = {"hardest": "challenging", "challenging": "easy", "easy": "easy"}
_DIFFICULTY_UPGRADE = {"easy": "challenging", "challenging": "hardest", "hardest": "hardest"}


def record_realm_verdict(task_id: int, verdict: str, notes: str = "") -> Task | None:
    """Record a Realm evaluation verdict and auto-adjust difficulty (warn-only).

    too_easy  → downgrade one difficulty tier (the question is easier than claimed)
    too_hard  → upgrade one tier
    approved  → leave difficulty untouched

    Warn-only by design: the author can override the new difficulty afterwards.
    """
    if verdict not in REALM_VERDICTS:
        raise ValueError(f"verdict must be one of {REALM_VERDICTS}")

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return None

        log = session.exec(
            select(SubmissionLog).where(SubmissionLog.task_id == task_id).order_by(SubmissionLog.submitted_at.desc())
        ).first()
        if not log:
            log = SubmissionLog(task_id=task_id)
        log.review_status = verdict
        log.reviewer_notes = notes[:2000]
        session.add(log)

        old_difficulty = task.difficulty
        new_difficulty = old_difficulty
        if verdict == "too_easy":
            new_difficulty = _DIFFICULTY_DOWNGRADE.get(old_difficulty, old_difficulty)
        elif verdict == "too_hard":
            new_difficulty = _DIFFICULTY_UPGRADE.get(old_difficulty, old_difficulty)

        difficulty_changed = new_difficulty != old_difficulty
        if difficulty_changed:
            task.difficulty = new_difficulty
            session.add(task)

        session.commit()
        session.refresh(task)

        log_task_event(
            task_id,
            "realm_verdict",
            {
                "verdict": verdict,
                "notes": notes[:500],
                "old_difficulty": old_difficulty,
                "new_difficulty": new_difficulty,
                "difficulty_changed": difficulty_changed,
            },
            quality_score=0.0 if verdict == "too_easy" else 1.0,
        )
        return task
    finally:
        session.close()


def get_stats() -> dict[str, Any]:
    """Get overall statistics."""
    session = get_session()
    try:
        all_tasks = list(session.exec(select(Task)).all())
        total = len(all_tasks)

        by_status: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        for t in all_tasks:
            by_status[t.status] = by_status.get(t.status, 0) + 1
            if t.difficulty:
                by_difficulty[t.difficulty] = by_difficulty.get(t.difficulty, 0) + 1

        figures = list(session.exec(select(Figure)).all())
        total_figures = len(figures)
        used_figures = sum(1 for f in figures if f.status == "used")

        return {
            "total_tasks": total,
            "by_status": by_status,
            "by_difficulty": by_difficulty,
            "total_figures": total_figures,
            "used_figures": used_figures,
        }
    finally:
        session.close()


def export_task(task_id: int) -> dict[str, str] | None:
    """Export a task in a format ready for the platform.

    Returns dict with question, answer, and metadata for copy-paste.
    """
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return None

        figure = session.get(Figure, task.figure_id)

        return {
            "title": task.title,
            "domain": task.domain,
            "question": task.question,
            "answer": task.answer,
            "answer_format": task.answer_format,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "image_path": task.image_path or (figure.image_path if figure else ""),
            "caption": figure.caption if figure else "",
        }
    finally:
        session.close()
