"""Authoring module: manual Q&A entry and AI-assisted drafting."""

from __future__ import annotations

import json

from ..models import AnswerFormat, Task, TaskStatus, TaskType


def log_task_event(
    task_id: int,
    event_type: str,
    details: dict | None = None,
    quality_score: float = 0.0,
) -> None:
    """Append an event to the task audit trail (TaskEvent table)."""
    from ..db import get_session
    from ..models import TaskEvent

    session = get_session()
    try:
        event = TaskEvent(
            task_id=task_id,
            event_type=event_type,
            details=json.dumps(details or {}),
            quality_score=quality_score,
        )
        session.add(event)
        session.commit()
    finally:
        session.close()


def create_task(
    figure_id: int,
    question: str,
    answer: str,
    answer_format: str = AnswerFormat.WORD.value,
    task_type: str = TaskType.CHART.value,
    domain: str = "Computer Science",
    ai_generated: bool = False,
    title: str = "",
    image_path: str = "",
) -> Task:
    """Create a new task draft."""
    from ..db import get_session
    from ..models import Figure

    session = get_session()
    try:
        # Auto-generate title from caption if not provided
        if not title:
            figure = session.get(Figure, figure_id)
            if figure and figure.caption:
                title = figure.caption[:60].strip()
            else:
                title = question[:60].strip()

        # Auto-set image_path from figure if not provided
        if not image_path:
            figure = session.get(Figure, figure_id)
            if figure:
                image_path = figure.image_path

        task = Task(
            figure_id=figure_id,
            title=title,
            image_path=image_path,
            question=question.strip(),
            answer=answer.strip(),
            answer_format=answer_format,
            task_type=task_type,
            domain=domain,
            ai_generated=ai_generated,
            status=TaskStatus.DRAFT.value,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        return task
    finally:
        session.close()


def update_task(task_id: int, **kwargs) -> Task | None:
    """Update fields on an existing task."""
    from ..db import get_session

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return None

        old_values = {k: str(getattr(task, k, "")) for k in kwargs if hasattr(task, k)}

        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        session.add(task)
        session.commit()
        session.refresh(task)

        log_task_event(
            task_id,
            "update",
            {
                "changed_fields": list(kwargs.keys()),
                "old_values": old_values,
                "new_values": {k: str(v) for k, v in kwargs.items()},
            },
        )
        return task
    finally:
        session.close()


def get_task(task_id: int) -> Task | None:
    """Get a task by ID."""
    from ..db import get_session

    session = get_session()
    try:
        return session.get(Task, task_id)
    finally:
        session.close()


def delete_task(task_id: int) -> bool:
    """Delete a task and all related records (generation_attempts, issue_reports, submission_logs)."""
    from sqlmodel import delete

    from ..db import get_session
    from ..models import GenerationAttempt, IssueReport, SubmissionLog

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return False

        # Log event before deletion (needs task data still available)
        log_task_event(
            task_id,
            "delete",
            {"figure_id": task.figure_id, "status": task.status, "difficulty": task.difficulty},
        )

        session.exec(delete(GenerationAttempt).where(GenerationAttempt.task_id == task_id))
        session.exec(delete(IssueReport).where(IssueReport.task_id == task_id))
        session.exec(delete(SubmissionLog).where(SubmissionLog.task_id == task_id))
        session.delete(task)
        session.commit()
        return True
    finally:
        session.close()


def list_tasks(status: str | None = None, limit: int = 50) -> list[Task]:
    """List tasks, optionally filtered by status."""
    from sqlmodel import select

    from ..db import get_session

    session = get_session()
    try:
        query = select(Task)
        if status:
            query = query.where(Task.status == status)
        query = query.order_by(Task.created_at.desc()).limit(limit)
        return list(session.exec(query).all())
    finally:
        session.close()
