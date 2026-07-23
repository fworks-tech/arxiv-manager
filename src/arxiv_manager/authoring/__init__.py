"""Authoring module: manual Q&A entry and AI-assisted drafting."""

from __future__ import annotations

from ..models import Task, TaskStatus, AnswerFormat, TaskType


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


def update_task(task_id: int, **kwargs) -> Task | None:
    """Update fields on an existing task."""
    from ..db import get_session

    session = get_session()
    task = session.get(Task, task_id)
    if not task:
        return None

    for key, value in kwargs.items():
        if hasattr(task, key):
            setattr(task, key, value)

    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def get_task(task_id: int) -> Task | None:
    """Get a task by ID."""
    from ..db import get_session
    session = get_session()
    return session.get(Task, task_id)


def list_tasks(status: str | None = None, limit: int = 50) -> list[Task]:
    """List tasks, optionally filtered by status."""
    from ..db import get_session
    from sqlmodel import select

    session = get_session()
    query = select(Task)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.desc()).limit(limit)
    return list(session.exec(query).all())
