"""Static page GET route handlers."""
from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlmodel import select

from ...models import Figure, Task
from ...db import get_session
from ...authoring.validator import validate_task
from ...tracking import get_stats
from . import TEMPLATES, router


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Dashboard home."""
    stats = get_stats()
    return TEMPLATES.TemplateResponse(request, "base.html", {"stats": stats})


@router.get("/images", response_class=HTMLResponse)
def images_page(
    request: Request,
    status: str = "",
    min_complexity: float = 0,
    figure_type: str = "",
    suitable_only: bool = False,
):
    """Image library page."""
    session = get_session()
    try:
        query = select(Figure)
        if status:
            query = query.where(Figure.status == status)
        if min_complexity > 0:
            query = query.where(Figure.complexity_score >= min_complexity)
        if figure_type:
            query = query.where(Figure.figure_type == figure_type)
        if suitable_only:
            query = query.where(Figure.is_suitable == True)  # noqa: E712
        query = query.order_by(Figure.complexity_score.desc())
        figures = list(session.exec(query).all())

        return TEMPLATES.TemplateResponse(request, "images.html", {
            "figures": figures,
            "status_filter": status,
            "min_complexity": min_complexity,
            "figure_type_filter": figure_type,
        })
    finally:
        session.close()


@router.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request, status: str = ""):
    """Tasks list page."""
    session = get_session()
    try:
        query = select(Task)
        if status:
            query = query.where(Task.status == status)
        query = query.order_by(Task.created_at.desc())
        tasks = list(session.exec(query).all())

        return TEMPLATES.TemplateResponse(request, "tasks.html", {
            "tasks": tasks,
            "status_filter": status,
        })
    finally:
        session.close()


@router.get("/task/new/{figure_id}", response_class=HTMLResponse)
def task_form(request: Request, figure_id: int):
    """Task authoring form for a specific image."""
    session = get_session()
    try:
        figure = session.get(Figure, figure_id)
        if not figure:
            return HTMLResponse("Image not found", status_code=404)

        return TEMPLATES.TemplateResponse(request, "task_form.html", {
            "figure": figure,
            "validation": None,
            "task": None,
        })
    finally:
        session.close()


@router.get("/task/{task_id}", response_class=HTMLResponse)
def task_detail(request: Request, task_id: int):
    """View/edit an existing task."""
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return HTMLResponse("Task not found", status_code=404)
        figure = session.get(Figure, task.figure_id)

        validation = validate_task(task.question, task.answer, task.answer_format)

        return TEMPLATES.TemplateResponse(request, "task_form.html", {
            "figure": figure,
            "task": task,
            "validation": validation,
        })
    finally:
        session.close()


@router.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request):
    """Statistics dashboard."""
    stats = get_stats()
    return TEMPLATES.TemplateResponse(request, "stats.html", {"stats": stats})


@router.get("/author", response_class=HTMLResponse)
def author_page(request: Request):
    """Main upload + Q&A authoring page."""
    return TEMPLATES.TemplateResponse(request, "author.html", {})
