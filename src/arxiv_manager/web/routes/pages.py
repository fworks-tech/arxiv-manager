"""Static page GET route handlers."""

import math

from fastapi import Request
from fastapi.responses import HTMLResponse
from sqlmodel import func, select

from ...authoring.validator import validate_task
from ...db import get_session
from ...models import Figure, Task
from ...tracking import get_stats
from . import TEMPLATES, router

SORTABLE_COLUMNS = {
    "id": Task.id,
    "title": Task.title,
    "question": Task.question,
    "status": Task.status,
    "difficulty": Task.difficulty,
    "created_at": Task.created_at,
}

DEFAULT_PER_PAGE = 20


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

        return TEMPLATES.TemplateResponse(
            request,
            "images.html",
            {
                "figures": figures,
                "status_filter": status,
                "min_complexity": min_complexity,
                "figure_type_filter": figure_type,
            },
        )
    finally:
        session.close()


@router.get("/tasks", response_class=HTMLResponse)
def tasks_page(request: Request, status: str = ""):
    """Tasks list page with initial table data."""
    session = get_session()
    try:
        query = select(Task)
        count_query = select(func.count()).select_from(Task)
        if status:
            query = query.where(Task.status == status)
            count_query = count_query.where(Task.status == status)

        total = session.exec(count_query).one()
        total_pages = max(1, math.ceil(total / DEFAULT_PER_PAGE))
        query = query.order_by(Task.id.desc()).limit(DEFAULT_PER_PAGE)
        tasks = list(session.exec(query).all())

        return TEMPLATES.TemplateResponse(
            request,
            "tasks.html",
            {
                "tasks": tasks,
                "total": total,
                "page": 1,
                "total_pages": total_pages,
                "per_page": DEFAULT_PER_PAGE,
                "sort": "id",
                "dir": "desc",
                "q": "",
                "status": status,
                "status_filter": status,
            },
        )
    finally:
        session.close()


@router.get("/tasks/table", response_class=HTMLResponse)
def tasks_table(
    request: Request,
    status: str = "",
    q: str = "",
    sort: str = "id",
    dir: str = "desc",
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
):
    """Tasks table partial — returns rendered table HTML for HTMX."""
    if sort not in SORTABLE_COLUMNS:
        sort = "id"
    if dir not in ("asc", "desc"):
        dir = "desc"
    if per_page not in (20, 50):
        per_page = DEFAULT_PER_PAGE
    page = max(1, page)

    session = get_session()
    try:
        query = select(Task)
        count_query = select(func.count()).select_from(Task)

        if status:
            query = query.where(Task.status == status)
            count_query = count_query.where(Task.status == status)
        if q:
            query = query.where(
                Task.title.contains(q)
                | Task.question.contains(q)
                | Task.answer.contains(q)
            )
            count_query = count_query.where(
                Task.title.contains(q)
                | Task.question.contains(q)
                | Task.answer.contains(q)
            )

        total = session.exec(count_query).one()
        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)

        sort_col = SORTABLE_COLUMNS[sort]
        order = sort_col.desc() if dir == "desc" else sort_col.asc()
        query = query.order_by(order)
        query = query.offset((page - 1) * per_page).limit(per_page)
        tasks = list(session.exec(query).all())

        return TEMPLATES.TemplateResponse(
            request,
            "tasks_table.html",
            {
                "tasks": tasks,
                "total": total,
                "page": page,
                "total_pages": total_pages,
                "per_page": per_page,
                "sort": sort,
                "dir": dir,
                "q": q,
                "status": status,
            },
        )
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

        return TEMPLATES.TemplateResponse(
            request,
            "task_form.html",
            {
                "figure": figure,
                "validation": None,
                "task": None,
            },
        )
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

        return TEMPLATES.TemplateResponse(
            request,
            "task_form.html",
            {
                "figure": figure,
                "task": task,
                "validation": validation,
            },
        )
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
