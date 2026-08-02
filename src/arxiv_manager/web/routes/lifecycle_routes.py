"""Task lifecycle, figure status, and Rhea review route handlers."""

import logging

from fastapi import Form, Request
from fastapi.responses import RedirectResponse

from ...authoring import log_task_event
from ...db import get_session
from ...models import Figure, Task
from ...tracking import mark_submitted, set_difficulty
from . import router

logger = logging.getLogger(__name__)


@router.post("/api/figure/{figure_id}/status")
def update_figure_status(figure_id: int, status: str = Form(...)):
    """Update figure status (HTMX endpoint)."""
    logger.info("figure status figure_id=%d -> %s", figure_id, status)
    session = get_session()
    try:
        figure = session.get(Figure, figure_id)
        if figure:
            figure.status = status
            session.add(figure)
            session.commit()
    finally:
        session.close()
    return RedirectResponse(url="/images", status_code=303)


@router.post("/api/figures/bulk-reject")
def bulk_reject_figures(figure_ids: list[int] = Form(default=[])):
    """Bulk reject multiple figures at once."""
    logger.info("bulk reject ids=%s", figure_ids)
    session = get_session()
    try:
        rejected = 0
        for fid in figure_ids:
            figure = session.get(Figure, fid)
            if figure:
                figure.status = "rejected"
                session.add(figure)
                rejected += 1
        session.commit()
    finally:
        session.close()
    return RedirectResponse(url="/images", status_code=303)


@router.post("/api/task/{task_id}/difficulty")
def update_task_difficulty(
    task_id: int,
    difficulty: str = Form(...),
    qwen: int = Form(0),
    gemini: int = Form(0),
    test_model_restriction: str = Form("all"),
):
    """Update task difficulty (HTMX endpoint)."""
    logger.info(
        "task difficulty task_id=%d difficulty=%s qwen=%d gemini=%d restriction=%s",
        task_id, difficulty, qwen, gemini, test_model_restriction,
    )
    from ...db import get_session as _get_session

    if test_model_restriction not in ("all", "qwen_only", "gemini_only"):
        test_model_restriction = "all"

    s = _get_session()
    try:
        t = s.get(Task, task_id)
        old_diff = t.difficulty if t else ""
        old_restriction = t.test_model_restriction if t else "all"
    finally:
        s.close()
    set_difficulty(task_id, difficulty, qwen, gemini)
    if old_restriction != test_model_restriction:
        from ...authoring import update_task as _update_task
        _update_task(task_id, test_model_restriction=test_model_restriction)
    log_task_event(
        task_id,
        "difficulty_change",
        {
            "old_difficulty": old_diff,
            "new_difficulty": difficulty,
            "qwen_passes": qwen,
            "gemini_passes": gemini,
            "test_model_restriction": test_model_restriction,
        },
    )
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.post("/api/task/{task_id}/submit")
def submit_task_route(task_id: int):
    """Mark task as submitted (HTMX endpoint)."""
    logger.info("task submit task_id=%d", task_id)
    mark_submitted(task_id)
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/api/task/{task_id}/verdict")
def record_verdict_route(task_id: int, verdict: str = Form(...), notes: str = Form("")):
    """Record a Realm evaluation verdict; auto-adjusts difficulty (warn-only)."""
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    from ...tracking import REALM_VERDICTS, record_realm_verdict

    logger.info("task verdict task_id=%d verdict=%s", task_id, verdict)
    if verdict not in REALM_VERDICTS:
        raise HTTPException(status_code=400, detail=f"verdict must be one of {REALM_VERDICTS}")
    task = record_realm_verdict(task_id, verdict, notes)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(
        {"ok": True, "verdict": verdict, "difficulty": task.difficulty}
    )


@router.post("/api/task/{task_id}/rhea")
def update_rhea(
    request: Request,
    task_id: int,
    rhea_reviewed: bool = Form(False),
    rhea_passed: bool = Form(False),
    rhea_notes: str = Form(""),
):
    """Update Rhea review status (HTMX endpoint)."""
    logger.info("task rhea task_id=%d reviewed=%s passed=%s", task_id, rhea_reviewed, rhea_passed)
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if task:
            task.rhea_reviewed = rhea_reviewed
            task.rhea_passed = rhea_passed
            task.rhea_notes = rhea_notes
            session.add(task)
            session.commit()
            log_task_event(
                task_id, "rhea_review",
                {"rhea_reviewed": rhea_reviewed, "rhea_passed": rhea_passed, "rhea_notes": rhea_notes},
            )
    finally:
        session.close()
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.post("/api/task/{task_id}/rhea-override")
def save_rhea_override(
    request: Request,
    task_id: int,
    rhea_override_notes: str = Form(...),
    rhea_passed: bool = Form(True),
):
    """Save author's override notes for a Rhea-rejected task."""
    logger.info("task rhea override task_id=%d passed=%s notes_len=%d", task_id, rhea_passed, len(rhea_override_notes))
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if task:
            task.rhea_override_notes = rhea_override_notes
            task.rhea_passed = rhea_passed
            session.add(task)
            session.commit()
    finally:
        session.close()
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)
