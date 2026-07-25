"""Task CRUD, validation, regeneration, and history route handlers."""
import json as _json
import logging

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from ...db import get_session
from ...models import Task, Figure, GenerationAttempt
from ...authoring import create_task, update_task
from ...authoring.validator import validate_task
from ...authoring.ai_draft import draft_qa, draft_with_self_critique
from ...authoring._draft_telemetry import log_generation_attempt
from ...storage import STORAGE_DIR
from . import TEMPLATES, router

logger = logging.getLogger(__name__)


@router.post("/api/task/create", response_class=HTMLResponse)
def api_create_task(
    request: Request,
    figure_id: int = Form(...),
    title: str = Form(""),
    domain: str = Form("Computer Science"),
    question: str = Form(...),
    answer: str = Form(...),
    answer_format: str = Form("word"),
    task_type: str = Form("chart"),
):
    """Create a new task (HTMX endpoint)."""
    logger.info("task create figure_id=%d type=%s format=%s", figure_id, task_type, answer_format)
    validation = validate_task(question, answer, answer_format)

    if validation.is_valid:
        task = create_task(
            figure_id=figure_id, title=title, domain=domain,
            question=question, answer=answer,
            answer_format=answer_format, task_type=task_type,
        )
        logger.info("task created id=%d", task.id)
        return RedirectResponse(url=f"/task/{task.id}", status_code=303)

    logger.warning("task create validation failed errors=%d", len(validation.errors))
    s = get_session()
    try:
        figure = s.get(Figure, figure_id)
    finally:
        s.close()
    return TEMPLATES.TemplateResponse(request, "task_form.html", {
        "figure": figure, "task": None, "validation": validation,
        "form_data": {"title": title, "domain": domain, "question": question,
                       "answer": answer, "answer_format": answer_format, "task_type": task_type},
    })


@router.post("/api/task/{task_id}/update", response_class=HTMLResponse)
def api_update_task(
    request: Request,
    task_id: int,
    title: str = Form(""),
    domain: str = Form("Computer Science"),
    question: str = Form(...),
    answer: str = Form(...),
    answer_format: str = Form("word"),
    task_type: str = Form("chart"),
):
    """Update an existing task (HTMX endpoint)."""
    logger.info("task update task_id=%d type=%s format=%s", task_id, task_type, answer_format)
    validation = validate_task(question, answer, answer_format)
    task = update_task(task_id, title=title, domain=domain, question=question,
                       answer=answer, answer_format=answer_format, task_type=task_type)
    figure = None
    if task:
        s = get_session()
        try:
            figure = s.get(Figure, task.figure_id)
        finally:
            s.close()
    logger.info("task updated id=%d valid=%s", task_id, validation.is_valid)

    return TEMPLATES.TemplateResponse(request, "task_form.html", {
        "figure": figure, "task": task, "validation": validation,
    })


@router.post("/api/task/{task_id}/validate", response_class=HTMLResponse)
def api_validate_task(request: Request, task_id: int):
    """Re-validate a task (HTMX endpoint)."""
    logger.info("task revalidate task_id=%d", task_id)
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return HTMLResponse("Not found", status_code=404)

        figure = session.get(Figure, task.figure_id)
        figure_type = getattr(figure, "figure_type", "") if figure else ""
        validation = validate_task(task.question, task.answer, task.answer_format,
                                   figure_type=figure_type, task_type=task.task_type)

        return TEMPLATES.TemplateResponse(request, "_validation.html", {"validation": validation})
    finally:
        session.close()


@router.get("/api/task/{task_id}/history", response_class=HTMLResponse)
def api_generation_history(request: Request, task_id: int):
    """Return generation history for a task's figure as HTML partial."""
    import json as _json

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return HTMLResponse("Task not found", status_code=404)

        rows = list(
            session.exec(
                select(GenerationAttempt)
                .where(
                    (GenerationAttempt.task_id == task_id)
                    | ((GenerationAttempt.figure_id == task.figure_id) & (GenerationAttempt.task_id != task_id))
                )
                .order_by(GenerationAttempt.created_at.desc())
                .limit(50)
            ).all()
        )

        attempts = []
        for r in rows:
            errors_list = _json.loads(r.validation_errors) if r.validation_errors and r.validation_errors.strip() not in ("", "[]") else []
            attempts.append({
                "generation_type": r.generation_type,
                "difficulty": r.difficulty or "",
                "model_name": r.model_name or "",
                "prompt_version_id": r.prompt_version_id or "",
                "generated_question": r.generated_question or "",
                "generated_answer": r.generated_answer or "",
                "generated_answer_format": r.generated_answer_format or "",
                "validation_quality": r.validation_quality,
                "errors": errors_list,
                "reasoning_trace": r.reasoning_trace or "",
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
            })

        return TEMPLATES.TemplateResponse(request, "_generation_history.html", {"attempts": attempts})
    finally:
        session.close()


def _do_regenerate(image_path, api_key, difficulty, figure_type, complexity, prev_question, figure_id):
    """Generate a single draft, with self-critique fallback for challenging/hardest."""
    try:
        if difficulty in ("challenging", "hardest"):
            draft = draft_with_self_critique(
                image_path=image_path, max_rounds=1,
                api_key=api_key, difficulty=difficulty,
                figure_type=figure_type, complexity_score=complexity,
                previous_question=prev_question, figure_id=figure_id,
            )
            if draft is None:
                draft = draft_qa(
                    image_path=image_path,
                    api_key=api_key, difficulty=difficulty,
                    figure_type=figure_type, complexity_score=complexity,
                    previous_question=prev_question, figure_id=figure_id,
                )
        else:
            draft = draft_qa(
                image_path=image_path,
                api_key=api_key, difficulty=difficulty,
                figure_type=figure_type, complexity_score=complexity,
                previous_question=prev_question, figure_id=figure_id,
            )
        return draft
    except ValueError:
        return None


def _log_attempt(figure_id, task_id, attempt_number, generation_type, draft, difficulty, figure_type, complexity, prev_question):
    """Log a generation attempt to telemetry."""
    log_generation_attempt(
        figure_id=figure_id, task_id=task_id, attempt_number=attempt_number,
        generation_type=generation_type, source_route="api_regenerate_task",
        prompt_template_name=f"{difficulty}_{figure_type}" if figure_type else difficulty,
        prompt_version_id=draft.get("_prompt_version_id", ""),
        prompt_text_hash=draft.get("_prompt_text_hash", ""),
        difficulty=difficulty, figure_type=figure_type, complexity_score=complexity,
        previous_question=prev_question,
        raw_response=draft.get("_raw_response", ""),
        reasoning_trace=draft.get("_reasoning_trace", ""),
        generated_question=draft.get("question", ""),
        generated_answer=draft.get("answer", ""),
        generated_answer_format=draft.get("answer_format", ""),
        generated_task_type=draft.get("task_type", ""),
        success=True,
    )


def _dedup_retry(img_path, api_key, difficulty, figure_type, complexity, prev_question, figure_id, task_answer, task_question):
    """If the draft matches the existing answer, retry up to 2 times."""
    for dedup_attempt in range(1, 3):
        draft2 = _do_regenerate(img_path, api_key, difficulty, figure_type, complexity, prev_question, figure_id)
        if draft2:
            _log_attempt(figure_id, figure_id, 1 + dedup_attempt, "dedup_retry",
                         draft2, difficulty, figure_type, complexity, prev_question)
            if draft2["answer"].strip().lower() != task_answer.strip().lower():
                return draft2
    return None


@router.post("/api/task/{task_id}/regenerate")
def api_regenerate_task(request: Request, task_id: int, difficulty: str = Form("challenging")):
    """Regenerate Q&A for a task using AI draft."""
    import os as os_mod

    logger.info("task regenerate task_id=%d difficulty=%s", task_id, difficulty)
    api_key = os_mod.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return {"error": "No OPENCODE_API_KEY set", "ok": False}

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return {"error": "Task not found", "ok": False}

        img_path = STORAGE_DIR / task.image_path
        if not img_path.exists():
            return {"error": "Image not found", "ok": False}

        figure = session.get(Figure, task.figure_id) if task.figure_id else None
        figure_type = getattr(figure, "figure_type", "") if figure else ""
        complexity = getattr(figure, "complexity_score", 0.0) if figure else 0.0
        prev_question = task.question

        draft = _do_regenerate(img_path, api_key, difficulty, figure_type, complexity, prev_question, task.figure_id)
        if not draft:
            return {"error": "Draft generation failed", "ok": False}

        _log_attempt(task.figure_id, task.id, 1, "regenerate_initial", draft, difficulty, figure_type, complexity, prev_question)

        # Dedup retries if answer unchanged
        if draft["answer"].strip().lower() == task.answer.strip().lower() or draft["question"].strip().lower() == task.question.strip().lower():
            better = _dedup_retry(img_path, api_key, difficulty, figure_type, complexity, prev_question, task.figure_id, task.answer, task.question)
            if better:
                draft = better

        # Apply and commit
        task.question = draft["question"]
        task.answer = draft["answer"]
        task.answer_format = draft.get("answer_format", "number")
        task.task_type = draft.get("task_type", "chart")
        task.difficulty = difficulty
        session.add(task)
        session.commit()
        logger.info("task regenerate ok task_id=%d", task_id)

        _log_attempt(task.figure_id, task.id, 2, "regenerate_final", draft, difficulty, figure_type, complexity, prev_question)

        return {
            "ok": True, "question": draft["question"], "answer": draft["answer"],
            "answer_format": draft.get("answer_format", "number"),
            "task_type": draft.get("task_type", "chart"),
        }
    finally:
        session.close()
