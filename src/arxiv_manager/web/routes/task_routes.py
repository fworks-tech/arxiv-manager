"""Task CRUD, validation, regeneration, and history route handlers."""

import json as _json
import logging

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from ...authoring import create_task, update_task
from ...authoring._draft_telemetry import log_generation_attempt
from ...authoring.ai_draft import draft_qa, draft_with_self_critique
from ...authoring.validator import validate_task
from ...db import get_session
from ...models import Figure, GenerationAttempt, Task
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
            figure_id=figure_id,
            title=title,
            domain=domain,
            question=question,
            answer=answer,
            answer_format=answer_format,
            task_type=task_type,
        )
        logger.info("task created id=%d", task.id)
        return RedirectResponse(url=f"/task/{task.id}", status_code=303)

    logger.warning("task create validation failed errors=%d", len(validation.errors))
    s = get_session()
    try:
        figure = s.get(Figure, figure_id)
    finally:
        s.close()
    return TEMPLATES.TemplateResponse(
        request,
        "task_form.html",
        {
            "figure": figure,
            "task": None,
            "validation": validation,
            "form_data": {
                "title": title,
                "domain": domain,
                "question": question,
                "answer": answer,
                "answer_format": answer_format,
                "task_type": task_type,
            },
        },
    )


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
    task = update_task(
        task_id,
        title=title,
        domain=domain,
        question=question,
        answer=answer,
        answer_format=answer_format,
        task_type=task_type,
    )
    figure = None
    if task:
        s = get_session()
        try:
            figure = s.get(Figure, task.figure_id)
        finally:
            s.close()
    logger.info("task updated id=%d valid=%s", task_id, validation.is_valid)

    return TEMPLATES.TemplateResponse(
        request,
        "task_form.html",
        {
            "figure": figure,
            "task": task,
            "validation": validation,
        },
    )


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
        validation = validate_task(
            task.question, task.answer, task.answer_format, figure_type=figure_type, task_type=task.task_type
        )

        return TEMPLATES.TemplateResponse(request, "_validation.html", {"validation": validation})
    finally:
        session.close()


@router.get("/api/task/{task_id}/history", response_class=HTMLResponse)
def api_generation_history(request: Request, task_id: int):
    """Return generation history for a task's figure as HTML partial."""
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
        seen_qa: set[tuple[str, str]] = set()
        for r in rows:
            q_key = (r.generated_question or "").strip().lower(), (r.generated_answer or "").strip().lower()
            if q_key in seen_qa:
                continue
            seen_qa.add(q_key)
            errors_list = (
                _json.loads(r.validation_errors)
                if r.validation_errors and r.validation_errors.strip() not in ("", "[]")
                else []
            )
            attempts.append(
                {
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
                    "total_tokens": r.total_tokens or 0,
                    "input_tokens": r.input_tokens or 0,
                    "output_tokens": r.output_tokens or 0,
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
                }
            )

        return TEMPLATES.TemplateResponse(request, "_generation_history.html", {"attempts": attempts})
    finally:
        session.close()


def _do_regenerate(
    image_path, api_key, difficulty, figure_type, complexity, prev_question, figure_id, validation_context=""
):
    """Generate a single draft, with self-critique fallback for challenging/hardest."""
    try:
        if difficulty in ("challenging", "hardest"):
            draft = draft_with_self_critique(
                image_path=image_path,
                max_rounds=2,
                api_key=api_key,
                difficulty=difficulty,
                figure_type=figure_type,
                complexity_score=complexity,
                previous_question=prev_question,
                validation_context=validation_context,
                figure_id=figure_id,
            )
            if draft is None:
                draft = draft_qa(
                    image_path=image_path,
                    api_key=api_key,
                    difficulty=difficulty,
                    figure_type=figure_type,
                    complexity_score=complexity,
                    previous_question=prev_question,
                    validation_context=validation_context,
                    figure_id=figure_id,
                )
        else:
            draft = draft_qa(
                image_path=image_path,
                api_key=api_key,
                difficulty=difficulty,
                figure_type=figure_type,
                complexity_score=complexity,
                previous_question=prev_question,
                validation_context=validation_context,
                figure_id=figure_id,
            )
        return draft
    except ValueError:
        return None


def _log_attempt(
    figure_id,
    task_id,
    attempt_number,
    generation_type,
    draft,
    difficulty,
    figure_type,
    complexity,
    prev_question,
    model_name="",
):
    """Log a generation attempt to telemetry."""
    usage = draft.get("_usage", {})
    log_generation_attempt(
        figure_id=figure_id,
        task_id=task_id,
        attempt_number=attempt_number,
        generation_type=generation_type,
        source_route="api_regenerate_task",
        prompt_template_name=f"{difficulty}_{figure_type}" if figure_type else difficulty,
        prompt_version_id=draft.get("_prompt_version_id", ""),
        prompt_text_hash=draft.get("_prompt_text_hash", ""),
        model_name=model_name or draft.get("_model", ""),
        difficulty=difficulty,
        figure_type=figure_type,
        complexity_score=complexity,
        previous_question=prev_question,
        raw_response=draft.get("_raw_response", ""),
        reasoning_trace=draft.get("_reasoning_trace", ""),
        generated_question=draft.get("question", ""),
        generated_answer=draft.get("answer", ""),
        generated_answer_format=draft.get("answer_format", ""),
        generated_task_type=draft.get("task_type", ""),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        success=True,
    )


def _dedup_retry(
    img_path,
    api_key,
    difficulty,
    figure_type,
    complexity,
    prev_question,
    figure_id,
    task_id,
    task_answer,
    task_question,
    validation_context="",
):
    """If the draft matches the existing answer, retry up to 2 times."""
    for dedup_attempt in range(1, 3):
        draft2 = _do_regenerate(
            img_path, api_key, difficulty, figure_type, complexity, prev_question, figure_id, validation_context
        )
        if draft2:
            _log_attempt(
                figure_id,
                task_id,
                1 + dedup_attempt,
                "dedup_retry",
                draft2,
                difficulty,
                figure_type,
                complexity,
                prev_question,
                draft2.get("_model", difficulty),
            )
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

        # Validate current task and build context for the model
        v = validate_task(
            task.question,
            task.answer,
            task.answer_format,
            figure_type=figure_type,
            task_type=task.task_type,
        )
        validation_context = ""
        if v.errors:
            validation_context += "Errors: " + "; ".join(v.errors[:3])
        if v.warnings:
            if validation_context:
                validation_context += " | "
            validation_context += "Warnings: " + "; ".join(v.warnings[:3])

        draft = _do_regenerate(
            img_path, api_key, difficulty, figure_type, complexity, prev_question, task.figure_id, validation_context
        )
        if not draft:
            return {"error": "Draft generation failed", "ok": False}

        model_name = draft.get("_model", difficulty)
        _log_attempt(
            task.figure_id,
            task.id,
            1,
            "regenerate_initial",
            draft,
            difficulty,
            figure_type,
            complexity,
            prev_question,
            model_name,
        )

        # Dedup retries if answer unchanged
        if (
            draft["answer"].strip().lower() == task.answer.strip().lower()
            or draft["question"].strip().lower() == task.question.strip().lower()
        ):
            better = _dedup_retry(
                img_path,
                api_key,
                difficulty,
                figure_type,
                complexity,
                prev_question,
                task.figure_id,
                task.id,
                task.answer,
                task.question,
                validation_context,
            )
            if better:
                draft = better
                model_name = draft.get("_model", difficulty)

        # Apply and commit
        task.question = draft["question"]
        task.answer = draft["answer"]
        task.answer_format = draft.get("answer_format", "number")
        task.task_type = draft.get("task_type", "chart")
        task.difficulty = difficulty
        session.add(task)
        session.commit()
        logger.info("task regenerate ok task_id=%d", task_id)

        _log_attempt(
            task.figure_id,
            task.id,
            2,
            "regenerate_final",
            draft,
            difficulty,
            figure_type,
            complexity,
            prev_question,
            model_name,
        )

        usage = draft.get("_usage", {})
        model = model_name
        from ...observability.cost_tracker import estimate_cost
        from ...observability.cost_tracker import format_cost as _fmt_cost

        tok_in = usage.get("input_tokens", 0)
        tok_out = usage.get("output_tokens", 0)
        cost = _fmt_cost(estimate_cost(model, tok_in, tok_out))

        return {
            "ok": True,
            "question": draft["question"],
            "answer": draft["answer"],
            "answer_format": draft.get("answer_format", "number"),
            "task_type": draft.get("task_type", "chart"),
            "model": model,
            "input_tokens": tok_in,
            "output_tokens": tok_out,
            "total_tokens": tok_in + tok_out,
            "cost": cost,
        }
    finally:
        session.close()


@router.post("/api/task/{task_id}/ai-fix", response_class=HTMLResponse)
def api_ai_fix(request: Request, task_id: int):
    """Use LLM to suggest a fix for validation errors on an existing task."""
    import os as os_mod

    from ...authoring._draft_config import CONFIG
    from ...authoring._draft_prompts import FIX_PROMPT
    from ...authoring._history_context import inject_history_into_prompt
    from ...authoring.ai_draft._api_client import _call_opencode as _call
    from ...authoring.ai_draft._response_parser import _parse_llm_response

    api_key = os_mod.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return HTMLResponse("<div class='text-red-500 p-4'>No API key set</div>")

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return HTMLResponse("Not found", status_code=404)

        figure = session.get(Figure, task.figure_id) if task.figure_id else None
        figure_type = getattr(figure, "figure_type", "") if figure else ""
        complexity = getattr(figure, "complexity_score", 0.0) if figure else 0.0
        validation = validate_task(
            task.question,
            task.answer,
            task.answer_format,
            figure_type=figure_type,
            task_type=task.task_type,
        )

        validation_context = ""
        if validation.errors:
            validation_context += "Errors: " + "; ".join(validation.errors[:3])
        if validation.warnings:
            if validation_context:
                validation_context += " | "
            validation_context += "Warnings: " + "; ".join(validation.warnings[:3])

        fix_prompt = FIX_PROMPT.text.format(
            question=task.question,
            answer=task.answer,
            answer_format=task.answer_format,
            task_type=task.task_type,
            errors="\n".join(f"- {e}" for e in validation.errors) if validation.errors else "None",
            warnings="\n".join(f"- {w}" for w in validation.warnings) if validation.warnings else "None",
        )

        fix_prompt = inject_history_into_prompt(
            fix_prompt,
            figure_id=task.figure_id,
            figure_type=figure_type,
            difficulty=task.difficulty,
            complexity_score=complexity,
            previous_question=task.question,
            validation_context=validation_context,
        )

        import base64 as _b64
        import io as _io

        from PIL import Image as PILImage

        img_path = STORAGE_DIR / task.image_path
        b64_image = ""
        if img_path.exists():
            with PILImage.open(img_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.thumbnail(CONFIG.thumbnail_size)
                buf = _io.BytesIO()
                img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
            b64_image = _b64.b64encode(buf.getvalue()).decode()

        model = CONFIG.select_model(needs_image=True)
        result = _call(
            api_key,
            fix_prompt,
            b64_image,
            model=model,
            retries=1,
            difficulty=task.difficulty or "challenging",
            parser=_parse_llm_response,
        )

        if not result:
            return HTMLResponse(
                "<div class='text-red-500 p-4'>AI Fix failed — model did not return a valid response</div>"
            )

        return TEMPLATES.TemplateResponse(
            request,
            "_ai_fix.html",
            {
                "fix": {
                    "task_id": task_id,
                    "current_question": task.question,
                    "current_answer": task.answer,
                    "question": result.get("question", task.question),
                    "answer": result.get("answer", task.answer),
                    "answer_format": result.get("answer_format", task.answer_format),
                    "task_type": result.get("task_type", task.task_type),
                    "fix_summary": result.get("fix_summary", ""),
                    "model": model,
                }
            },
        )
    finally:
        session.close()


@router.post("/api/task/{task_id}/apply-fix", response_class=HTMLResponse)
def api_apply_fix(
    request: Request,
    task_id: int,
    question: str = Form(...),
    answer: str = Form(...),
    answer_format: str = Form("word"),
    task_type: str = Form("chart"),
):
    """Apply an AI-suggested fix to a task."""
    from ...authoring import update_task

    logger.info("task apply-fix task_id=%d", task_id)
    _v = validate_task(question, answer, answer_format, task_type=task_type)
    if not _v.is_valid and _v.quality_score < 30:
        return HTMLResponse(
            "<div class='text-red-500 p-4'>Fix rejected — validation score too low</div>", status_code=400
        )
    update_task(
        task_id,
        title="",
        domain="",
        question=question,
        answer=answer,
        answer_format=answer_format,
        task_type=task_type,
    )
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.get("/api/task/{task_id}/report-form", response_class=HTMLResponse)
def api_report_form(request: Request, task_id: int):
    """Return the issue report form for a task."""
    return TEMPLATES.TemplateResponse(request, "_issue_report.html", {"task_id": task_id, "generation_attempt_id": 0})


@router.post("/api/task/{task_id}/report-issue", response_class=HTMLResponse)
def api_report_issue(
    request: Request,
    task_id: int,
    reason: str = Form(...),
    description: str = Form(""),
    corrected_answer: str = Form(""),
    generation_attempt_id: int = Form(0),
):
    """Store a user-reported issue about a generation attempt."""
    from ...models import IssueReport

    logger.info("task report-issue task_id=%d reason=%s corrected=%s", task_id, reason, corrected_answer or "none")
    session = get_session()
    try:
        task = session.get(Task, task_id) if task_id else None
        report = IssueReport(
            generation_attempt_id=generation_attempt_id if generation_attempt_id else None,
            task_id=task_id if task_id else None,
            figure_id=task.figure_id if task else None,
            reason=reason,
            description=description[:500],
            corrected_answer=corrected_answer.strip()[:100],
            reported_by="user",
        )
        session.add(report)
        session.commit()

        return HTMLResponse("""
        <div class="bg-green-50 border border-green-200 rounded-xl p-4 mt-4">
            <div class="flex items-center gap-2 text-sm">
                <span>✅</span>
                <span class="font-semibold text-green-800">Issue reported</span>
                <span class="text-xs text-green-600 ml-auto">Your feedback helps improve future generations</span>
            </div>
        </div>
        """)
    finally:
        session.close()


@router.get("/api/task/{task_id}/quality-trend", response_class=HTMLResponse)
def api_quality_trend(request: Request, task_id: int):
    """Return quality scores across generation attempts for sparkline display."""
    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return HTMLResponse("")

        rows = list(
            session.exec(
                select(GenerationAttempt)
                .where(GenerationAttempt.figure_id == task.figure_id)
                .order_by(GenerationAttempt.created_at.asc())
            ).all()
        )

        scores = [r.validation_quality for r in rows if r.validation_quality > 0]
        return TEMPLATES.TemplateResponse(request, "_quality_trend.html", {"scores": scores})
    finally:
        session.close()
