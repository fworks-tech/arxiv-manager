"""Task CRUD, validation, regeneration, and history route handlers."""

import json as _json
import logging

from fastapi import Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from ...authoring import create_task, log_task_event, update_task
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


@router.get("/api/task/{task_id}/task-history", response_class=HTMLResponse)
def api_task_history(request: Request, task_id: int):
    """Return unified task history (generation attempts + task events) as HTML partial."""
    from ...models import TaskEvent

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return HTMLResponse("Task not found", status_code=404)

        # Query generation attempts
        attempts = list(
            session.exec(
                select(GenerationAttempt)
                .where(GenerationAttempt.task_id == task_id)
                .order_by(GenerationAttempt.created_at.desc())
                .limit(50)
            ).all()
        )

        # Query task events
        events = list(
            session.exec(
                select(TaskEvent)
                .where(TaskEvent.task_id == task_id)
                .order_by(TaskEvent.created_at.desc())
                .limit(50)
            ).all()
        )

        # Merge as unified event list sorted by created_at DESC
        merged: list[dict] = []

        for r in attempts:
            errors_list = (
                _json.loads(r.validation_errors)
                if r.validation_errors and r.validation_errors.strip() not in ("", "[]")
                else []
            )
            merged.append(
                {
                    "event_type": "regeneration",
                    "generation_type": r.generation_type,
                    "difficulty": r.difficulty or "",
                    "model_name": r.model_name or "",
                    "generated_question": r.generated_question or "",
                    "generated_answer": r.generated_answer or "",
                    "generated_answer_format": r.generated_answer_format or "",
                    "quality": r.validation_quality,
                    "errors": errors_list,
                    "total_tokens": r.total_tokens or 0,
                    "input_tokens": r.input_tokens or 0,
                    "output_tokens": r.output_tokens or 0,
                    "cost_str": "",
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
                }
            )

        for e in events:
            try:
                details = _json.loads(e.details) if e.details else {}
            except (_json.JSONDecodeError, TypeError):
                details = {}
            ts = e.created_at.strftime("%Y-%m-%d %H:%M") if e.created_at else ""
            event_dict: dict = {
                "event_type": e.event_type,
                "created_at": ts,
                "quality": e.quality_score or 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_str": "",
                "generation_type": "",
                "difficulty": "",
                "model_name": "",
                "generated_question": "",
                "generated_answer": "",
                "generated_answer_format": "",
                "errors": [],
                "changed_fields": [],
                "old_values": {},
                "new_values": {},
                "reason": "",
                "description": "",
                "corrected_answer": "",
                "old_question": "",
                "new_question": "",
                "old_answer": "",
                "new_answer": "",
                "old_difficulty": "",
                "new_difficulty": "",
                "qwen_passes": 0,
                "gemini_passes": 0,
                "rhea_passed": False,
                "rhea_notes": "",
            }
            event_dict.update(details)
            merged.append(event_dict)

        merged.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return TEMPLATES.TemplateResponse(request, "_task_history.html", {"events": merged})
    finally:
        session.close()


def _do_regenerate(
    image_path, api_key, difficulty, figure_type, complexity, prev_question, figure_id,
    validation_context="", task_id=None,
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
                task_id=task_id,
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
                    task_id=task_id,
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
                task_id=task_id,
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
        validation_quality=draft.get("_validation_quality", 0.0),
        validation_is_valid=draft.get("_validation_is_valid", False),
        validation_errors=_json.dumps(draft.get("_validation_errors", [])),
        validation_warnings=_json.dumps(draft.get("_validation_warnings", [])),
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
            img_path, api_key, difficulty, figure_type, complexity,
            prev_question, figure_id, validation_context, task_id=task_id,
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
            img_path, api_key, difficulty, figure_type, complexity,
            prev_question, task.figure_id, validation_context, task_id=task.id,
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

        # Validate draft before saving (safety net for self-critique rewrites)
        final_v = validate_task(
            draft["question"],
            draft["answer"],
            draft.get("answer_format", "number"),
            figure_type=figure_type,
            task_type=draft.get("task_type", "chart"),
        )
        if final_v.errors:
            err_msg = "Validation errors: " + "; ".join(final_v.errors[:3])
            logger.warning("regenerate validation failed task_id=%d: %s", task_id, err_msg)
            return {"error": err_msg, "ok": False}

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
    from ...db import get_session as _get_session

    logger.info("task apply-fix task_id=%d", task_id)
    s = _get_session()
    try:
        task_before = s.get(Task, task_id)
        old_q = task_before.question if task_before else ""
        old_a = task_before.answer if task_before else ""
    finally:
        s.close()

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
    log_task_event(
        task_id,
        "ai_fix",
        {"old_question": old_q, "old_answer": old_a, "new_question": question, "new_answer": answer},
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

        log_task_event(
            task_id,
            "issue_report",
            {"reason": reason, "description": description, "corrected_answer": corrected_answer},
        )

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


@router.post("/api/task/{task_id}/delete")
def api_delete_task(task_id: int):
    """Delete a task and redirect to tasks list."""
    from ...authoring import delete_task as _delete_task

    logger.info("task delete task_id=%d", task_id)
    log_task_event(task_id, "delete", {"reason": "user_requested"})
    if not _delete_task(task_id):
        return HTMLResponse("Task not found", status_code=404)
    return RedirectResponse(url="/tasks", status_code=303)


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


@router.post("/api/task/{task_id}/check-answer", response_class=HTMLResponse)
def api_check_answer(request: Request, task_id: int):
    """Send task image + question to minimax-m3, then verify answer against golden answer."""
    import base64 as _b64
    import io as _io
    import json as _json
    import os as _os

    from PIL import Image as PILImage

    from ...authoring._draft_config import CONFIG
    from ...authoring._draft_prompts import CHECK_ANSWER_PROMPT, VERIFY_ANSWER_PROMPT
    from ...authoring.ai_draft._api_client import _call_opencode as _call

    api_key = _os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return TEMPLATES.TemplateResponse(
            request, "_check_answer.html", {"error": "No OPENCODE_API_KEY set"}
        )

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return TEMPLATES.TemplateResponse(
                request, "_check_answer.html", {"error": "Task not found"}
            )

        # Build prompt and load image
        check_prompt = CHECK_ANSWER_PROMPT.text.format(question=task.question)

        img_path = STORAGE_DIR / task.image_path
        b64_image = ""
        if not img_path.exists():
            return TEMPLATES.TemplateResponse(
                request, "_check_answer.html", {"error": "Image not found"}
            )

        with PILImage.open(img_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail(CONFIG.thumbnail_size)
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=CONFIG.jpeg_quality, optimize=True)
        b64_image = _b64.b64encode(buf.getvalue()).decode()

        # Step 1: ask minimax-m3 to answer
        def _parse_answer(content: str, raw_text: str = "") -> dict | None:
            text = content.strip()
            if text.startswith("```"):
                import re as _re
                text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = _re.sub(r"\n?```\s*$", "", text)
            try:
                data = _json.loads(text)
                if "answer" in data and data["answer"]:
                    return data
            except _json.JSONDecodeError:
                pass
            start = text.find("{")
            if start >= 0:
                depth = 0
                for end in range(start, len(text)):
                    if text[end] == "{":
                        depth += 1
                    elif text[end] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                data = _json.loads(text[start:end+1])
                                if "answer" in data and data["answer"]:
                                    return data
                            except _json.JSONDecodeError:
                                pass
                            break
            return None

        vlm_result = _call(
            api_key, check_prompt, b64_image,
            model=CONFIG.default_model,
            retries=1,
            difficulty=task.difficulty or "challenging",
            parser=_parse_answer,
        )

        if not vlm_result or not vlm_result.get("answer"):
            return TEMPLATES.TemplateResponse(
                request, "_check_answer.html", {"error": "VLM did not return a valid answer"}
            )

        vlm_answer = vlm_result["answer"].strip()
        vlm_reasoning = vlm_result.get("reasoning", "").strip()
        input_tokens = vlm_result.get("_usage", {}).get("input_tokens", 0)
        output_tokens = vlm_result.get("_usage", {}).get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        # Step 2: verify with deepseek-v4-flash (text-only)
        verify_prompt = VERIFY_ANSWER_PROMPT.text.format(
            question=task.question,
            golden_answer=task.answer,
            vlm_answer=vlm_answer,
            vlm_reasoning=vlm_reasoning,
        )

        def _parse_verification(content: str, raw_text: str = "") -> dict | None:
            text = content.strip()
            if text.startswith("```"):
                import re as _re
                text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = _re.sub(r"\n?```\s*$", "", text)
            try:
                data = _json.loads(text)
                if "match" in data:
                    return data
            except _json.JSONDecodeError:
                pass
            start = text.find("{")
            if start >= 0:
                depth = 0
                for end in range(start, len(text)):
                    if text[end] == "{":
                        depth += 1
                    elif text[end] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                data = _json.loads(text[start:end+1])
                                if "match" in data:
                                    return data
                            except _json.JSONDecodeError:
                                pass
                            break
            return None

        verify_result = _call(
            api_key, verify_prompt, "",
            model=CONFIG.text_model,
            retries=1,
            difficulty=task.difficulty or "challenging",
            parser=_parse_verification,
        )

        match = False
        explanation = ""
        analysis = ""
        if verify_result:
            match = bool(verify_result.get("match", False))
            explanation = verify_result.get("explanation", "")
            analysis = verify_result.get("analysis", "")

        # Log to TaskEvent
        log_task_event(
            task_id,
            "check_answer",
            {
                "model": CONFIG.default_model,
                "verifier": CONFIG.text_model,
                "golden_answer": task.answer,
                "vlm_answer": vlm_answer,
                "vlm_reasoning": vlm_reasoning,
                "match": match,
                "explanation": explanation,
                "analysis": analysis,
                "tokens": total_tokens,
            },
            quality_score=1.0 if match else 0.0,
        )

        return TEMPLATES.TemplateResponse(
            request,
            "_check_answer.html",
            {
                "golden_answer": task.answer,
                "vlm_answer": vlm_answer,
                "vlm_reasoning": vlm_reasoning,
                "match": match,
                "explanation": explanation,
                "analysis": analysis,
                "tokens": total_tokens,
            },
        )
    finally:
        session.close()
