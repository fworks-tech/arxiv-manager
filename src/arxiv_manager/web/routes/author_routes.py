"""Image upload, draft, and propose route handlers."""
import hashlib
import io
import json
import logging
import os
from pathlib import Path

from fastapi import File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image as PILImage

from ...db import get_session
from ...models import Figure, ImageStatus
from ...authoring.image_analyzer import analyze_uploaded_image, validate_draft
from ...authoring.ai_draft import draft_qa, draft_with_self_critique
from ...authoring._draft_telemetry import log_generation_attempt
from ...sourcing.filters import compute_file_hash, audit_figure
from ...storage import UPLOADS_DIR, STORAGE_DIR
from . import TEMPLATES, router, _upload_cache

logger = logging.getLogger(__name__)


def _save_upload(file_bytes: bytes, filename: str = "upload") -> tuple[str, str]:
    """Save an uploaded image, return (upload_id, path)."""
    upload_id = "upload_" + hashlib.sha256(file_bytes).hexdigest()[:16]
    ext = Path(filename).suffix if filename else ".png"
    dest = UPLOADS_DIR / f"{upload_id}.jpg"
    try:
        img = PILImage.open(io.BytesIO(file_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(str(dest), "JPEG", quality=92)
    except Exception:
        raw_ext = ext.lower()
        if raw_ext in (".png", ".jpg", ".jpeg", ".webp"):
            dest = UPLOADS_DIR / f"{upload_id}{raw_ext}"
            dest.write_bytes(file_bytes)
        else:
            raise
    return upload_id, str(dest)


@router.post("/api/image/upload", response_class=HTMLResponse)
def api_upload_image(
    request: Request,
    image: UploadFile = File(None),
    arxiv_figure_path: str = Form(""),
):
    """Upload an image or select from arXiv extraction."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = ""
    result = None
    error = ""

    try:
        if arxiv_figure_path:
            logger.info("upload via arxiv_figure_path=%s", arxiv_figure_path)
            src = STORAGE_DIR / arxiv_figure_path
            if src.exists():
                data = src.read_bytes()
                upload_id, dest = _save_upload(data, src.name)
                result = analyze_uploaded_image(dest)
            else:
                error = "Figure file not found"
        elif image and image.filename:
            data = image.file.read()
            logger.info("upload via browser filename=%s size=%d", image.filename, len(data))
            if len(data) > 20 * 1024 * 1024:
                error = "File too large (max 20MB)"
            else:
                upload_id, dest = _save_upload(data, image.filename)
                result = analyze_uploaded_image(dest)
        else:
            error = "No image provided"

        if error:
            return TEMPLATES.TemplateResponse(
                request, "_author_analysis.html", {"result": None, "error": error, "upload_id": ""}
            )

        _upload_cache[upload_id] = result

        return TEMPLATES.TemplateResponse(
            request, "_author_analysis.html",
            {"result": result, "upload_id": upload_id, "error": ""},
        )
    except Exception as e:
        logger.error("upload exception: %s", e, exc_info=True)
        return TEMPLATES.TemplateResponse(
            request, "_author_analysis.html",
            {"result": None, "error": str(e)[:150], "upload_id": ""},
        )


@router.post("/api/image/draft", response_class=HTMLResponse)
def api_draft_qa(
    request: Request,
    upload_id: str = Form(...),
    difficulty: str = Form("challenging"),
    previous_question: str = Form(""),
):
    """Generate a Q&A draft for the uploaded image."""

    logger.info("draft request upload_id=%s difficulty=%s", upload_id, difficulty)

    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": "No OPENCODE_API_KEY set",
             "upload_id": upload_id, "difficulty": difficulty},
        )

    analysis = _upload_cache.get(upload_id)
    if not analysis:
        logger.info("draft: cache miss, re-analyzing from disk")
        img_path = None
        for ext in [".jpg", ".png", ".webp", ".jpeg"]:
            p = UPLOADS_DIR / f"{upload_id}{ext}"
            if p.exists():
                img_path = p
                break
        if not img_path:
            return TEMPLATES.TemplateResponse(
                request, "_author_draft.html",
                {"draft": None, "validation": None, "error": "Upload not found — please re-upload",
                 "upload_id": upload_id, "difficulty": difficulty},
            )
        analysis = analyze_uploaded_image(img_path)
        _upload_cache[upload_id] = analysis

    img_path = UPLOADS_DIR / f"{upload_id}.jpg"
    if not img_path.exists():
        for ext in (".png", ".webp", ".jpeg"):
            p = UPLOADS_DIR / f"{upload_id}{ext}"
            if p.exists():
                img_path = p
                break
    figure_type = analysis["audit"].get("figure_type", "")
    complexity = analysis["audit"].get("complexity_score", 0.0)
    suitability = analysis.get("suitability", "")

    try:
        if difficulty in ("challenging", "hardest"):
            draft = draft_with_self_critique(
                image_path=img_path, max_rounds=1,
                api_key=api_key, difficulty=difficulty,
                figure_type=figure_type, complexity_score=complexity,
                previous_question=previous_question,
            )
            if draft is None:
                draft = draft_qa(
                    image_path=img_path,
                    api_key=api_key, difficulty=difficulty,
                    figure_type=figure_type, complexity_score=complexity,
                    previous_question=previous_question,
                )
        else:
            draft = draft_qa(
                image_path=img_path,
                api_key=api_key, difficulty=difficulty,
                figure_type=figure_type, complexity_score=complexity,
                previous_question=previous_question,
            )
    except ValueError as e:
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": str(e),
             "upload_id": upload_id, "difficulty": difficulty},
        )

    if not draft:
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": "Draft generation failed — API error",
             "upload_id": upload_id, "difficulty": difficulty},
        )

    validation = validate_draft(draft, figure_type=figure_type)
    logger.info("draft ok upload_id=%s quality=%.2f errors=%d",
                upload_id, validation.get("quality_score", 0), len(validation.get("errors", [])))

    usage = draft.get("_usage", {})
    log_generation_attempt(
        attempt_number=1,
        generation_type="self_critique" if difficulty in ("challenging", "hardest") else "draft",
        source_route="api_draft_qa",
        prompt_template_name=f"{difficulty}_{figure_type}" if figure_type else difficulty,
        prompt_version_id=draft.get("_prompt_version_id", ""),
        prompt_text_hash=draft.get("_prompt_text_hash", ""),
        difficulty=difficulty, figure_type=figure_type,
        complexity_score=complexity, previous_question=previous_question,
        raw_response=draft.get("_raw_response", ""),
        reasoning_trace=draft.get("_reasoning_trace", ""),
        generated_question=draft.get("question", ""),
        generated_answer=draft.get("answer", ""),
        generated_answer_format=draft.get("answer_format", ""),
        generated_task_type=draft.get("task_type", ""),
        validation_quality=validation.get("quality_score", 0),
        validation_is_valid=validation.get("is_valid", False),
        validation_errors=json.dumps(validation.get("errors", [])),
        validation_warnings=json.dumps(validation.get("warnings", [])),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        success=True, elapsed_ms=0,
    )

    return TEMPLATES.TemplateResponse(
        request, "_author_draft.html",
        {"draft": draft, "validation": validation, "error": "",
         "upload_id": upload_id, "difficulty": difficulty},
    )


@router.post("/api/image/discard", response_class=HTMLResponse)
def api_discard_image(
    request: Request,
    upload_id: str = Form(...),
):
    """Delete an uploaded image and clear cache."""
    logger.info("discard upload_id=%s", upload_id)
    for ext in [".jpg", ".png"]:
        p = UPLOADS_DIR / f"{upload_id}{ext}"
        if p.exists():
            p.unlink()
    _upload_cache.pop(upload_id, None)
    return HTMLResponse("")


@router.post("/api/image/propose")
def api_propose_task(
    upload_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    answer_format: str = Form("word"),
    task_type: str = Form("chart"),
    domain: str = Form("Computer Science"),
    title: str = Form(""),
):
    """Save the uploaded image as a Figure + Task in the database."""
    from ...authoring import create_task

    logger.info("propose upload_id=%s type=%s format=%s", upload_id, task_type, answer_format)

    session = get_session()
    try:
        img_path = UPLOADS_DIR / f"{upload_id}.jpg"
        if not img_path.exists():
            for ext in (".png", ".webp", ".jpeg"):
                p = UPLOADS_DIR / f"{upload_id}{ext}"
                if p.exists():
                    img_path = p
                    break
        if not img_path.exists():
            return HTMLResponse("Upload not found", status_code=404)

        img_hash = compute_file_hash(img_path)
        img = PILImage.open(img_path)
        audit = audit_figure(img_path)

        fig_path = f"figures/user_{upload_id}.jpg"
        fig_dest = STORAGE_DIR / fig_path
        import shutil
        shutil.copy2(str(img_path), str(fig_dest))

        figure = Figure(
            paper_id="user_upload",
            image_path=fig_path,
            image_hash=img_hash,
            caption="", page_num=0, figure_num="",
            width=audit["width"], height=audit["height"],
            width_height_ratio=audit["width_height_ratio"],
            filesize_bytes=audit["filesize_bytes"],
            complexity_score=audit["complexity_score"],
            figure_type=audit["figure_type"],
            is_dense=audit["is_dense"],
            is_suitable=audit["is_suitable"],
            status=ImageStatus.USED.value,
        )
        session.add(figure)
        session.commit()
        session.refresh(figure)

        task = create_task(
            figure_id=figure.id,
            title=title or f"User upload — {upload_id[:12]}",
            domain=domain, question=question, answer=answer,
            answer_format=answer_format, task_type=task_type,
            ai_generated=True,
        )
        return RedirectResponse(url=f"/task/{task.id}", status_code=303)
    finally:
        session.close()
