"""Web routes (HTML pages + HTMX endpoints)."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time as time_module
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage
from sqlmodel import select

from ..db import get_session
from ..models import Figure, Paper, Task, ImageStatus, TaskStatus
from ..authoring.validator import validate_task
from ..authoring.image_analyzer import analyze_uploaded_image, validate_draft
from ..authoring.ai_draft import draft_qa, draft_with_self_critique
from ..sourcing.arxiv import search_papers
from ..sourcing.downloader import download_pdf
from ..sourcing.extractor import extract_figures
from ..sourcing.filters import compute_file_hash, audit_figure
from ..storage import STORAGE_DIR, UPLOADS_DIR
from ..tracking import get_stats

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
router = APIRouter()

# Cache for uploaded image analysis (keyed by upload_id)
_upload_cache: dict[str, dict] = {}
_upload_cache_ts: dict[str, float] = {}


# ─── PAGES ────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Dashboard home."""
    stats = get_stats()
    return TEMPLATES.TemplateResponse(request, "base.html", {"stats": stats})


@router.get("/images", response_class=HTMLResponse)
async def images_page(
    request: Request,
    status: str = "",
    min_complexity: float = 0,
    figure_type: str = "",
    suitable_only: bool = False,
):
    """Image library page."""
    session = get_session()
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


@router.get("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request, status: str = ""):
    """Tasks list page."""
    session = get_session()
    query = select(Task)
    if status:
        query = query.where(Task.status == status)
    query = query.order_by(Task.created_at.desc())
    tasks = list(session.exec(query).all())

    return TEMPLATES.TemplateResponse(request, "tasks.html", {
        "tasks": tasks,
        "status_filter": status,
    })


@router.get("/task/new/{figure_id}", response_class=HTMLResponse)
async def task_form(request: Request, figure_id: int):
    """Task authoring form for a specific image."""
    session = get_session()
    figure = session.get(Figure, figure_id)
    if not figure:
        return HTMLResponse("Image not found", status_code=404)

    return TEMPLATES.TemplateResponse(request, "task_form.html", {
        "figure": figure,
        "validation": None,
        "task": None,
    })


@router.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: int):
    """View/edit an existing task."""
    session = get_session()
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


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Statistics dashboard."""
    logger.info("stats page")
    stats = get_stats()
    return TEMPLATES.TemplateResponse(request, "stats.html", {"stats": stats})


# ─── UPLOAD / AUTHOR PAGE (Tier 4) ────────────────────────────────


@router.get("/author", response_class=HTMLResponse)
async def author_page(request: Request):
    """Main upload + Q&A authoring page."""
    return TEMPLATES.TemplateResponse(request, "author.html", {})


def _save_upload(file_bytes: bytes, filename: str = "upload") -> tuple[str, Path]:
    """Save an uploaded image to _uploads/, return (upload_id, path)."""
    upload_id = "upload_" + hashlib.sha256(file_bytes).hexdigest()[:16]
    ext = Path(filename).suffix or ".png"
    dest = UPLOADS_DIR / f"{upload_id}.jpg"
    # Convert to JPEG for consistency
    try:
        img = PILImage.open(io.BytesIO(file_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(str(dest), "JPEG", quality=92)
    except Exception:
        # Fallback: save raw
        dest = UPLOADS_DIR / f"{upload_id}{ext}"
        dest.write_bytes(file_bytes)
    return upload_id, dest


@router.post("/api/image/upload", response_class=HTMLResponse)
async def api_upload_image(
    request: Request,
    image: UploadFile = File(None),
    arxiv_figure_path: str = Form(""),
):
    """Upload an image or select from arXiv extraction.

    Runs audit_figure() and returns suitability analysis.
    """
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
                logger.info("upload from arxiv ok upload_id=%s size=%d result=%s", upload_id, len(data), result.get("suitability"))
            else:
                error = "Figure file not found"
                logger.warning("arxiv figure not found: %s", src)
        elif image and image.filename:
            data = await image.read()
            logger.info("upload via browser filename=%s size=%d", image.filename, len(data))
            if len(data) > 20 * 1024 * 1024:
                error = "File too large (max 20MB)"
                logger.warning("upload too large: %d bytes", len(data))
            else:
                upload_id, dest = _save_upload(data, image.filename)
                result = analyze_uploaded_image(dest)
                logger.info("upload ok upload_id=%s suitability=%s", upload_id, result.get("suitability"))
        else:
            error = "No image provided"
            logger.warning("upload called without image or arxiv_figure_path")

        if error:
            logger.warning("upload error: %s", error)
            return TEMPLATES.TemplateResponse(
                request, "_author_analysis.html", {"result": None, "error": error, "upload_id": ""}
            )

        _upload_cache[upload_id] = result
        _upload_cache_ts[upload_id] = time_module.time()

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
async def api_draft_qa(
    request: Request,
    upload_id: str = Form(...),
    difficulty: str = Form("challenging"),
    previous_question: str = Form(""),
):
    """Generate a Q&A draft for the uploaded image."""
    import os as os_mod

    logger.info("draft request upload_id=%s difficulty=%s", upload_id, difficulty)

    api_key = os_mod.environ.get("OPENCODE_API_KEY")
    if not api_key:
        logger.warning("draft failed: no OPENCODE_API_KEY set")
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": "No OPENCODE_API_KEY set", "upload_id": upload_id, "difficulty": difficulty},
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
            logger.warning("draft failed: upload not found for upload_id=%s", upload_id)
            return TEMPLATES.TemplateResponse(
                request, "_author_draft.html",
                {"draft": None, "validation": None, "error": "Upload not found — please re-upload", "upload_id": upload_id, "difficulty": difficulty},
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
    logger.info("draft difficulty=%s suitability=%s figure_type=%s complexity=%.3f", difficulty, suitability, figure_type, complexity)

    if difficulty in ("challenging", "hardest"):
        logger.info("draft using self_critique flow difficulty=%s", difficulty)
        draft = draft_with_self_critique(
            image_path=img_path,
            max_rounds=1,
            provider="opencode",
            api_key=api_key,
            difficulty=difficulty,
            figure_type=figure_type,
            complexity_score=complexity,
            previous_question=previous_question,
        )
        if not draft:
            logger.info("self_critique returned None, falling back to plain draft_qa")
            draft = draft_qa(
                image_path=img_path,
                provider="opencode",
                api_key=api_key,
                difficulty=difficulty,
                figure_type=figure_type,
                complexity_score=complexity,
                previous_question=previous_question,
            )
    else:
        draft = draft_qa(
            image_path=img_path,
            provider="opencode",
            api_key=api_key,
            difficulty=difficulty,
            figure_type=figure_type,
            complexity_score=complexity,
            previous_question=previous_question,
        )

    if not draft:
        logger.error("draft generation returned None for upload_id=%s", upload_id)
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": "Draft generation failed — API error", "upload_id": upload_id, "difficulty": difficulty},
        )

    validation = validate_draft(draft, figure_type=figure_type)
    logger.info("draft ok upload_id=%s quality=%.2f errors=%d", upload_id, validation.get("quality_score", 0), len(validation.get("errors", [])))

    return TEMPLATES.TemplateResponse(
        request, "_author_draft.html",
        {"draft": draft, "validation": validation, "error": "", "upload_id": upload_id, "difficulty": difficulty},
    )


@router.post("/api/image/discard", response_class=HTMLResponse)
async def api_discard_image(
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
    _upload_cache_ts.pop(upload_id, None)
    return HTMLResponse("")


@router.post("/api/image/propose")
async def api_propose_task(
    upload_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    answer_format: str = Form("word"),
    task_type: str = Form("chart"),
    domain: str = Form("Computer Science"),
    title: str = Form(""),
):
    """Save the uploaded image as a Figure + Task in the database."""
    from ..authoring import create_task

    logger.info("propose upload_id=%s type=%s format=%s", upload_id, task_type, answer_format)

    session = get_session()

    img_path = UPLOADS_DIR / f"{upload_id}.jpg"
    if not img_path.exists():
        for ext in (".png", ".webp", ".jpeg"):
            p = UPLOADS_DIR / f"{upload_id}{ext}"
            if p.exists():
                img_path = p
                break
    if not img_path.exists():
        logger.warning("propose failed: upload not found upload_id=%s", upload_id)
        return HTMLResponse("Upload not found", status_code=404)

    img_hash = compute_file_hash(img_path)
    img = PILImage.open(img_path)
    w, h = img.size
    audit = audit_figure(img_path)

    fig_path = f"figures/user_{upload_id}.jpg"
    fig_dest = STORAGE_DIR / fig_path
    import shutil
    shutil.copy2(str(img_path), str(fig_dest))

    figure = Figure(
        paper_id="user_upload",
        image_path=fig_path,
        image_hash=img_hash,
        caption="",
        page_num=0,
        figure_num="",
        width=audit["width"],
        height=audit["height"],
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
    logger.info("propose figure created id=%d", figure.id)

    # Create task
    task = create_task(
        figure_id=figure.id,
        title=title or f"User upload — {upload_id[:12]}",
        domain=domain,
        question=question,
        answer=answer,
        answer_format=answer_format,
        task_type=task_type,
        ai_generated=True,
    )
    return RedirectResponse(url=f"/task/{task.id}", status_code=303)


@router.get("/api/arxiv/search", response_class=HTMLResponse)
async def api_arxiv_search(
    request: Request,
    q: str = Query(""),
    domain: str = Query(""),
    limit: int = Query(10),
):
    """Search arXiv CC0 papers."""
    logger.info("arxiv search q=%s domain=%s limit=%d", q, domain, limit)
    if not q:
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": None, "error": "Enter search terms"}
        )
    try:
        term_list = [t.strip() for t in q.split(",") if t.strip()] if q else None
        papers = search_papers(terms=term_list, domain=domain or None, limit=limit)
        logger.info("arxiv search found %d papers", len(papers) if papers else 0)
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": papers, "error": ""}
        )
    except Exception as e:
        logger.error("arxiv search error: %s", e, exc_info=True)
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": None, "error": str(e)[:200]}
        )


@router.post("/api/arxiv/extract", response_class=HTMLResponse)
async def api_arxiv_extract(
    request: Request,
    arxiv_id: str = Form(...),
):
    """Download a paper PDF, extract figures, and return top candidates."""
    logger.info("arxiv extract arxiv_id=%s", arxiv_id)
    try:
        import sqlite3
        conn = sqlite3.connect(str(STORAGE_DIR / "arxiv-manager.db"))
        c = conn.cursor()

        pdf_path = download_pdf(arxiv_id)
        extracted = extract_figures(pdf_path)
        logger.info("arxiv extract extracted %d raw figures", len(extracted))

        figures = []
        for img_data in extracted:
            full_path = STORAGE_DIR / img_data["image_path"]
            if not full_path.exists():
                continue
            audit = audit_figure(full_path)
            if not audit["is_suitable"]:
                continue
            img_hash = compute_file_hash(full_path)
            figures.append({
                "image_path": img_data["image_path"],
                "image_hash": img_hash,
                "width": audit["width"],
                "height": audit["height"],
                "complexity_score": audit["complexity_score"],
                "figure_type": audit["figure_type"],
                "is_dense": audit["is_dense"],
                "page_num": img_data.get("page_num", 0),
            })

        figures.sort(key=lambda f: f["complexity_score"], reverse=True)
        figures = figures[:3]
        logger.info("arxiv extract %d suitable figures (top 3)", len(figures))

        conn.close()
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_figures.html", {"figures": figures, "error": ""}
        )
    except Exception as e:
        logger.error("arxiv extract error: %s", e, exc_info=True)
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_figures.html", {"figures": None, "error": str(e)[:200]}
        )


# ─── METRICS DASHBOARD (Tier 2b) ──────────────────────────────────

_metrics_cache: dict | None = None
_metrics_cache_ts: float = 0


def _compute_metrics() -> dict:
    """Read telemetry JSONL and return aggregated metrics.

    Cached for 60s to avoid re-reading on every page refresh.
    """
    global _metrics_cache, _metrics_cache_ts

    now = time_module.time()
    if _metrics_cache is not None and now - _metrics_cache_ts < 60:
        return _metrics_cache

    telemetry_path = STORAGE_DIR / "_draft_telemetry.jsonl"
    records: list[dict] = []
    if telemetry_path.exists():
        with open(str(telemetry_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total = len(records)
    ok = sum(1 for r in records if r.get("ok"))
    verify_count = sum(1 for r in records if r.get("difficulty") == "verify")
    draft_count = total - verify_count

    # Latency stats (drafts only, skip verify)
    draft_latencies = [r["elapsed_s"] for r in records if r.get("difficulty") != "verify" and r.get("elapsed_s")]
    valid_drafts = [r for r in records if r.get("difficulty") != "verify"]

    avg_lat = round(sum(draft_latencies) / len(draft_latencies), 1) if draft_latencies else 0
    min_lat = round(min(draft_latencies), 1) if draft_latencies else 0
    max_lat = round(max(draft_latencies), 1) if draft_latencies else 0
    sorted_lat = sorted(draft_latencies)
    p50_lat = round(sorted_lat[len(sorted_lat) // 2], 1) if sorted_lat else 0

    # By difficulty
    by_diff: dict[str, dict] = {}
    for r in valid_drafts:
        diff = r.get("difficulty") or "unknown"
        if diff == "challenging" or diff == "hardest" or diff == "":
            label = diff or "manual"
        else:
            label = diff
        bucket = by_diff.setdefault(label, {"total": 0, "ok": 0})
        bucket["total"] += 1
        if r.get("ok"):
            bucket["ok"] += 1

    # By figure_type
    by_type: dict[str, dict] = {}
    for r in valid_drafts:
        ft = r.get("figure_type") or "unknown"
        bucket = by_type.setdefault(ft, {"total": 0, "ok": 0})
        bucket["total"] += 1
        if r.get("ok"):
            bucket["ok"] += 1

    # Recent timeline: buckets per hour (last 24 items)
    recent = records[-24:] if len(records) > 24 else records

    metrics = {
        "total_drafts": draft_count,
        "total_verify": verify_count,
        "success_rate": round(100 * ok / max(draft_count, 1), 1),
        "avg_latency": avg_lat,
        "min_latency": min_lat,
        "max_latency": max_lat,
        "p50_latency": p50_lat,
        "by_difficulty": by_diff,
        "by_figure_type": by_type,
        "recent": [dict(r, **{"es": r.get("elapsed_s", 0), "ok_bool": r.get("ok")}) for r in recent[-24:]],
    }

    _metrics_cache = metrics
    _metrics_cache_ts = now
    return metrics


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    """AI draft performance dashboard."""
    logger.info("metrics page")
    metrics = _compute_metrics()
    return TEMPLATES.TemplateResponse(request, "metrics.html", {"m": metrics})


# ─── HTMX API ENDPOINTS ─────────────────────────────────────────

@router.post("/api/task/create", response_class=HTMLResponse)
async def api_create_task(
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
    from ..authoring import create_task

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
    figure = get_session().get(Figure, figure_id)
    return TEMPLATES.TemplateResponse(request, "task_form.html", {
        "figure": figure,
        "task": None,
        "validation": validation,
        "form_data": {"title": title, "domain": domain, "question": question, "answer": answer, "answer_format": answer_format, "task_type": task_type},
    })


@router.post("/api/task/{task_id}/update", response_class=HTMLResponse)
async def api_update_task(
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
    from ..authoring import update_task

    logger.info("task update task_id=%d type=%s format=%s", task_id, task_type, answer_format)
    validation = validate_task(question, answer, answer_format)
    task = update_task(task_id, title=title, domain=domain, question=question, answer=answer, answer_format=answer_format, task_type=task_type)
    figure = get_session().get(Figure, task.figure_id) if task else None
    logger.info("task updated id=%d valid=%s", task_id, validation.is_valid)

    return TEMPLATES.TemplateResponse(request, "task_form.html", {
        "figure": figure,
        "task": task,
        "validation": validation,
    })


@router.post("/api/task/{task_id}/validate", response_class=HTMLResponse)
async def api_validate_task(request: Request, task_id: int):
    """Re-validate a task (HTMX endpoint)."""
    logger.info("task revalidate task_id=%d", task_id)
    session = get_session()
    task = session.get(Task, task_id)
    if not task:
        logger.warning("task revalidate not found task_id=%d", task_id)
        return HTMLResponse("Not found", status_code=404)

    figure = session.get(Figure, task.figure_id)
    figure_type = getattr(figure, "figure_type", "") if figure else ""
    validation = validate_task(task.question, task.answer, task.answer_format,
                               figure_type=figure_type, task_type=task.task_type)
    logger.info("task revalidate ok task_id=%d valid=%s score=%.1f", task_id, validation.is_valid, validation.quality_score)

    return TEMPLATES.TemplateResponse(request, "_validation.html", {
        "validation": validation,
    })


@router.post("/api/task/{task_id}/regenerate")
async def api_regenerate_task(request: Request, task_id: int, difficulty: str = Form("challenging")):
    """Regenerate Q&A for a task using AI draft."""
    logger.info("task regenerate task_id=%d difficulty=%s", task_id, difficulty)
    import os as os_mod
    api_key = os_mod.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return {"error": "No OPENCODE_API_KEY set", "ok": False}

    session = get_session()
    task = session.get(Task, task_id)
    if not task:
        return {"error": "Task not found", "ok": False}

    img_path = STORAGE_DIR / task.image_path
    if not img_path.exists():
        logger.warning("task regenerate: image not found at %s", img_path)
        return {"error": "Image not found", "ok": False}

    figure = session.get(Figure, task.figure_id) if task.figure_id else None
    figure_type = getattr(figure, "figure_type", "") if figure else ""
    complexity = getattr(figure, "complexity_score", 0.0) if figure else 0.0
    prev_question = task.question

    if difficulty in ("challenging", "hardest"):
        draft = draft_with_self_critique(
            image_path=img_path, max_rounds=1, provider="opencode",
            api_key=api_key, difficulty=difficulty,
            figure_type=figure_type, complexity_score=complexity,
            previous_question=prev_question,
        )
        if not draft:
            draft = draft_qa(
                image_path=img_path, provider="opencode",
                api_key=api_key, difficulty=difficulty,
                figure_type=figure_type, complexity_score=complexity,
                previous_question=prev_question,
            )
    else:
        draft = draft_qa(
            image_path=img_path, provider="opencode",
            api_key=api_key, difficulty=difficulty,
            figure_type=figure_type, complexity_score=complexity,
            previous_question=prev_question,
        )

    if not draft:
        return {"error": "Draft generation failed", "ok": False}

    # Dedup: if answer unchanged or question too similar, try harder
    same_answer = draft["answer"].strip().lower() == task.answer.strip().lower()
    if same_answer or draft["question"].strip().lower() == task.question.strip().lower():
        for _ in range(2):
            if difficulty in ("challenging", "hardest"):
                draft2 = draft_with_self_critique(
                    image_path=img_path, max_rounds=1, provider="opencode",
                    api_key=api_key, difficulty=difficulty,
                    figure_type=figure_type, complexity_score=complexity,
                    previous_question=prev_question,
                )
            else:
                draft2 = draft_qa(
                    image_path=img_path, provider="opencode",
                    api_key=api_key, difficulty=difficulty,
                    figure_type=figure_type, complexity_score=complexity,
                    previous_question=prev_question,
                )
            if draft2 and draft2["answer"].strip().lower() != task.answer.strip().lower():
                draft = draft2
                break

    task.question = draft["question"]
    task.answer = draft["answer"]
    task.answer_format = draft.get("answer_format", "number")
    task.task_type = draft.get("task_type", "chart")
    task.difficulty = difficulty
    session.add(task)
    session.commit()
    logger.info("task regenerate ok task_id=%d", task_id)

    return {
        "ok": True,
        "question": draft["question"],
        "answer": draft["answer"],
        "answer_format": draft.get("answer_format", "number"),
        "task_type": draft.get("task_type", "chart"),
    }


@router.post("/api/figure/{figure_id}/status")
async def update_figure_status(figure_id: int, status: str = Form(...)):
    """Update figure status (HTMX endpoint)."""
    logger.info("figure status figure_id=%d -> %s", figure_id, status)
    session = get_session()
    figure = session.get(Figure, figure_id)
    if figure:
        figure.status = status
        session.add(figure)
        session.commit()
        logger.info("figure status updated figure_id=%d status=%s", figure_id, status)
    return RedirectResponse(url="/images", status_code=303)


@router.post("/api/figures/bulk-reject")
async def bulk_reject_figures(figure_ids: list[int] = Form(default=[])):
    """Bulk reject multiple figures at once."""
    logger.info("bulk reject ids=%s", figure_ids)
    session = get_session()
    rejected = 0
    for fid in figure_ids:
        figure = session.get(Figure, fid)
        if figure:
            figure.status = "rejected"
            session.add(figure)
            rejected += 1
    session.commit()
    logger.info("bulk reject done count=%d", rejected)
    return RedirectResponse(url="/images", status_code=303)


@router.post("/api/task/{task_id}/difficulty")
async def update_task_difficulty(
    task_id: int,
    difficulty: str = Form(...),
    qwen: int = Form(0),
    gemini: int = Form(0),
):
    """Update task difficulty (HTMX endpoint)."""
    logger.info("task difficulty task_id=%d difficulty=%s qwen=%d gemini=%d", task_id, difficulty, qwen, gemini)
    from ..tracking import set_difficulty
    set_difficulty(task_id, difficulty, qwen, gemini)
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.post("/api/task/{task_id}/submit")
async def submit_task(task_id: int):
    """Mark task as submitted (HTMX endpoint)."""
    logger.info("task submit task_id=%d", task_id)
    from ..tracking import mark_submitted
    mark_submitted(task_id)
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/api/task/{task_id}/rhea")
async def update_rhea(
    request: Request,
    task_id: int,
    rhea_reviewed: bool = Form(False),
    rhea_passed: bool = Form(False),
    rhea_notes: str = Form(""),
):
    """Update Rhea review status (HTMX endpoint)."""
    logger.info("task rhea task_id=%d reviewed=%s passed=%s", task_id, rhea_reviewed, rhea_passed)
    session = get_session()
    task = session.get(Task, task_id)
    if task:
        task.rhea_reviewed = rhea_reviewed
        task.rhea_passed = rhea_passed
        task.rhea_notes = rhea_notes
        session.add(task)
        session.commit()
        logger.info("task rhea updated task_id=%d", task_id)
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.post("/api/task/{task_id}/rhea-override")
async def save_rhea_override(
    request: Request,
    task_id: int,
    rhea_override_notes: str = Form(...),
    rhea_passed: bool = Form(True),
):
    """Save author's override notes for a Rhea-rejected task.

    Sets rhea_passed=True and stores the author's justification.
    This is a first-class signal that the author disagreed with Rhea.
    """
    logger.info("task rhea override task_id=%d passed=%s notes_len=%d", task_id, rhea_passed, len(rhea_override_notes))
    session = get_session()
    task = session.get(Task, task_id)
    if task:
        task.rhea_override_notes = rhea_override_notes
        task.rhea_passed = rhea_passed
        session.add(task)
        session.commit()
        logger.info("task rhea override saved task_id=%d", task_id)
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)
