"""Web routes (HTML pages + HTMX endpoints)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time as time_module
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image as PILImage
from sqlmodel import select

from ..db import get_session
from ..models import Figure, Paper, Task, ImageStatus, TaskStatus
from ..authoring.validator import validate_task
from ..authoring.image_analyzer import analyze_uploaded_image, validate_draft
from ..authoring.ai_draft import draft_qa
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
            # Figure selected from arXiv extraction — copy into _uploads
            src = STORAGE_DIR / arxiv_figure_path
            if src.exists():
                data = src.read_bytes()
                upload_id, dest = _save_upload(data, src.name)
            else:
                error = "Figure file not found"
        elif image and image.filename:
            data = await image.read()
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

        # Cache analysis
        _upload_cache[upload_id] = result
        _upload_cache_ts[upload_id] = time_module.time()

        return TEMPLATES.TemplateResponse(
            request, "_author_analysis.html",
            {"result": result, "upload_id": upload_id, "error": ""},
        )
    except Exception as e:
        return TEMPLATES.TemplateResponse(
            request, "_author_analysis.html",
            {"result": None, "error": str(e)[:150], "upload_id": ""},
        )


@router.post("/api/image/draft", response_class=HTMLResponse)
async def api_draft_qa(
    request: Request,
    upload_id: str = Form(...),
    difficulty: str = Form("challenging"),
):
    """Generate a Q&A draft for the uploaded image."""
    import os as os_mod

    api_key = os_mod.environ.get("OPENCODE_API_KEY")
    if not api_key:
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": "No OPENCODE_API_KEY set", "upload_id": upload_id, "difficulty": difficulty},
        )

    analysis = _upload_cache.get(upload_id)
    if not analysis:
        # Try loading from disk
        img_path = UPLOADS_DIR / f"{upload_id}.jpg"
        if not img_path.exists():
            return TEMPLATES.TemplateResponse(
                request, "_author_draft.html",
                {"draft": None, "validation": None, "error": "Upload not found — please re-upload", "upload_id": upload_id, "difficulty": difficulty},
            )
        analysis = analyze_uploaded_image(img_path)
        _upload_cache[upload_id] = analysis

    img_path = UPLOADS_DIR / f"{upload_id}.jpg"
    figure_type = analysis["audit"].get("figure_type", "")
    complexity = analysis["audit"].get("complexity_score", 0.0)

    draft = draft_qa(
        image_path=img_path,
        provider="opencode",
        api_key=api_key,
        difficulty=difficulty,
        figure_type=figure_type,
        complexity_score=complexity,
    )

    if not draft:
        return TEMPLATES.TemplateResponse(
            request, "_author_draft.html",
            {"draft": None, "validation": None, "error": "Draft generation failed — API error", "upload_id": upload_id, "difficulty": difficulty},
        )

    validation = validate_draft(draft, figure_type=figure_type)

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
    # Delete file
    for ext in [".jpg", ".png"]:
        p = UPLOADS_DIR / f"{upload_id}{ext}"
        if p.exists():
            p.unlink()
    # Clear cache
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

    session = get_session()

    # Find the uploaded file
    img_path = UPLOADS_DIR / f"{upload_id}.jpg"
    if not img_path.exists():
        return HTMLResponse("Upload not found", status_code=404)

    # Compute hash
    img_hash = compute_file_hash(img_path)
    img = PILImage.open(img_path)
    w, h = img.size
    audit = audit_figure(img_path)

    # Insert Figure record
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
    if not q:
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": None, "error": "Enter search terms"}
        )
    try:
        term_list = [t.strip() for t in q.split(",") if t.strip()] if q else None
        papers = search_papers(terms=term_list, domain=domain or None, limit=limit)
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": papers, "error": ""}
        )
    except Exception as e:
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": None, "error": str(e)[:200]}
        )


@router.post("/api/arxiv/extract", response_class=HTMLResponse)
async def api_arxiv_extract(
    request: Request,
    arxiv_id: str = Form(...),
):
    """Download a paper PDF, extract figures, and return top candidates."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(STORAGE_DIR / "arxiv-manager.db"))
        c = conn.cursor()

        pdf_path = download_pdf(arxiv_id)
        extracted = extract_figures(pdf_path)

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

        # Sort by complexity, keep top 3
        figures.sort(key=lambda f: f["complexity_score"], reverse=True)
        figures = figures[:3]

        conn.close()
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_figures.html", {"figures": figures, "error": ""}
        )
    except Exception as e:
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
        return RedirectResponse(url=f"/task/{task.id}", status_code=303)

    # Show errors
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

    validation = validate_task(question, answer, answer_format)
    task = update_task(task_id, title=title, domain=domain, question=question, answer=answer, answer_format=answer_format, task_type=task_type)
    figure = get_session().get(Figure, task.figure_id) if task else None

    return TEMPLATES.TemplateResponse("task_form.html", {
        "request": request,
        "figure": figure,
        "task": task,
        "validation": validation,
    })


@router.post("/api/task/{task_id}/validate", response_class=HTMLResponse)
async def api_validate_task(request: Request, task_id: int):
    """Re-validate a task (HTMX endpoint)."""
    session = get_session()
    task = session.get(Task, task_id)
    if not task:
        return HTMLResponse("Not found", status_code=404)

    figure = session.get(Figure, task.figure_id)
    validation = validate_task(task.question, task.answer, task.answer_format)

    return TEMPLATES.TemplateResponse(request, "_validation.html", {
        "validation": validation,
    })


@router.post("/api/figure/{figure_id}/status")
async def update_figure_status(figure_id: int, status: str = Form(...)):
    """Update figure status (HTMX endpoint)."""
    session = get_session()
    figure = session.get(Figure, figure_id)
    if figure:
        figure.status = status
        session.add(figure)
        session.commit()
    return RedirectResponse(url="/images", status_code=303)


@router.post("/api/figures/bulk-reject")
async def bulk_reject_figures(figure_ids: list[int] = Form(default=[])):
    """Bulk reject multiple figures at once."""
    session = get_session()
    rejected = 0
    for fid in figure_ids:
        figure = session.get(Figure, fid)
        if figure:
            figure.status = "rejected"
            session.add(figure)
            rejected += 1
    session.commit()
    return RedirectResponse(url="/images", status_code=303)


@router.post("/api/task/{task_id}/difficulty")
async def update_task_difficulty(
    task_id: int,
    difficulty: str = Form(...),
    qwen: int = Form(0),
    gemini: int = Form(0),
):
    """Update task difficulty (HTMX endpoint)."""
    from ..tracking import set_difficulty
    set_difficulty(task_id, difficulty, qwen, gemini)
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.post("/api/task/{task_id}/submit")
async def submit_task(task_id: int):
    """Mark task as submitted (HTMX endpoint)."""
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
    session = get_session()
    task = session.get(Task, task_id)
    if task:
        task.rhea_reviewed = rhea_reviewed
        task.rhea_passed = rhea_passed
        task.rhea_notes = rhea_notes
        session.add(task)
        session.commit()
    return RedirectResponse(url=f"/task/{task_id}", status_code=303)
