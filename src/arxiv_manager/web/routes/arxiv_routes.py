"""arXiv search and extract route handlers."""
import logging

from fastapi import Form, Query, Request
from fastapi.responses import HTMLResponse

from ...sourcing.arxiv import search_papers
from ...sourcing.downloader import download_pdf
from ...sourcing.extractor import extract_figures
from ...sourcing.filters import compute_file_hash, audit_figure
from ...storage import STORAGE_DIR
from . import TEMPLATES, router

logger = logging.getLogger(__name__)


@router.get("/api/arxiv/search", response_class=HTMLResponse)
def api_arxiv_search(
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
        logger.error("arxiv search error: %s", e, exc_info=True)
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_search_results.html", {"papers": None, "error": str(e)[:200]}
        )


@router.post("/api/arxiv/extract", response_class=HTMLResponse)
def api_arxiv_extract(
    request: Request,
    arxiv_id: str = Form(...),
):
    """Download a paper PDF, extract figures, and return top candidates."""
    logger.info("arxiv extract arxiv_id=%s", arxiv_id)
    try:
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

        return TEMPLATES.TemplateResponse(
            request, "_arxiv_figures.html", {"figures": figures, "error": ""}
        )
    except Exception as e:
        logger.error("arxiv extract error: %s", e, exc_info=True)
        return TEMPLATES.TemplateResponse(
            request, "_arxiv_figures.html", {"figures": None, "error": str(e)[:200]}
        )
