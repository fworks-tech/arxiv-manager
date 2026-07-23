"""FastAPI web application."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..db import init_db
from ..storage import FIGURES_DIR, PAPERS_DIR, UPLOADS_DIR

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Ensure all app loggers propagate to the root uvicorn handler
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
for name in ("arxiv_manager.web.routes", "arxiv_manager.authoring.ai_draft",
              "arxiv_manager.authoring.image_analyzer", "arxiv_manager.authoring.validator",
              "arxiv_manager.sourcing.filters"):
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.INFO)
    lgr.propagate = True


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    init_db()

    app = FastAPI(title="ArXiv Manager", version="0.1.0")

    # Mount storage for serving files
    if FIGURES_DIR.exists():
        app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")
    if PAPERS_DIR.exists():
        app.mount("/papers", StaticFiles(directory=str(PAPERS_DIR)), name="papers")
    if UPLOADS_DIR.exists():
        app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

    # Register routes
    from .routes import router
    app.include_router(router)

    return app
