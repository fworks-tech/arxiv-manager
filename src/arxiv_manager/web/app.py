"""FastAPI web application."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..db import init_db
from ..observability.tracer import setup_structured_logging
from ..storage import FIGURES_DIR, PAPERS_DIR, UPLOADS_DIR

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Ensure all app loggers propagate to the root uvicorn handler
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
for name in (
    "arxiv_manager.web.routes.pages",
    "arxiv_manager.web.routes.author_routes",
    "arxiv_manager.web.routes.arxiv_routes",
    "arxiv_manager.web.routes.task_routes",
    "arxiv_manager.web.routes.lifecycle_routes",
    "arxiv_manager.web.routes.metrics",
    "arxiv_manager.web.routes.health",
    "arxiv_manager.authoring.ai_draft",
    "arxiv_manager.authoring.image_analyzer",
    "arxiv_manager.authoring.validator",
    "arxiv_manager.sourcing.filters",
):
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.INFO)
    lgr.propagate = True

# Add structured JSON logging
setup_structured_logging()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    init_db()

    # Pre-warm RAG models in background to avoid cold-start delay on first request
    import threading

    def _prewarm_rag():
        try:
            from ..components.reranker import _get_reranker
            from ..services.rag_pipeline import get_pipeline
            get_pipeline()
            _get_reranker()
            logging.getLogger(__name__).info("RAG models pre-warmed successfully")
        except Exception as exc:
            logging.getLogger(__name__).warning("RAG pre-warm failed: %s", exc)

    threading.Thread(target=_prewarm_rag, daemon=True).start()

    app = FastAPI(title="ArXiv Manager", version="0.2.0")

    # Rate limiting middleware (simple IP-based, 60 req/min)
    _rate_limit_store: dict[str, list[float]] = {}

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        import time

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        timestamps = _rate_limit_store.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < 60]
        if len(timestamps) > 120:
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests", "retry_after": 60},
            )
        timestamps.append(now)
        _rate_limit_store[client_ip] = timestamps

        response = await call_next(request)
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    # Register MCP endpoints
    from ..mcp import mcp_router

    app.include_router(mcp_router)

    # Auth middleware (must be added after routes for public path detection)
    from ..personalization.middleware import AuthMiddleware

    app.add_middleware(AuthMiddleware)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Log unhandled exceptions to the structured log and return a JSON 500.

        Without this, tracebacks only reach uvicorn stderr and are invisible
        in storage/_structured_log.jsonl.
        """
        import traceback

        logger = logging.getLogger("arxiv_manager.web.app")
        logger.error(
            "unhandled exception %s %s: %s",
            request.method,
            request.url.path,
            "".join(traceback.format_exception(exc)),
        )
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Internal server error"},
        )

    return app
