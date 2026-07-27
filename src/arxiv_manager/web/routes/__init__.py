"""Web route handlers — split into domain sub-modules."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))
router = APIRouter()
_upload_cache: dict[str, dict] = {}
_UPLOAD_CACHE_MAX = 100

# Import sub-modules to register their routes on the shared router
from ...personalization.routes import router as personalization_router  # noqa: E402
from ...scheduler.routes import router as scheduler_router  # noqa: E402
from . import (  # noqa: E402
    arxiv_routes,  # noqa: F401
    author_routes,  # noqa: F401
    health,  # noqa: F401
    lifecycle_routes,  # noqa: F401
    metrics,  # noqa: F401
    pages,  # noqa: F811, F401
    prompt_routes,  # noqa: F401
    task_routes,  # noqa: F401
)

router.include_router(scheduler_router)
router.include_router(personalization_router)
