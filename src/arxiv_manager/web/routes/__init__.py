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
from . import pages  # noqa: E402, F811
from . import author_routes  # noqa: E402
from . import arxiv_routes  # noqa: E402
from . import task_routes  # noqa: E402
from . import lifecycle_routes  # noqa: E402
from . import metrics  # noqa: E402
from . import health  # noqa: E402
from . import prompt_routes  # noqa: E402
from ...scheduler.routes import router as scheduler_router  # noqa: E402
from ...personalization.routes import router as personalization_router  # noqa: E402
router.include_router(scheduler_router)
router.include_router(personalization_router)
