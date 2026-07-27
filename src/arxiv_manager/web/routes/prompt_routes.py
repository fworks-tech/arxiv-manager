"""Prompt management API — hot-swappable prompt templates."""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...prompts import list_prompts, load_prompts, rollback_prompt, save_prompt
from . import router

logger = logging.getLogger(__name__)


class SavePromptBody(BaseModel):
    text: str
    author: str = ""
    description: str = ""
    tags: str = ""
    status: str = "active"


class RollbackPromptBody(BaseModel):
    version: int


@router.get("/api/prompts")
def api_list_prompts() -> JSONResponse:
    """List all prompt templates with current version info."""
    return JSONResponse({"prompts": list_prompts()})


@router.post("/api/prompts/{name}/save")
def api_save_prompt(name: str, body: SavePromptBody) -> JSONResponse:
    """Save a new version of a prompt template."""
    if not body.text:
        return JSONResponse({"error": "Missing 'text' field"}, status_code=400)
    version = save_prompt(
        name=name,
        text=body.text,
        author=body.author,
        description=body.description,
        tags=body.tags,
        status=body.status,
    )
    return JSONResponse({"name": name, "version": version, "status": "saved"})


@router.post("/api/prompts/{name}/rollback")
def api_rollback_prompt(name: str, body: RollbackPromptBody) -> JSONResponse:
    """Rollback a prompt to a previous version."""
    if body.version < 1:
        return JSONResponse({"error": "Invalid version"}, status_code=400)
    ok = rollback_prompt(name, body.version)
    if ok:
        return JSONResponse({"name": name, "version": body.version, "status": "rolled_back"})
    return JSONResponse({"error": "Rollback failed"}, status_code=404)


@router.post("/api/prompts/reload")
def api_reload_prompts() -> JSONResponse:
    """Force-reload prompts from the database."""
    load_prompts(force=True)
    return JSONResponse({"status": "reloaded"})
