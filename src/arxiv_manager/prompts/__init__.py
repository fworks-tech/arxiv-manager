"""Prompt registry — hot-swappable prompt templates stored in the database.

Allows loading templates from the DB at startup, with fallback to
the hardcoded defaults in _draft_prompts.py. Supports hot-reload
via API without restarting the server.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ..db import get_session
from ..models import PromptTemplateRecord

logger = logging.getLogger(__name__)

# In-memory cache: name -> (version, text, version_id)
_cache: dict[str, tuple[int, str, str]] = {}
_cache_loaded: bool = False


def _compute_version_id(name: str, text: str) -> str:
    """Compute a content-addressed version ID."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"{name}@{h}"


def load_prompts(force: bool = False) -> None:
    """Load all active prompts from the database into the in-memory cache.

    Falls back gracefully if the table doesn't exist yet.
    """
    global _cache, _cache_loaded
    if _cache_loaded and not force:
        return

    try:
        session = get_session()
        try:
            rows = list(session.query(PromptTemplateRecord).filter(
                PromptTemplateRecord.status == "active"
            ).all())
        finally:
            session.close()

        for row in rows:
            vid = _compute_version_id(row.name, row.text)
            _cache[row.name] = (row.version, row.text, vid)
            logger.debug("prompt_registry: loaded %s v%d (%s)", row.name, row.version, vid)

        _cache_loaded = True
        logger.info("prompt_registry: loaded %d active prompt templates", len(rows))
    except Exception as e:
        logger.debug("prompt_registry: DB not available, using defaults: %s", e)


def get_prompt(name: str, fallback_text: str = "") -> str:
    """Get the text for a named prompt template.

    Falls back to the provided fallback_text if the DB doesn't have it.
    """
    load_prompts()
    cached = _cache.get(name)
    if cached:
        return cached[1]
    return fallback_text


def get_version_id(name: str) -> str:
    """Get the current version ID for a named prompt."""
    load_prompts()
    cached = _cache.get(name)
    if cached:
        return cached[2]
    return f"{name}@cache_miss"


def save_prompt(
    name: str,
    text: str,
    author: str = "",
    description: str = "",
    tags: str = "",
    status: str = "active",
) -> int:
    """Save a new version of a prompt template.

    Returns the new version number.
    """
    from datetime import datetime

    session = get_session()
    try:
        existing = session.query(PromptTemplateRecord).filter(
            PromptTemplateRecord.name == name
        ).order_by(PromptTemplateRecord.version.desc()).first()

        new_version = (existing.version + 1) if existing else 1
        now = datetime.now()

        # Deprecate old version
        if existing:
            existing.status = "deprecated"
            existing.updated_at = now

        record = PromptTemplateRecord(
            name=name,
            version=new_version,
            text=text,
            author=author,
            description=description,
            tags=tags,
            status=status,
            created_at=now,
            updated_at=now,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        version_id = record.id

        # Update cache
        vid = _compute_version_id(name, text)
        _cache[name] = (new_version, text, vid)

        logger.info("prompt_registry: saved %s v%d (%s)", name, new_version, vid)
        return new_version
    except Exception as e:
        session.rollback()
        logger.error("prompt_registry: failed to save %s: %s", name, e)
        return 0
    finally:
        session.close()


def list_prompts() -> list[dict[str, Any]]:
    """List all prompt templates with their current version info."""
    load_prompts()
    session = get_session()
    try:
        rows = list(session.query(PromptTemplateRecord).order_by(
            PromptTemplateRecord.name,
            PromptTemplateRecord.version.desc(),
        ).all())
    finally:
        session.close()

    seen: set[str] = set()
    results = []
    for row in rows:
        if row.name not in seen:
            seen.add(row.name)
            results.append({
                "name": row.name,
                "version": row.version,
                "status": row.status,
                "author": row.author,
                "description": row.description,
                "tags": row.tags.split(",") if row.tags else [],
                "updated_at": row.updated_at.isoformat(),
                "version_id": _compute_version_id(row.name, row.text),
            })
    return results


def rollback_prompt(name: str, target_version: int) -> bool:
    """Rollback a prompt to a previous version.

    Finds the target version, copies its text to a new version entry,
    and activates it.
    """
    session = get_session()
    try:
        target = session.query(PromptTemplateRecord).filter(
            PromptTemplateRecord.name == name,
            PromptTemplateRecord.version == target_version,
        ).first()

        if not target:
            logger.warning("prompt_registry: rollback target %s v%d not found", name, target_version)
            return False

        save_prompt(
            name=name,
            text=target.text,
            author="system",
            description=f"Rollback to v{target_version}",
            status="active",
        )
        logger.info("prompt_registry: rolled back %s to v%d", name, target_version)
        return True
    except Exception as e:
        logger.error("prompt_registry: rollback failed: %s", e)
        return False
    finally:
        session.close()
