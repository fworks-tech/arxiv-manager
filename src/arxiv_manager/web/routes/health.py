"""Health check endpoint for monitoring and orchestration."""

from __future__ import annotations

import logging
import os
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from ... import __version__
from ...db import get_session
from . import router

logger = logging.getLogger(__name__)


def _check_db() -> dict:
    """Check database connectivity and return status + counts."""
    try:
        session = get_session()
        try:
            from sqlmodel import func, select
            from ...models import Figure, Paper, Task
            paper_count = session.exec(select(func.count(Paper.id))).one()
            fig_count = session.exec(select(func.count(Figure.id))).one()
            task_count = session.exec(select(func.count(Task.id))).one()
        finally:
            session.close()
        return {
            "status": "ok",
            "papers": paper_count,
            "figures": fig_count,
            "tasks": task_count,
        }
    except Exception as exc:
        logger.warning("health: db check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


def _check_api_key() -> dict:
    """Check whether the API key is configured."""
    key = os.environ.get("OPENCODE_API_KEY", "")
    if not key:
        return {"status": "missing"}
    return {"status": "ok", "present": bool(key)}


def _check_llm() -> dict:
    """Quick LLM connectivity check (no image, no prompt overhead)."""
    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        return {"status": "skipped", "reason": "no api key"}
    try:
        import httpx
        start = time.monotonic()
        resp = httpx.post(
            "https://opencode.ai/zen/go/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "minimax-m3",
                "messages": [{"role": "user", "content": "Reply: ok"}],
                "max_tokens": 10,
            },
            timeout=10,
        )
        elapsed = time.monotonic() - start
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content.strip():
            return {"status": "ok", "latency_ms": round(elapsed * 1000)}
        return {"status": "error", "detail": "empty response"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@router.get("/health")
def health_check(request: Request) -> JSONResponse:
    """Return overall system health.

    Checks database connectivity, API key presence, and optionally LLM
    connectivity.  Slow checks (LLM) run only when ?full=true is set.
    """
    full = request.query_params.get("full", "").lower() in ("1", "true", "yes")

    db_status = _check_db()
    key_status = _check_api_key()

    checks = {
        "db": db_status,
        "api_key": key_status,
    }

    if full and key_status["status"] == "ok":
        checks["llm"] = _check_llm()

    all_ok = all(
        v.get("status") == "ok" or v.get("status") == "skipped"
        for v in checks.values()
    )

    return JSONResponse(
        content={
            "status": "ok" if all_ok else "degraded",
            "version": __version__,
            "checks": checks,
        },
        status_code=200 if all_ok else 503,
    )
