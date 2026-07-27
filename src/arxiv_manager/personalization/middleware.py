"""FastAPI middleware for token-based authentication.

Checks the Authorization: Bearer <token> header and attaches the
user to request.state.user.

Set OPENCODE_TESTING=1 to disable auth (used in test suite).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import validate_token
from .models import User

logger = logging.getLogger(__name__)

# Paths that don't require authentication
_PUBLIC_MATCH = (
    "/",
    "/favicon.ico",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
)
# Auth/register is explicitly public; all other /auth/ paths require auth
_AUTH_PUBLIC = ("/auth/login", "/auth/register")


def _is_public(path: str) -> bool:
    """Check if a path is public (no auth required)."""
    if path in _PUBLIC_MATCH:
        return True
    if path in _AUTH_PUBLIC:
        return True
    if path.startswith("/mcp/"):
        return True
    if path.startswith(("/figures/", "/papers/", "/uploads/")):
        return True
    # UI page routes (GET only, serve HTML)
    if (
        path.startswith("/tasks")
        or path.startswith("/author")
        or path.startswith("/images")
        or path.startswith("/stats")
        or path.startswith("/metrics")
        or path.startswith("/task/")
    ):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates Bearer tokens and attaches the user."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        path = request.url.path

        # Skip auth entirely when testing (checked per-request, not at import time)
        if os.environ.get("OPENCODE_TESTING") == "1":
            request.state.user = None
            return await call_next(request)

        # Skip auth for public paths
        if _is_public(path):
            request.state.user = None
            return await call_next(request)

        # Skip auth for static files
        if path.startswith(("/figures/", "/papers/", "/uploads/")):
            request.state.user = None
            return await call_next(request)

        # Extract token from header
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        # Also check X-API-Key header
        if not token:
            token = request.headers.get("X-API-Key", "")

        if not token:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing authentication token"},
            )

        user = validate_token(token)
        if user is None:
            # Also try as API key
            user = _validate_api_key(token)

        if user is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or expired token"},
            )

        request.state.user = user
        return await call_next(request)


def _validate_api_key(api_key: str) -> User | None:
    """Validate an API key and return the associated user."""
    from ..db import get_session

    session = get_session()
    try:
        return (
            session.query(User)
            .filter(
                User.api_key == api_key,
                User.is_active,
            )
            .first()
        )
    finally:
        session.close()


def get_current_user(request: Request) -> User | None:
    """Get the authenticated user from a request (injected by middleware).

    Usage in route handlers:
        user = get_current_user(request)
        if user is None:
            return JSONResponse(status_code=401, ...)
    """
    return getattr(request.state, "user", None)
