"""FastAPI route handlers for authentication and personalization."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..db import get_session
from .auth import hash_password, login_user
from .middleware import get_current_user
from .models import User, UserPreference, UserProfile
from .personalizer import apply_preferences, record_preference

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def api_login(body: dict[str, str]) -> dict[str, Any]:
    """Authenticate and receive a Bearer token.

    Body: {"username": "...", "password": "..."}
    """
    username = body.get("username", "")
    password = body.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    result = login_user(username, password)
    if "error" in result:
        raise HTTPException(status_code=401, detail=result["error"])
    return result


@router.post("/register")
def api_register(body: dict[str, str]) -> dict[str, Any]:
    """Register a new user account.

    Body: {"username": "...", "password": "..."}
    """
    username = body.get("username", "").strip()
    password = body.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="password must be at least 6 characters")

    session = get_session()
    try:
        existing = session.query(User).filter(User.username == username).first()
        if existing:
            raise HTTPException(status_code=409, detail="Username already taken")

        user = User(
            username=username,
            password_hash=hash_password(password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return {"user_id": user.id, "username": user.username}
    finally:
        session.close()


@router.get("/profile")
def api_get_profile(request: Request) -> dict[str, Any]:
    """Get the authenticated user's profile and preferences."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = get_session()
    try:
        profile = session.query(UserProfile).filter(
            UserProfile.user_id == user.id,
        ).first()

        prefs = session.query(UserPreference).filter(
            UserPreference.user_id == user.id,
        ).all()

        return {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "profile": {
                "preferred_model": profile.preferred_model if profile else "",
                "preferred_difficulty": profile.preferred_difficulty if profile else "",
                "preferred_figure_type": profile.preferred_figure_type if profile else "",
                "prompt_style": profile.prompt_style if profile else "default",
            } if profile else {},
            "preferences": {p.key: p.value for p in prefs},
        }
    finally:
        session.close()


@router.put("/profile")
def api_update_profile(request: Request, body: dict[str, str]) -> dict[str, Any]:
    """Update the authenticated user's profile."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = get_session()
    try:
        profile = session.query(UserProfile).filter(
            UserProfile.user_id == user.id,
        ).first()

        if profile is None:
            profile = UserProfile(user_id=user.id)
            session.add(profile)

        for field in ("preferred_model", "preferred_difficulty",
                      "preferred_figure_type", "prompt_style"):
            if field in body:
                setattr(profile, field, body[field])

        session.commit()
        return {"status": "updated"}
    finally:
        session.close()


@router.get("/preferences")
def api_get_preferences(request: Request) -> dict[str, str]:
    """Get all key-value preferences for the authenticated user."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = get_session()
    try:
        prefs = session.query(UserPreference).filter(
            UserPreference.user_id == user.id,
        ).all()
        return {p.key: p.value for p in prefs}
    finally:
        session.close()


@router.put("/preferences/{key}")
def api_set_preference(request: Request, key: str, body: dict[str, str]) -> dict[str, Any]:
    """Set a key-value preference."""
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    value = body.get("value", "")
    pref = record_preference(user.id, key, value)
    return {"key": pref.key, "value": pref.value}
