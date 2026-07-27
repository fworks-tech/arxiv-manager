"""Personalization logic — applies user preferences to generation routing.

Provides functions to adjust routing configs based on user preferences
and to record implicit preference signals from user behavior.
"""

from __future__ import annotations

import logging
from typing import Any

from ..db import get_session
from .models import UserPreference, UserProfile

logger = logging.getLogger(__name__)


def apply_preferences(
    user_id: int | None,
    routing_config: dict[str, Any],
) -> dict[str, Any]:
    """Apply user preferences to a routing config.

    Modifies the config in-place and returns it.

    Args:
        user_id: The user ID, or None to skip preferences.
        routing_config: The routing config dict from query_router.

    Returns:
        The modified routing config.
    """
    if user_id is None:
        return routing_config

    profile = _get_profile(user_id)
    if profile is None:
        return routing_config

    if profile.preferred_model:
        routing_config["preferred_model"] = profile.preferred_model

    if profile.preferred_difficulty:
        routing_config["preferred_difficulty"] = profile.preferred_difficulty

    if profile.prompt_style and profile.prompt_style != "default":
        routing_config["prompt_style"] = profile.prompt_style

    kv_prefs = _get_preferences(user_id)
    for key, value in kv_prefs.items():
        routing_config[f"user_{key}"] = value

    return routing_config


def record_preference(
    user_id: int,
    key: str,
    value: str,
) -> UserPreference:
    """Record or update a key-value preference for a user.

    Args:
        user_id: The user ID.
        key: Preference key (e.g. "model_preference", "difficulty").
        value: Preference value.

    Returns:
        The created or updated UserPreference.
    """
    session = get_session()
    try:
        existing = session.query(UserPreference).filter(
            UserPreference.user_id == user_id,
            UserPreference.key == key,
        ).first()

        if existing:
            existing.value = value
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing

        pref = UserPreference(user_id=user_id, key=key, value=value)
        session.add(pref)
        session.commit()
        session.refresh(pref)
        return pref
    finally:
        session.close()


def get_effective_config(user_id: int | None, difficulty: str, figure_type: str) -> dict[str, Any]:
    """Get the effective generation config for a user.

    Combines the default routing config with user preferences.

    Args:
        user_id: The user ID, or None.
        difficulty: Requested difficulty.
        figure_type: Figure type.

    Returns:
        A routing config dict with preferences applied.
    """
    from ..services.query_router import route_request

    config = route_request(
        difficulty=difficulty,
        figure_type=figure_type,
        complexity_score=0.5,
    )
    return apply_preferences(user_id, config)


def _get_profile(user_id: int) -> UserProfile | None:
    session = get_session()
    try:
        return session.query(UserProfile).filter(
            UserProfile.user_id == user_id,
        ).first()
    finally:
        session.close()


def _get_preferences(user_id: int) -> dict[str, str]:
    session = get_session()
    try:
        rows = session.query(UserPreference).filter(
            UserPreference.user_id == user_id,
        ).all()
        return {r.key: r.value for r in rows}
    finally:
        session.close()
