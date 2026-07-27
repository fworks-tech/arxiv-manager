"""Personalization module — user accounts, auth, preferences, and learning.

Provides basic token-based authentication and user preference management
for personalizing the AI generation pipeline.
"""

from __future__ import annotations

from .auth import (
    create_token,
    hash_password,
    login_user,
    validate_token,
    verify_password,
)
from .middleware import AuthMiddleware, get_current_user
from .models import AuthToken, User, UserPreference, UserProfile
from .personalizer import apply_preferences, get_effective_config, record_preference

__all__ = [
    "User",
    "AuthToken",
    "UserProfile",
    "UserPreference",
    "hash_password",
    "verify_password",
    "create_token",
    "validate_token",
    "login_user",
    "AuthMiddleware",
    "get_current_user",
    "apply_preferences",
    "record_preference",
    "get_effective_config",
]
