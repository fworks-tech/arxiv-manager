"""Personalization module — user accounts, auth, preferences, and learning.

Provides basic token-based authentication and user preference management
for personalizing the AI generation pipeline.
"""

from __future__ import annotations

from .models import User, AuthToken, UserProfile, UserPreference
from .auth import (
    hash_password,
    verify_password,
    create_token,
    validate_token,
    login_user,
)
from .middleware import AuthMiddleware, get_current_user
from .personalizer import apply_preferences, record_preference, get_effective_config

__all__ = [
    "User", "AuthToken", "UserProfile", "UserPreference",
    "hash_password", "verify_password", "create_token",
    "validate_token", "login_user",
    "AuthMiddleware", "get_current_user",
    "apply_preferences", "record_preference", "get_effective_config",
]
