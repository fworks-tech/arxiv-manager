"""Token-based authentication with password hashing.

Uses SHA-256 with a random salt (no external deps). Tokens are UUID-based
strings stored in the database with optional expiry.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from ..db import get_session
from .models import AuthToken, User

logger = logging.getLogger(__name__)

_SALT_LENGTH = 16


def hash_password(password: str) -> str:
    """Hash a password with a random salt.

    Returns: salt_hex:hash_hex
    """
    salt = os.urandom(_SALT_LENGTH)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return salt.hex() + ":" + pwd_hash.hex()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: The raw password to check.
        stored: The stored hash in salt:hash format.

    Returns:
        True if the password matches.
    """
    try:
        salt_hex, hash_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        return actual == expected
    except (ValueError, AttributeError):
        return False


def create_token(
    user_id: int,
    expires_in_days: int = 30,
) -> AuthToken:
    """Create a new auth token for a user.

    Args:
        user_id: The user ID.
        expires_in_days: Token validity in days (0 = no expiry).

    Returns:
        The created AuthToken record.
    """
    session = get_session()
    try:
        token = AuthToken(
            token=uuid.uuid4().hex,
            user_id=user_id,
            expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days > 0 else None,
        )
        session.add(token)
        session.commit()
        session.refresh(token)
        return token
    finally:
        session.close()


def validate_token(token_str: str) -> User | None:
    """Validate a token string and return the associated user.

    Checks expiry and returns the User if valid, None otherwise.
    """
    session = get_session()
    try:
        record = session.query(AuthToken).filter(AuthToken.token == token_str).first()
        if record is None:
            return None
        if record.expires_at and record.expires_at < datetime.now():
            logger.debug("auth: token %s... expired", token_str[:8])
            return None

        user = session.get(User, record.user_id)
        if user is None or not user.is_active:
            return None
        return user
    finally:
        session.close()


def login_user(username: str, password: str) -> dict[str, Any]:
    """Authenticate a user and return a token.

    Args:
        username: The username.
        password: The raw password.

    Returns:
        dict with "token" on success, or "error" on failure.
    """
    session = get_session()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user is None:
            return {"error": "Invalid username or password"}

        if not verify_password(password, user.password_hash):
            return {"error": "Invalid username or password"}

        if not user.is_active:
            return {"error": "Account is disabled"}

        token = create_token(user.id)
        return {
            "token": token.token,
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
        }
    finally:
        session.close()
