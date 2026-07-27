"""Data models for user accounts and personalization."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """Registered user account."""

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    password_hash: str = ""
    api_key: str = Field(default="", index=True)
    role: str = "user"  # user | admin
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AuthToken(SQLModel, table=True):
    """Authentication token for API access."""

    __tablename__ = "auth_tokens"

    id: int | None = Field(default=None, primary_key=True)
    token: str = Field(unique=True, index=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime | None = None


class UserProfile(SQLModel, table=True):
    """User-level generation preferences."""

    __tablename__ = "user_profiles"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)
    preferred_model: str = ""
    preferred_difficulty: str = ""
    preferred_figure_type: str = ""
    prompt_style: str = "default"  # concise | detailed | default
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class UserPreference(SQLModel, table=True):
    """Key-value user preferences for fine-grained control."""

    __tablename__ = "user_preferences"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    key: str = ""
    value: str = ""
