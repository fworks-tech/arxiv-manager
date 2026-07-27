"""Tests for personalization/auth.py — password hashing, tokens, login."""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from arxiv_manager.personalization.auth import (
    create_token,
    hash_password,
    login_user,
    validate_token,
    verify_password,
)
from arxiv_manager.personalization.models import User


@pytest.fixture
def auth_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_auth.db'}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_auth_session(auth_engine, monkeypatch):
    def _fake():
        return Session(auth_engine)
    import arxiv_manager.db as db_mod
    import arxiv_manager.personalization.auth as auth_mod
    monkeypatch.setattr(db_mod, "get_session", _fake)
    monkeypatch.setattr(auth_mod, "get_session", _fake)


@pytest.fixture
def sample_user(patch_auth_session, auth_engine):
    session = Session(auth_engine)
    user = User(username="testuser", password_hash=hash_password("password123"))
    session.add(user)
    session.commit()
    uid = user.id
    session.close()
    return uid


class TestPasswordHashing:

    def test_hash_and_verify(self):
        pwd = "my_secret_password_123"
        hashed = hash_password(pwd)
        assert verify_password(pwd, hashed) is True

    def test_wrong_password(self):
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_empty_password(self):
        hashed = hash_password("")
        assert verify_password("", hashed) is True

    def test_invalid_hash_format(self):
        assert verify_password("test", "not-a-valid-format") is False

    def test_different_hashes_each_time(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


class TestTokenCreation:

    def test_create_and_validate(self, patch_auth_session, sample_user):
        token = create_token(sample_user)
        user = validate_token(token.token)
        assert user is not None
        assert user.id == sample_user
        assert user.username == "testuser"

    def test_invalid_token(self, patch_auth_session):
        assert validate_token("nonexistent-token") is None


class TestLogin:

    def test_login_success(self, patch_auth_session, sample_user):
        result = login_user("testuser", "password123")
        assert "token" in result
        assert result["username"] == "testuser"
        assert result["user_id"] == sample_user

    def test_login_wrong_password(self, patch_auth_session, sample_user):
        result = login_user("testuser", "wrongpassword")
        assert "error" in result

    def test_login_nonexistent_user(self, patch_auth_session):
        result = login_user("nobody", "password")
        assert "error" in result

    def test_login_inactive_user(self, patch_auth_session, auth_engine, sample_user):
        session = Session(auth_engine)
        user = session.get(User, sample_user)
        user.is_active = False
        session.add(user)
        session.commit()
        session.close()

        result = login_user("testuser", "password123")
        assert "error" in result
