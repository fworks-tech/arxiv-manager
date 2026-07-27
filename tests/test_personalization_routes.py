"""Tests for personalization/routes.py — HTTP auth and profile endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


@pytest.fixture
def p_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_pers_routes.db'}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(monkeypatch, p_engine):
    def _fake_session():
        return Session(p_engine)

    import arxiv_manager.db as db_mod
    import arxiv_manager.personalization.auth as a
    import arxiv_manager.personalization.personalizer as p
    import arxiv_manager.personalization.routes as r
    for mod in (a, r, p):
        monkeypatch.setattr(mod, "get_session", _fake_session)
    monkeypatch.setattr(db_mod, "get_session", _fake_session)

    from fastapi import FastAPI

    # Install auth middleware so protected endpoints work
    from arxiv_manager.personalization.middleware import AuthMiddleware
    from arxiv_manager.personalization.routes import router
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)

    with TestClient(app) as c:
        yield c


@pytest.fixture
def registered_user(client, p_engine):
    """Register a user and return credentials."""
    resp = client.post("/auth/register", json={
        "username": "test_user",
        "password": "password123",
    })
    assert resp.status_code == 200
    return resp.json()


class TestRegister:

    def test_register_success(self, client):
        resp = client.post("/auth/register", json={
            "username": "new_user",
            "password": "secure_password",
        })
        assert resp.status_code == 200
        assert "user_id" in resp.json()
        assert resp.json()["username"] == "new_user"

    def test_register_missing_fields(self, client):
        resp = client.post("/auth/register", json={"username": "user"})
        assert resp.status_code == 400

    def test_register_short_password(self, client):
        resp = client.post("/auth/register", json={
            "username": "user", "password": "ab",
        })
        assert resp.status_code == 400

    def test_register_duplicate(self, client):
        client.post("/auth/register", json={"username": "dup", "password": "password123"})
        resp = client.post("/auth/register", json={"username": "dup", "password": "password123"})
        assert resp.status_code == 409


class TestLogin:

    def test_login_success(self, client, registered_user):
        resp = client.post("/auth/login", json={
            "username": "test_user",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["username"] == "test_user"
        assert "user_id" in data

    def test_login_wrong_password(self, client, registered_user):
        resp = client.post("/auth/login", json={
            "username": "test_user",
            "password": "wrong",
        })
        assert resp.status_code == 401

    def test_login_missing_fields(self, client):
        resp = client.post("/auth/login", json={"username": "x"})
        assert resp.status_code == 400


class TestProfile:

    def test_get_profile_requires_auth(self, client):
        resp = client.get("/auth/profile")
        assert resp.status_code == 401

    def test_get_profile_with_token(self, client, registered_user):
        login_resp = client.post("/auth/login", json={
            "username": "test_user", "password": "password123",
        })
        token = login_resp.json()["token"]

        resp = client.get("/auth/profile", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "test_user"

    def test_update_profile(self, client, registered_user):
        login_resp = client.post("/auth/login", json={
            "username": "test_user", "password": "password123",
        })
        token = login_resp.json()["token"]

        resp = client.put("/auth/profile", json={
            "preferred_model": "gpt-5",
            "preferred_difficulty": "hardest",
            "prompt_style": "detailed",
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200


class TestPreferences:

    def test_set_preference(self, client, registered_user):
        login_resp = client.post("/auth/login", json={
            "username": "test_user", "password": "password123",
        })
        token = login_resp.json()["token"]

        resp = client.put("/auth/preferences/theme", json={"value": "dark"},
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["value"] == "dark"

    def test_get_preferences(self, client, registered_user):
        login_resp = client.post("/auth/login", json={
            "username": "test_user", "password": "password123",
        })
        token = login_resp.json()["token"]

        client.put("/auth/preferences/lang", json={"value": "en"},
                   headers={"Authorization": f"Bearer {token}"})

        resp = client.get("/auth/preferences",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["lang"] == "en"
