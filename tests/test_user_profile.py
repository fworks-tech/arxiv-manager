"""Tests for personalization models — User, UserProfile, UserPreference CRUD."""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from arxiv_manager.personalization.auth import hash_password
from arxiv_manager.personalization.models import User, UserPreference, UserProfile
from arxiv_manager.personalization.personalizer import record_preference


@pytest.fixture
def profile_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_profile.db'}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_profile_session(profile_engine, monkeypatch):
    def _fake():
        return Session(profile_engine)

    import arxiv_manager.db as db_mod
    import arxiv_manager.personalization.personalizer as p_mod

    monkeypatch.setattr(db_mod, "get_session", _fake)
    monkeypatch.setattr(p_mod, "get_session", _fake)


@pytest.fixture
def sample_user(patch_profile_session, profile_engine):
    session = Session(profile_engine)
    user = User(username="user1", password_hash=hash_password("pass"))
    session.add(user)
    session.commit()
    uid = user.id
    session.close()
    return uid


class TestUser:
    def test_create_user(self, profile_engine):
        session = Session(profile_engine)
        user = User(username="new_user", password_hash="hash")
        session.add(user)
        session.commit()
        assert user.id is not None
        assert user.is_active is True
        assert user.role == "user"
        session.close()

    def test_unique_username(self, profile_engine):
        session = Session(profile_engine)
        session.add(User(username="dup", password_hash="h"))
        session.commit()
        session.close()

        session = Session(profile_engine)
        session.add(User(username="dup", password_hash="h"))
        with pytest.raises(Exception):
            session.commit()
        session.close()


class TestUserProfile:
    def test_create_profile(self, patch_profile_session, sample_user, profile_engine):
        session = Session(profile_engine)
        profile = UserProfile(
            user_id=sample_user,
            preferred_model="claude-sonnet-4",
            preferred_difficulty="challenging",
            prompt_style="concise",
        )
        session.add(profile)
        session.commit()
        pid = profile.id
        session.close()
        assert pid is not None

    def test_update_profile(self, patch_profile_session, sample_user, profile_engine):
        session = Session(profile_engine)
        profile = UserProfile(user_id=sample_user)
        session.add(profile)
        session.commit()
        pid = profile.id
        session.close()

        session = Session(profile_engine)
        p = session.get(UserProfile, pid)
        p.preferred_model = "deepseek-v3"
        session.add(p)
        session.commit()
        session.close()

        session = Session(profile_engine)
        p = session.get(UserProfile, pid)
        assert p.preferred_model == "deepseek-v3"
        session.close()


class TestUserPreference:
    def test_record_preference(self, patch_profile_session, sample_user):
        pref = record_preference(sample_user, "model_preference", "gemini-2.5-pro")
        assert pref.key == "model_preference"
        assert pref.value == "gemini-2.5-pro"

    def test_update_preference(self, patch_profile_session, sample_user):
        record_preference(sample_user, "model_preference", "gpt-5")
        pref = record_preference(sample_user, "model_preference", "claude-opus-4")
        assert pref.value == "claude-opus-4"

    def test_multiple_preferences(self, patch_profile_session, sample_user, profile_engine):
        record_preference(sample_user, "theme", "dark")
        record_preference(sample_user, "language", "en")
        session = Session(profile_engine)
        prefs = (
            session.query(UserPreference)
            .filter(
                UserPreference.user_id == sample_user,
            )
            .all()
        )
        assert len(prefs) == 2
        session.close()
