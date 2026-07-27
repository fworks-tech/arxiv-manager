"""Tests for personalization/personalizer.py — applying preferences to routing."""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from arxiv_manager.personalization.auth import hash_password
from arxiv_manager.personalization.models import User, UserPreference, UserProfile
from arxiv_manager.personalization.personalizer import (
    apply_preferences,
    get_effective_config,
    record_preference,
)


@pytest.fixture
def p_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_pers.db'}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def patch_pers_session(p_engine, monkeypatch):
    def _fake():
        return Session(p_engine)
    import arxiv_manager.db as db_mod
    import arxiv_manager.personalization.personalizer as p_mod
    monkeypatch.setattr(db_mod, "get_session", _fake)
    monkeypatch.setattr(p_mod, "get_session", _fake)


@pytest.fixture
def user_with_profile(patch_pers_session, p_engine):
    session = Session(p_engine)
    user = User(username="pro_user", password_hash=hash_password("pass"))
    session.add(user)
    session.commit()

    profile = UserProfile(
        user_id=user.id,
        preferred_model="deepseek-v3",
        preferred_difficulty="challenging",
        prompt_style="concise",
    )
    session.add(profile)
    session.commit()
    uid = user.id
    session.close()
    return uid


class TestApplyPreferences:

    def test_no_user_returns_unchanged(self):
        config = {"pipeline": "simple"}
        result = apply_preferences(None, config)
        assert result == config

    def test_applies_profile_preferences(self, user_with_profile, patch_pers_session):
        config = {"pipeline": "rag_enhanced"}
        result = apply_preferences(user_with_profile, config)
        assert result["preferred_model"] == "deepseek-v3"
        assert result["preferred_difficulty"] == "challenging"
        assert result["prompt_style"] == "concise"

    def test_applies_kv_preferences(self, patch_pers_session, user_with_profile, p_engine):
        record_preference(user_with_profile, "custom_key", "custom_val")
        config = {"pipeline": "simple"}
        result = apply_preferences(user_with_profile, config)
        assert result["user_custom_key"] == "custom_val"

    def test_empty_profile_returns_unchanged(self, patch_pers_session, p_engine):
        session = Session(p_engine)
        user = User(username="no_profile", password_hash=hash_password("pass"))
        session.add(user)
        session.commit()
        uid = user.id
        session.close()

        config = {"pipeline": "simple"}
        result = apply_preferences(uid, config)
        assert result == {"pipeline": "simple"}

    def test_prompt_style_default_skips(self, patch_pers_session, p_engine):
        session = Session(p_engine)
        user = User(username="default_style", password_hash=hash_password("pass"))
        session.add(user)
        session.commit()
        profile = UserProfile(user_id=user.id, prompt_style="default")
        session.add(profile)
        session.commit()
        uid = user.id
        session.close()

        config = {"pipeline": "simple"}
        result = apply_preferences(uid, config)
        assert "prompt_style" not in result


class TestGetEffectiveConfig:

    def test_integrates_with_query_router(self, user_with_profile, patch_pers_session):
        config = get_effective_config(user_with_profile, "challenging", "chart_graph_text")
        assert "pipeline" in config
        assert "preferred_model" in config
