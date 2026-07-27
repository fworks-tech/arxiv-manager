"""Tests for agents/registry.py — agent registry, lookup, lifecycle."""

from __future__ import annotations

import pytest

from arxiv_manager.agents.registry import (
    clear_registry,
    find_agents,
    get_agent,
    list_agents,
    register_agent,
    unregister_agent,
)


@pytest.fixture(autouse=True)
def _reset():
    clear_registry()


class TestRegisterAndGet:

    def test_register_basic(self):
        meta = register_agent("generator", "Generates Q&A drafts", ["draft_qa"])
        assert meta.name == "generator"
        assert meta.description == "Generates Q&A drafts"
        assert meta.capabilities == ["draft_qa"]
        assert meta.status == "active"

    def test_register_with_extra(self):
        meta = register_agent("custom", capabilities=["custom"], version="2.0.0", model="gpt-4")
        assert meta.version == "2.0.0"
        assert meta.extra["model"] == "gpt-4"

    def test_get_existing(self):
        register_agent("generator", capabilities=["draft_qa"])
        meta = get_agent("generator")
        assert meta is not None
        assert meta.name == "generator"

    def test_get_missing(self):
        assert get_agent("nonexistent") is None

    def test_register_overwrites(self):
        register_agent("a", capabilities=["v1"])
        register_agent("a", capabilities=["v2"])
        assert get_agent("a").capabilities == ["v2"]


class TestFindAgents:

    def test_find_by_capability(self):
        register_agent("gen", capabilities=["draft_qa"])
        register_agent("rev", capabilities=["critique"])
        register_agent("orc", capabilities=["orchestrate", "draft_qa"])

        results = find_agents("draft_qa")
        assert len(results) == 2
        assert {r.name for r in results} == {"gen", "orc"}

    def test_find_none(self):
        register_agent("gen", capabilities=["draft_qa"])
        assert find_agents("nonexistent") == []

    def test_find_by_status(self):
        register_agent("active_agent", capabilities=["draft_qa"])
        meta = register_agent("inactive_agent", capabilities=["draft_qa"])
        meta.status = "inactive"

        results = find_agents("draft_qa", status="active")
        assert len(results) == 1
        assert results[0].name == "active_agent"

    def test_find_all_statuses(self):
        register_agent("gen", capabilities=["draft_qa"])
        meta = register_agent("rev", capabilities=["critique"])
        meta.status = "inactive"
        results = find_agents("critique", status="")
        assert len(results) == 1


class TestListAgents:

    def test_list_all(self):
        register_agent("a")
        register_agent("b")
        assert len(list_agents()) == 2

    def test_list_by_status(self):
        register_agent("active_a")
        meta = register_agent("inactive_b")
        meta.status = "inactive"
        register_agent("active_c")

        active = list_agents(status="active")
        assert len(active) == 2
        inactive = list_agents(status="inactive")
        assert len(inactive) == 1

    def test_list_empty(self):
        assert list_agents() == []


class TestUnregister:

    def test_unregister_existing(self):
        register_agent("test")
        assert unregister_agent("test") is True
        assert get_agent("test") is None

    def test_unregister_missing(self):
        assert unregister_agent("nonexistent") is False

    def test_clear(self):
        register_agent("a")
        register_agent("b")
        clear_registry()
        assert list_agents() == []
