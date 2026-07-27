"""Tests for agents/context.py — AgentContext and context management."""

from __future__ import annotations

from arxiv_manager.agents.context import new_context


class TestNewContext:

    def test_creates_context_with_trace_id(self):
        ctx = new_context(figure_id=1, difficulty="challenging", figure_type="chart_graph_text")
        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 16
        assert ctx.figure_id == 1
        assert ctx.difficulty == "challenging"
        assert ctx.figure_type == "chart_graph_text"
        assert ctx.artifacts == {}
        assert ctx.delegation_chain == []

    def test_uses_provided_trace_id(self):
        ctx = new_context(1, "easy", "general_image", trace_id="custom-trace")
        assert ctx.trace_id == "custom-trace"

    def test_sets_user_id(self):
        ctx = new_context(1, "easy", "chart_graph_text", user_id="user-42")
        assert ctx.user_id == "user-42"

    def test_defaults(self):
        ctx = new_context(1, "hardest", "chart_graph_text")
        assert ctx.attempt_id is None
        assert ctx.parent_context_id is None
        assert ctx.model_name == ""


class TestAgentContext:

    def test_artifacts(self):
        ctx = new_context(1, "easy", "chart_graph_text")
        ctx.set_artifact("key1", "value1")
        assert ctx.get_artifact("key1") == "value1"

    def test_artifact_default(self):
        ctx = new_context(1, "easy", "chart_graph_text")
        assert ctx.get_artifact("nonexistent", "fallback") == "fallback"

    def test_artifact_none_default(self):
        ctx = new_context(1, "easy", "chart_graph_text")
        assert ctx.get_artifact("nonexistent") is None

    def test_artifacts_shared_in_fork(self):
        ctx = new_context(1, "challenging", "chart_graph_text")
        ctx.set_artifact("draft", "original")
        child = ctx.fork("generator")
        assert child.get_artifact("draft") == "original"

    def test_fork_modifications_isolated(self):
        ctx = new_context(1, "challenging", "chart_graph_text")
        child = ctx.fork("generator")
        child.set_artifact("child_only", "secret")
        assert ctx.get_artifact("child_only") is None

    def test_fork_overrides(self):
        ctx = new_context(1, "easy", "chart_graph_text")
        child = ctx.fork("generator", difficulty="hardest")
        assert child.difficulty == "hardest"
        assert ctx.difficulty == "easy"

    def test_delegation_chain(self):
        ctx = new_context(1, "challenging", "chart_graph_text")
        child = ctx.fork("generator")
        assert child.delegation_chain == ["generator"]
        grandchild = child.fork("reviewer")
        assert grandchild.delegation_chain == ["generator", "reviewer"]

    def test_parent_context_id(self):
        ctx = new_context(1, "challenging", "chart_graph_text")
        child = ctx.fork("reviewer")
        assert child.parent_context_id == ctx.trace_id
