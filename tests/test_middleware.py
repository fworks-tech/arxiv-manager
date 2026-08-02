"""Tests for the auth middleware public-path rules and dashboard access."""

import pytest

from arxiv_manager.personalization.middleware import _is_public


class TestPublicPaths:
    @pytest.mark.parametrize(
        "path",
        [
            "/",
            "/health",
            "/tasks",
            "/tasks/table",
            "/task/44",
            "/author",
            "/images",
            "/stats",
            "/metrics",
            "/analytics/strategies",
            "/analytics/anything",
            "/figures/x.jpg",
            "/api/task/44/regenerate",
        ],
    )
    def test_public_pages(self, path):
        assert _is_public(path) is True

    @pytest.mark.parametrize(
        "path",
        ["/auth/profile", "/auth/xyz", "/some/unknown/page"],
    )
    def test_non_public(self, path):
        assert _is_public(path) is False


class TestAnalyticsPageAccess:
    def test_strategies_page_served(self, test_client):
        resp = test_client.get("/analytics/strategies")
        assert resp.status_code == 200
        assert "Strategy Analytics" in resp.text
