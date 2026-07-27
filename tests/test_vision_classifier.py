"""Tests for vision/classifier.py — figure type classification with CNN + heuristic fallback."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from arxiv_manager.vision.classifier import (
    classify_figure,
    clear_prototypes,
    store_prototype,
)


class TestClassifyFigureFallback:
    """Tests the heuristic fallback when CNN is not available."""

    @patch("arxiv_manager.vision.classifier.extract_features", return_value=None)
    @patch("arxiv_manager.vision.classifier.heuristic_classify")
    def test_fallback_to_heuristic(self, mock_heuristic, mock_extract, tmp_path):
        mock_heuristic.return_value = {
            "figure_type": "chart_graph_text",
            "confidence": 0.85,
        }
        img = tmp_path / "test.png"
        img.write_bytes(b"fake-png-data")

        result = classify_figure(str(img))

        assert result["figure_type"] == "chart_graph_text"
        assert result["confidence"] == 0.85
        assert result["method"] == "heuristic"
        assert result["cnn_available"] is False

    @patch("arxiv_manager.vision.classifier.extract_features", return_value=None)
    def test_fallback_defaults_to_general_image(self, mock_extract, tmp_path):
        img = tmp_path / "blank.png"
        img.write_bytes(b"fake-png-data")

        result = classify_figure(str(img))

        assert result["method"] == "heuristic"
        assert result["cnn_available"] is False


class TestClassifyFigureCNN:
    """Tests CNN-based classification with prototype embeddings."""

    def setup_method(self):
        clear_prototypes()

    def test_classify_with_prototypes(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(b"fake-png")

        chart_vec = np.ones(512, dtype=np.float32)

        store_prototype("chart_graph_text", chart_vec)

        with patch("arxiv_manager.vision.classifier.extract_features", return_value=chart_vec * 0.95 + 0.05):
            with patch("arxiv_manager.vision.classifier.heuristic_classify") as mock_h:
                mock_h.return_value = {
                    "figure_type": "general_image",
                    "confidence": 0.6,
                }
                result = classify_figure(str(img))

        assert result["figure_type"] == "chart_graph_text"
        assert result["method"] == "cnn"
        assert result["cnn_available"] is True

    def test_classify_no_prototypes_still_returns_heuristic(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"fake-png")

        fake_feat = np.random.randn(512).astype(np.float32)

        with patch("arxiv_manager.vision.classifier.extract_features", return_value=fake_feat):
            with patch("arxiv_manager.vision.classifier.heuristic_classify") as mock_h:
                mock_h.return_value = {
                    "figure_type": "general_image",
                    "confidence": 0.72,
                }
                result = classify_figure(str(img))

        assert result["figure_type"] == "general_image"
        assert result["confidence"] == 0.72
        assert result["method"] == "heuristic"
        assert result["cnn_available"] is True


class TestPrototypeManagement:
    def setup_method(self):
        clear_prototypes()

    def test_store_and_clear(self):
        assert _get_prototype_count() == 0
        store_prototype("chart_graph_text", np.ones(512, dtype=np.float32))
        assert _get_prototype_count() == 1
        clear_prototypes()
        assert _get_prototype_count() == 0

    def test_multiple_prototypes_same_type(self):
        store_prototype("chart_graph_text", np.ones(512, dtype=np.float32))
        store_prototype("chart_graph_text", np.ones(512, dtype=np.float32) * 2)
        assert _get_prototype_count() == 2

    def test_multiple_types(self):
        store_prototype("chart_graph_text", np.ones(512, dtype=np.float32))
        store_prototype("general_image", np.zeros(512, dtype=np.float32))
        assert _get_prototype_count() == 2


def _get_prototype_count() -> int:
    from arxiv_manager.vision.classifier import _PROTOTYPES

    return sum(len(v) for v in _PROTOTYPES.values())
