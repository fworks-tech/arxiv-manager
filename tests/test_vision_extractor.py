"""Tests for vision/extractor.py — CNN feature extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from arxiv_manager.vision.extractor import cosine_similarity, extract_features


class _FakeFeatureExtractor(nn.Module):
    """Emits a fixed 512-dim feature vector per input."""
    def forward(self, x):
        return torch.ones(1, 512, 1, 1)


class TestExtractFeaturesFallback:
    """Tests fallback behavior when CNN model is unavailable."""

    @patch("arxiv_manager.vision.extractor.model_is_available", return_value=False)
    def test_returns_none_when_model_unavailable(self, mock_avail, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"fake-png-data")
        result = extract_features(str(img))
        assert result is None


class TestExtractFeaturesSuccess:
    """Tests feature extraction with mocked model."""

    @patch("arxiv_manager.vision.extractor.model_is_available", return_value=True)
    @patch("arxiv_manager.vision.extractor.load_model")
    def test_returns_512_dim_vector(self, mock_load, mock_avail, tmp_path):
        model = _FakeFeatureExtractor()
        transforms = lambda img: torch.randn(3, 224, 224)
        mock_load.return_value = (model, transforms)

        img_path = tmp_path / "chart.jpg"
        Image.new("RGB", (300, 300), (128, 128, 128)).save(img_path)

        vec = extract_features(str(img_path))
        assert vec is not None
        assert vec.shape == (512,)
        assert vec.dtype == np.float32

    @patch("arxiv_manager.vision.extractor.model_is_available", return_value=True)
    @patch("arxiv_manager.vision.extractor.load_model")
    def test_handles_missing_file_gracefully(self, mock_load, mock_avail):
        model = _FakeFeatureExtractor()
        transforms = lambda img: torch.randn(3, 224, 224)
        mock_load.return_value = (model, transforms)

        result = extract_features("/nonexistent/path.png")
        assert result is None


class TestCosineSimilarity:

    def test_identical_vectors(self):
        a = np.ones(512, dtype=np.float32)
        b = np.ones(512, dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 1.0], dtype=np.float32)
        b = np.array([-1.0, -1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.ones(512, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_partial_similarity(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_different_lengths(self):
        a = np.array([1.0, 1.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        expected = 1.0 / (np.sqrt(2) * 1.0)
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)
