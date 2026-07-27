"""Feature extraction using the local ResNet-18 CNN.

Extracts a 512-dimensional feature vector from a figure image.
Falls back gracefully if the CNN is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .models import load_model, model_is_available

logger = logging.getLogger(__name__)


def extract_features(image_path: str | Path) -> np.ndarray | None:
    """Extract a 512-dim feature vector from a figure image using ResNet-18.

    Args:
        image_path: Path to the image file.

    Returns:
        A float32 NumPy array of shape (512,) on success, or None if the
        CNN model is not available or the image cannot be processed.
    """
    if not model_is_available():
        return None

    model, transforms = load_model()
    if model is None or transforms is None:
        return None

    try:
        import torch

        image = Image.open(image_path).convert("RGB")
        input_tensor = transforms(image).unsqueeze(0)

        with torch.no_grad():
            features = model(input_tensor)

        vec = features.squeeze().cpu().numpy().astype(np.float32)
        return vec

    except Exception as exc:
        logger.debug("vision: feature extraction failed: %s", exc)
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
