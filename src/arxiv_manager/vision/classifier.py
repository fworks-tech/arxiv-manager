"""Figure type classifier using CNN features + heuristic fallback.

Combines the local ResNet-18 feature extractor with the heuristic
classifier from sourcing/filters.py. Falls back to the heuristic-only
result when the CNN is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..sourcing.filters import classify_figure_type as heuristic_classify
from .extractor import extract_features, cosine_similarity

logger = logging.getLogger(__name__)

# Prototype embeddings for few-shot figure type classification.
# Populated lazily by store_prototype() and used by classify_figure()
# to compare against known examples via cosine similarity.
_PROTOTYPES: dict[str, list[np.ndarray]] = {}


def store_prototype(figure_type: str, embedding: np.ndarray) -> None:
    """Store a prototype embedding for few-shot classification.

    Args:
        figure_type: "chart_graph_text" or "general_image".
        embedding: 512-dim feature vector from extract_features().
    """
    _PROTOTYPES.setdefault(figure_type, []).append(embedding)


def clear_prototypes() -> None:
    """Clear all stored prototype embeddings (useful for testing)."""
    _PROTOTYPES.clear()


def classify_figure(image_path: str | Path) -> dict:
    """Classify a figure using CNN features + heuristic fallback.

    Uses the ResNet-18 feature extractor when available, comparing
    against stored prototypes via cosine similarity. Falls back to
    the heuristic classifier from sourcing/filters.py.

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys:
        - figure_type: "chart_graph_text" | "general_image"
        - confidence: float 0.0-1.0
        - method: "cnn" | "heuristic"
        - cnn_available: bool
    """
    heuristic = heuristic_classify(Path(image_path))
    features = extract_features(image_path)

    if features is None:
        return {
            "figure_type": heuristic["figure_type"],
            "confidence": heuristic["confidence"],
            "method": "heuristic",
            "cnn_available": False,
        }

    # If we have prototypes, use CNN-based classification
    if _PROTOTYPES:
        best_type = heuristic["figure_type"]
        best_sim = 0.0

        for ftype, prototypes in _PROTOTYPES.items():
            sims = [cosine_similarity(features, p) for p in prototypes]
            avg_sim = float(np.mean(sims)) if sims else 0.0
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_type = ftype

        return {
            "figure_type": best_type,
            "confidence": round(best_sim, 3),
            "method": "cnn",
            "cnn_available": True,
        }

    # No prototypes: use heuristic result, note CNN availability
    return {
        "figure_type": heuristic["figure_type"],
        "confidence": heuristic["confidence"],
        "method": "heuristic",
        "cnn_available": True,
    }
