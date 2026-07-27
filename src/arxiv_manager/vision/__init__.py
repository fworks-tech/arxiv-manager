"""Local vision module — CNN-based figure analysis and classification.

Provides a ResNet-18 feature extractor that complements (and can replace)
the heuristic-based figure type classification in sourcing/filters.py.

The CNN model is lazy-loaded so importing this module has zero overhead
when torch/torchvision are not available.
"""

from __future__ import annotations

from .classifier import classify_figure
from .extractor import extract_features

__all__ = ["classify_figure", "extract_features"]
