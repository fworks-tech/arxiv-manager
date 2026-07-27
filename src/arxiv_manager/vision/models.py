"""Lazy-loaded ResNet-18 feature extractor model.

The model is loaded on first use (not at import time) so that code
paths that never call the vision module pay zero cost.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MODEL = None
_TRANSFORMS = None

_model_available: bool = False


def load_model() -> tuple[Any, Any] | tuple[None, None]:
    """Lazy-load ResNet-18 feature extractor and its preprocessing transforms.

    Returns (model, transforms) on success, (None, None) if torch/torchvision
    are not installed or the model cannot be loaded.
    """
    global _MODEL, _TRANSFORMS, _model_available

    if _MODEL is not None:
        return _MODEL, _TRANSFORMS

    try:
        import torch.nn as nn
        import torchvision.models as models
        from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()

        _TRANSFORMS = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _MODEL = model
        _model_available = True
        logger.info("vision: ResNet-18 loaded successfully")
        return _MODEL, _TRANSFORMS

    except ImportError:
        logger.debug("vision: torch/torchvision not available, CNN disabled")
        _model_available = False
        return None, None
    except Exception as exc:
        logger.warning("vision: failed to load ResNet-18: %s", exc)
        _model_available = False
        return None, None


def model_is_available() -> bool:
    """Return whether the CNN model was loaded successfully."""
    if _MODEL is None:
        load_model()
    return _model_available
