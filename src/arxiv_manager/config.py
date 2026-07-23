"""Configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from the config directory."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_domains() -> list[dict[str, Any]]:
    """Load domain search configurations."""
    cfg = load_config("domains")
    return cfg.get("domains", [])
