"""Centralized storage paths and utilities."""

from __future__ import annotations

from pathlib import Path

# All storage lives under the project root's storage/ directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
DB_PATH = STORAGE_DIR / "arxiv-manager.db"
PAPERS_DIR = STORAGE_DIR / "papers"
FIGURES_DIR = STORAGE_DIR / "figures"
UPLOADS_DIR = STORAGE_DIR / "_uploads"


def ensure_dirs() -> None:
    """Create all storage directories if they don't exist."""
    for d in [STORAGE_DIR, PAPERS_DIR, FIGURES_DIR, UPLOADS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
