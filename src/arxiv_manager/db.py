"""SQLite database setup and session management."""

from __future__ import annotations

import logging
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy import text

from .storage import DB_PATH, ensure_dirs

logger = logging.getLogger(__name__)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False)


def _migrate() -> None:
    """Add columns to existing tables that were added after initial creation."""
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(tasks)")).fetchall()]
        if "rhea_override_notes" not in cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN rhea_override_notes TEXT NOT NULL DEFAULT ''"))
            conn.commit()
            logger.info("migration: added tasks.rhea_override_notes")


def init_db() -> None:
    """Create all tables if they don't exist."""
    ensure_dirs()
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    _migrate()


def reset_db() -> None:
    """Drop and recreate all tables (dev use only)."""
    ensure_dirs()
    from . import models  # noqa: F401
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return Session(engine)
