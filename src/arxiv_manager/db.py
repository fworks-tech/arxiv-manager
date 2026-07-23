"""SQLite database setup and session management."""

from __future__ import annotations

from sqlmodel import SQLModel, Session, create_engine

from .storage import DB_PATH, ensure_dirs

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    ensure_dirs()
    # Import models to register them with SQLModel metadata
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)


def reset_db() -> None:
    """Drop and recreate all tables (dev use only)."""
    ensure_dirs()
    from . import models  # noqa: F401
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return Session(engine)
