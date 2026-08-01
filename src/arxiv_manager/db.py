"""SQLite database setup and session management."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from .storage import DB_PATH, ensure_dirs

logger = logging.getLogger(__name__)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def _enable_wal() -> None:
    """Enable WAL mode for better concurrent read performance."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.commit()


def _migrate() -> None:
    """Add columns to existing tables that were added after initial creation."""
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(tasks)")).fetchall()]
        if "rhea_override_notes" not in cols:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN rhea_override_notes TEXT NOT NULL DEFAULT ''"))
            conn.commit()
            logger.info("migration: added tasks.rhea_override_notes")

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(generation_attempts)")).fetchall()]
        if "prompt_text_hash" not in cols:
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN prompt_text_hash TEXT NOT NULL DEFAULT ''"))
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN prompt_version_id TEXT NOT NULL DEFAULT ''"))
            conn.commit()
            logger.info("migration: added generation_attempts.prompt_text_hash and prompt_version_id")

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(generation_attempts)")).fetchall()]
        if "input_tokens" not in cols:
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0"))
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0"))
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN total_tokens INTEGER NOT NULL DEFAULT 0"))
            conn.commit()
            logger.info("migration: added generation_attempts.input_tokens, output_tokens, total_tokens")

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(generation_attempts)")).fetchall()]
        if "rhea_passed" not in cols:
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN rhea_passed INTEGER NOT NULL DEFAULT 0"))
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN rhea_notes TEXT NOT NULL DEFAULT ''"))
            conn.commit()
            logger.info("migration: added generation_attempts.rhea_passed, rhea_notes")

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(generation_attempts)")).fetchall()]
        if "qwen_passes" not in cols:
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN qwen_passes INTEGER NOT NULL DEFAULT 0"))
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN gemini_passes INTEGER NOT NULL DEFAULT 0"))
            conn.commit()
            logger.info("migration: added generation_attempts.qwen_passes, gemini_passes")

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(generation_attempts)")).fetchall()]
        if "fact_check_errors" not in cols:
            conn.execute(text("ALTER TABLE generation_attempts ADD COLUMN fact_check_errors TEXT NOT NULL DEFAULT ''"))
            conn.commit()
            logger.info("migration: added generation_attempts.fact_check_errors")

    with engine.connect() as conn:
        tables = [row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()]
        if "prompt_templates" not in tables:
            from . import models  # noqa: F401

            SQLModel.metadata.create_all(engine, tables=["prompt_templates"])

    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(issue_reports)")).fetchall()]
        if "corrected_answer" not in cols:
            conn.execute(text("ALTER TABLE issue_reports ADD COLUMN corrected_answer TEXT NOT NULL DEFAULT ''"))
            conn.commit()
            logger.info("migration: added issue_reports.corrected_answer")
            logger.info("migration: created prompt_templates table")

    with engine.connect() as conn:
        tables = [row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()]
        if "issue_reports" not in tables:
            from . import models  # noqa: F401

            SQLModel.metadata.create_all(engine, tables=["issue_reports"])
            logger.info("migration: created issue_reports table")


def init_db() -> None:
    """Create all tables if they don't exist."""
    ensure_dirs()
    from . import models  # noqa: F401
    from .personalization import models as _pers_models  # noqa: F401
    from .scheduler import models as _sched_models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _enable_wal()
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
