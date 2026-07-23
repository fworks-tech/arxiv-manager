"""Data models for ArXiv Manager."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path

from sqlmodel import SQLModel, Field, Relationship


# --- Enums ---

class TaskType(str, Enum):
    CHART = "chart"
    GENERAL_IMAGE = "general_image"
    SPATIAL = "spatial"


class TaskStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REWORK = "rework"


class Difficulty(str, Enum):
    EASY = "easy"
    CHALLENGING = "challenging"
    HARDEST = "hardest"


class ImageStatus(str, Enum):
    NEW = "new"
    REVIEWED = "reviewed"
    USED = "used"
    REJECTED = "rejected"


class AnswerFormat(str, Enum):
    NUMBER = "number"
    WORD = "word"
    PHRASE = "phrase"
    YEAR = "year"
    PERCENT = "percent"
    INTEGER = "integer"


# --- Models ---

class Paper(SQLModel, table=True):
    """Source paper from arXiv."""
    __tablename__ = "papers"

    id: str = Field(primary_key=True)  # arXiv ID e.g. "2301.12345"
    title: str
    license: str = "CC0"
    categories: str = ""  # space-separated
    source: str = "arXiv CC0"
    pdf_url: str = ""
    abstract: str = ""
    fetched_at: datetime = Field(default_factory=datetime.now)
    is_suitable: bool = False  # Paper yields at least one Challenging-suitable figure


class Figure(SQLModel, table=True):
    """Extracted image from a paper."""
    __tablename__ = "figures"

    id: int | None = Field(default=None, primary_key=True)
    paper_id: str = Field(foreign_key="papers.id", index=True)
    image_path: str  # relative to storage/
    image_hash: str = Field(index=True)  # SHA256
    caption: str = ""
    page_num: int = 0
    figure_num: str = ""
    width: int = 0
    height: int = 0
    complexity_score: float = 0.0
    figure_type: str = ""  # "chart_graph_text" | "general_image" | ""
    is_dense: bool = False  # High element density (Challenging-friendly)
    width_height_ratio: float = 0.0  # w/h for aspect filter
    filesize_bytes: int = 0  # For trash detection
    is_suitable: bool = False  # Passes all gates (set by audit)
    perceptual_hash: str = ""  # imagehash.phash for near-duplicate detection
    status: str = Field(default=ImageStatus.NEW.value, index=True)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def full_path(self) -> Path:
        from .storage import STORAGE_DIR
        return STORAGE_DIR / self.image_path

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


class Task(SQLModel, table=True):
    """A complete Q&A task unit."""
    __tablename__ = "tasks"

    id: int | None = Field(default=None, primary_key=True)
    figure_id: int = Field(foreign_key="figures.id", index=True)
    title: str = ""  # Task title for the UI
    image_path: str = ""  # Local path to image for upload
    question: str
    answer: str
    answer_format: str = Field(default=AnswerFormat.WORD.value)
    task_type: str = Field(default=TaskType.CHART.value)
    domain: str = Field(default="Computer Science")
    difficulty: str = ""
    status: str = Field(default=TaskStatus.DRAFT.value, index=True)
    ai_generated: bool = False
    qwen_passes: int = 0
    gemini_passes: int = 0
    total_runs: int = 0
    rhea_reviewed: bool = False  # Has Rhea review been requested?
    rhea_passed: bool = False  # Did Rhea review pass?
    rhea_notes: str = ""  # Rhea review feedback
    rhea_override_notes: str = ""  # Author's justification for overriding Rhea's verdict
    created_at: datetime = Field(default_factory=datetime.now)
    submitted_at: datetime | None = None
    platform_task_id: str = ""


class SubmissionLog(SQLModel, table=True):
    """Tracks task submissions."""
    __tablename__ = "submission_logs"

    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="tasks.id", index=True)
    submitted_at: datetime = Field(default_factory=datetime.now)
    review_status: str = "pending"  # pending | approved | rework
    reviewer_notes: str = ""
