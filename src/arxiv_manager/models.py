"""Data models for ArXiv Manager."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path

from sqlmodel import Field, SQLModel

# --- Enums ---


class TaskType(str, Enum):
    CHART = "chart"
    GENERAL_IMAGE = "general_image"
    SPATIAL = "spatial"


class TaskStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"


class Difficulty(str, Enum):
    EASY = "easy"
    CHALLENGING = "challenging"
    HARDEST = "hardest"


class ImageStatus(str, Enum):
    NEW = "new"
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


class IssueReport(SQLModel, table=True):
    """User-reported issue on a generation attempt.

    Used to build a negative-examples dataset and exclude low-quality
    generations from few-shot selection. Optionally stores the corrected
    answer so future generations can learn from corrections.
    """

    __tablename__ = "issue_reports"

    id: int | None = Field(default=None, primary_key=True)
    generation_attempt_id: int | None = Field(default=None, foreign_key="generation_attempts.id", index=True)
    task_id: int | None = Field(default=None, foreign_key="tasks.id", index=True)
    figure_id: int | None = Field(default=None, foreign_key="figures.id", index=True)
    reason: str = ""
    description: str = ""
    corrected_answer: str = ""
    reported_by: str = "user"
    created_at: datetime = Field(default_factory=datetime.now)


class PromptTemplateRecord(SQLModel, table=True):
    """Database-backed prompt template registry.

    Supports hot-swapping templates at runtime without code changes.
    """

    __tablename__ = "prompt_templates"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)  # e.g. "CHALLENGING_PROMPT"
    version: int = Field(default=1, index=True)
    text: str = ""
    author: str = ""
    description: str = ""
    tags: str = ""  # comma-separated
    status: str = Field(default="active")  # active | deprecated | experimental
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class TaskEvent(SQLModel, table=True):
    """Audit trail for all task state changes.

    Tracks validation results, updates, difficulty changes, Rhea reviews,
    issue reports, AI fixes, deletions, and submissions. Used by the
    Task History section and injected as context during regeneration.
    """

    __tablename__ = "task_events"

    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="tasks.id", index=True)
    event_type: str = Field(index=True)  # See docstring for allowed values
    details: str = ""  # JSON string with event-specific old/new data
    quality_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)


class SubmissionLog(SQLModel, table=True):
    """Tracks task submissions."""

    __tablename__ = "submission_logs"

    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(foreign_key="tasks.id", index=True)
    submitted_at: datetime = Field(default_factory=datetime.now)
    review_status: str = "pending"  # pending | approved | rework
    reviewer_notes: str = ""


class GenerationAttempt(SQLModel, table=True):
    """Records every Q&A generation attempt for traceability and learning.

    Links to the figure (always) and optionally to the resulting task.
    The parent_attempt_id field builds iteration trees (draft → critique → regen).
    """

    __tablename__ = "generation_attempts"

    id: int | None = Field(default=None, primary_key=True)
    figure_id: int | None = Field(default=None, foreign_key="figures.id", index=True)
    task_id: int | None = Field(default=None, foreign_key="tasks.id", index=True)
    parent_attempt_id: int | None = Field(default=None, foreign_key="generation_attempts.id")

    attempt_number: int = 0
    generation_type: str = ""  # draft | critique | verify | regen | consensus
    source_route: str = ""  # which endpoint triggered it: api_draft_qa | api_regenerate_task | cli

    # Prompt context
    prompt_template_name: str = ""
    prompt_text: str = ""
    prompt_text_hash: str = ""  # SHA-256 of the final assembled prompt, first 20 chars
    prompt_version_id: str = ""  # e.g. "CHALLENGING_PROMPT@a1b2c3d4e5f6"
    difficulty: str = ""
    figure_type: str = ""
    complexity_score: float = 0.0
    previous_question: str = ""
    feedback_text: str = ""

    # Model parameters
    model_name: str = Field(default="", index=True)
    max_tokens: int = 0
    timeout_s: int = 0

    # Raw model output
    raw_response: str = ""
    reasoning_trace: str = ""  # <think> blocks from models like minimax-m3

    # Parsed output
    generated_question: str = ""
    generated_answer: str = ""
    generated_answer_format: str = ""
    generated_task_type: str = ""

    # Validation result
    validation_quality: float = Field(default=0.0, index=True)
    validation_is_valid: bool = False
    validation_errors: str = ""  # JSON list
    validation_warnings: str = ""  # JSON list
    fact_check_errors: str = ""  # JSON list of unsupported premise claims

    # Rhea feedback (captured on submit)
    rhea_passed: bool = False
    rhea_notes: str = ""

    # Model rollout results (captured on submit from Task)
    qwen_passes: int = 0
    gemini_passes: int = 0

    # Self-critique result (for critique/regen flows)
    critique_score: int = 0
    critique_rewrite_question: str = ""
    critique_rewrite_answer: str = ""
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Outcome
    success: bool = False
    error_message: str = ""
    elapsed_ms: int = 0

    created_at: datetime = Field(default_factory=datetime.now)
