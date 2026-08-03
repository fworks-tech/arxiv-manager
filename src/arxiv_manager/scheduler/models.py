"""Data model for the DB-backed task scheduler."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class ScheduledTask(SQLModel, table=True):
    """A queued task for the subprocess worker pool.

    Status transitions: queued → running → done | failed | cancelled
    """

    __tablename__ = "scheduled_tasks"

    id: int | None = Field(default=None, primary_key=True)
    type: str = ""  # "generate_qa" | "regenerate_task" | "validate_batch" | "rag_index"
    status: str = Field(default="queued", index=True)  # queued | running | done | failed | cancelled
    payload: str = ""  # JSON-serialized kwargs
    priority: int = 0  # higher = more urgent
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str = ""  # JSON result or error message
    attempts: int = 0
    max_attempts: int = 3
    worker_id: int = 0  # which worker pool slot is processing this job
