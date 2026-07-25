"""Structured JSON logging for observability and tracing.

Provides a `Tracer` context manager for per-request tracing and a
structured JSON log handler that emits machine-parseable log lines.
Uses buffered writes to avoid per-line filesystem overhead.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ..storage import STORAGE_DIR

_logger = logging.getLogger("arxiv_manager.observability")

# Per-thread/async trace context
_trace_ctx: threading.local = threading.local()


# ---------------------------------------------------------------------------
# Structured JSON log handler (buffered)
# ---------------------------------------------------------------------------

class StructuredLogHandler(logging.Handler):
    """Emits structured JSON log records, one per line.

    Keeps the file handle open and flushes periodically for performance.
    """

    def __init__(self, log_path: Path | None = None) -> None:
        super().__init__()
        self.log_path = log_path or STORAGE_DIR / "_structured_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = None
        self._lock = threading.Lock()

    def _open(self) -> TextIO:
        if self._file is None:
            self._file = open(self.log_path, "a", encoding="utf-8")
        return self._file

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": self.format(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "trace_id": getattr(_trace_ctx, "trace_id", None),
                "span": getattr(_trace_ctx, "span", None),
            }
            if record.exc_info and record.exc_info[0]:
                entry["exc_type"] = record.exc_info[0].__name__
            if hasattr(record, "extra_fields"):
                entry.update(record.extra_fields)
            line = json.dumps(entry, default=str) + "\n"
            with self._lock:
                f = self._open()
                f.write(line)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
        super().close()


# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

@dataclass
class Span:
    name: str
    start_ns: int = field(default_factory=time.monotonic_ns)
    end_ns: int | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        self.end_ns = time.monotonic_ns()
        self.duration_ms = (self.end_ns - self.start_ns) / 1_000_000


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _ensure_trace() -> None:
    if not hasattr(_trace_ctx, "trace_id") or _trace_ctx.trace_id is None:
        _trace_ctx.trace_id = uuid.uuid4().hex[:16]
        _trace_ctx.trace = Trace(trace_id=_trace_ctx.trace_id)
        _trace_ctx.span = None


def current_trace_id() -> str | None:
    return getattr(_trace_ctx, "trace_id", None)


@contextmanager
def span(name: str, **metadata: Any) -> Generator[Span, None, None]:
    """Context manager for timing a named operation within the current trace."""
    _ensure_trace()
    s = Span(name=name, metadata=metadata)
    prev_span = getattr(_trace_ctx, "span", None)
    _trace_ctx.span = name
    _trace_ctx.trace.spans.append(s)
    try:
        yield s
    finally:
        s.close()
        _trace_ctx.span = prev_span


def reset_trace() -> None:
    _trace_ctx.trace_id = None
    _trace_ctx.trace = None
    _trace_ctx.span = None


# ---------------------------------------------------------------------------
# Structured logging helper
# ---------------------------------------------------------------------------

def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra: Any,
) -> None:
    """Log a structured event with extra fields."""
    record = logger.makeRecord(
        logger.name,
        level,
        fn="",
        lno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.extra_fields = extra
    logger.handle(record)


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup_structured_logging(
    log_path: Path | None = None,
    level: int = logging.INFO,
) -> StructuredLogHandler:
    """Add a StructuredLogHandler to the root logger and return it."""
    handler = StructuredLogHandler(log_path=log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03dZ"))
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return handler
