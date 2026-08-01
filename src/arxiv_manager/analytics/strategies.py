"""Strategy analytics — question-strategy classes × model/verdict outcomes.

Aggregates every available signal about whether generated questions actually
fail the target models:

- Realm verdicts (SubmissionLog.review_status + issue_report reasons)
- manual Qwen/Gemini pass counts (Task.qwen_passes / gemini_passes)
- check-answer VLM match (TaskEvent "check_answer")
- determinism results (TaskEvent "determinism_check")

The data-provider seam (task_verdict_sources) lets a future rollout engine
plug per-run model verdicts in as an additional source without changing
the aggregation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from sqlmodel import select

logger = logging.getLogger(__name__)

STRATEGY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("percentage_change", re.compile(r"percent(age)? (point )?(increase|decrease|change)|increas.+by .+percent", re.I)),
    ("rank", re.compile(r"\brank|position|drops|rises|how many positions", re.I)),
    ("cross_panel_sum_diff", re.compile(r"\bsum|total|difference between|exceeds|difference of|averag", re.I)),
    ("spatial", re.compile(r"\b(above|below|left|right|top|bottom|corner|between)\b", re.I)),
    ("comparison", re.compile(r"\bwhich (is|panel|one)|higher|larger|steeper|greater|largest|smallest", re.I)),
    ("counting", re.compile(r"\bhow many|count|\bnumber of\b", re.I)),
    ("single_lookup", re.compile(r"^what is the .+ (of|in) ", re.I)),
]
DEFAULT_STRATEGY = "other"


def classify_strategy(question: str) -> str:
    """Assign a coarse strategy class to a question by pattern matching.

    First matching pattern wins; order matters (specific before generic).
    """
    q = (question or "").strip()
    if not q:
        return DEFAULT_STRATEGY
    for name, pattern in STRATEGY_PATTERNS:
        if pattern.search(q):
            return name
    return DEFAULT_STRATEGY


def task_verdict_sources(session, task) -> list[dict[str, Any]]:
    """Collect every available verdict signal for one task.

    Returns a list of {"source": str, "verdict": str, "detail": str}.
    Verdict values: "pass" | "fail" | "too_easy" | "too_hard" | "approved" | "unknown".
    This is the pluggable seam: a rollout engine can append per-model run
    verdicts here.
    """
    from ..models import SubmissionLog, TaskEvent

    sources: list[dict[str, Any]] = []

    # Realm verdict via submission log (P6)
    logs = list(
        session.exec(
            select(SubmissionLog).where(SubmissionLog.task_id == task.id).order_by(SubmissionLog.submitted_at.desc())
        ).all()
    )
    if logs:
        latest = logs[0]
        if latest.review_status in ("approved", "too_easy", "too_hard", "rework"):
            sources.append(
                {
                    "source": "realm",
                    "verdict": latest.review_status,
                    "detail": latest.reviewer_notes or "",
                }
            )

    # Issue reports (Realm feedback arrives here today)
    events = list(
        session.exec(select(TaskEvent).where(TaskEvent.task_id == task.id).order_by(TaskEvent.created_at)).all()
    )
    for e in events:
        try:
            details = json.loads(e.details) if e.details else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if e.event_type == "issue_report":
            reason = details.get("reason", "")
            if reason in ("too_easy", "too_hard", "unclear", "wrong_answer"):
                sources.append(
                    {
                        "source": "issue_report",
                        "verdict": reason,
                        "detail": str(details.get("description", ""))[:120],
                    }
                )
        elif e.event_type == "check_answer":
            match = details.get("match", False)
            sources.append(
                {
                    "source": "check_answer",
                    "verdict": "pass" if match else "fail",
                    "detail": f"vlm={details.get('vlm_answer', '')}",
                }
            )
        elif e.event_type == "determinism_check":
            deterministic = details.get("deterministic", False)
            sources.append(
                {
                    "source": "determinism",
                    "verdict": "pass" if deterministic else "fail",
                    "detail": f"runs={len(details.get('model_answers', []))}",
                }
            )

    # Manual pass counts
    if task.qwen_passes > 0:
        sources.append({"source": "qwen", "verdict": "pass", "detail": f"{task.qwen_passes}/{task.total_runs}"})
    if task.gemini_passes > 0:
        sources.append({"source": "gemini", "verdict": "pass", "detail": f"{task.gemini_passes}/{task.total_runs}"})

    return sources


def build_strategy_analytics(session=None) -> dict[str, Any]:
    """Aggregate verdict sources across all tasks, grouped by strategy class."""
    from ..db import get_session
    from ..models import Task

    own_session = session is None
    if own_session:
        session = get_session()
    try:
        tasks = list(session.exec(select(Task)).all())

        stats: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for t in tasks:
            strategy = classify_strategy(t.question)
            if strategy not in stats:
                stats[strategy] = {
                    "count": 0,
                    "realm_too_easy": 0,
                    "realm_too_hard": 0,
                    "realm_approved": 0,
                    "issue_reports": 0,
                    "determinism_pass": 0,
                    "determinism_fail": 0,
                    "determinism_checked": 0,
                    "check_answer_pass": 0,
                    "check_answer_fail": 0,
                    "check_answer_checked": 0,
                    "qwen_passes_any": 0,
                    "gemini_passes_any": 0,
                    "hardest": 0,
                }
                order.append(strategy)
            s = stats[strategy]
            s["count"] += 1
            if t.difficulty == "hardest":
                s["hardest"] += 1

            for src in task_verdict_sources(session, t):
                if src["source"] == "realm":
                    if src["verdict"] == "too_easy":
                        s["realm_too_easy"] += 1
                    elif src["verdict"] == "too_hard":
                        s["realm_too_hard"] += 1
                    elif src["verdict"] == "approved":
                        s["realm_approved"] += 1
                elif src["source"] == "issue_report":
                    s["issue_reports"] += 1
                elif src["source"] == "determinism":
                    s["determinism_checked"] += 1
                    if src["verdict"] == "pass":
                        s["determinism_pass"] += 1
                    else:
                        s["determinism_fail"] += 1
                elif src["source"] == "check_answer":
                    s["check_answer_checked"] += 1
                    if src["verdict"] == "pass":
                        s["check_answer_pass"] += 1
                    else:
                        s["check_answer_fail"] += 1
                elif src["source"] == "qwen":
                    s["qwen_passes_any"] += 1
                elif src["source"] == "gemini":
                    s["gemini_passes_any"] += 1

        return {"strategies": stats, "order": order, "total_tasks": len(tasks)}
    finally:
        if own_session:
            session.close()


DATA_PROVIDERS: list[Callable] = [task_verdict_sources]
