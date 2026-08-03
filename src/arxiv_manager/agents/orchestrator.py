"""Orchestrator — event-driven pipeline planner and executor.

Receives a regeneration or issue-report request, wires up the
agent pipeline via the EventBus, and drives it to completion.
"""

from __future__ import annotations

import json as _json
import logging
import time
from typing import Any

from .context import new_context
from .events import TERMINAL_EVENTS, EventBus, PipelineEvent

logger = logging.getLogger(__name__)


def _nonempty_json_list(value: str | None) -> list:
    """Parse a JSON-list column safely; return [] on None/invalid/empty."""
    try:
        data = _json.loads(value) if value else []
        return data if isinstance(data, list) else []
    except _json.JSONDecodeError:
        return []


def _check_consecutive_failures(session, task_id: int) -> str | None:
    """Check if the task has hit the consecutive-failure cap (3 in a row).

    Returns an error message if blocked, None if allowed to proceed.
    A manual edit (update/restore/ai_fix) since the last failure resets the cap.
    """
    from sqlmodel import select

    from ..models import GenerationAttempt, TaskEvent

    consecutive = session.exec(
        select(GenerationAttempt)
        .where(GenerationAttempt.task_id == task_id)
        .where(GenerationAttempt.generation_type == "regenerate_initial")
        .order_by(GenerationAttempt.created_at.desc())
        .limit(3)
    ).all()

    failed_reasons: list[str] = []
    for a in consecutive:
        if not a.success:
            failed_reasons.append("generation failed")
        elif not a.validation_is_valid:
            failed_reasons.append("validation rejected the draft")
        elif _nonempty_json_list(a.fact_check_errors):
            failed_reasons.append("premise fact-check failed")
        elif _nonempty_json_list(a.determinism_errors):
            failed_reasons.append("determinism check failed")
        else:
            failed_reasons = []
            break

    if len(failed_reasons) == 3:
        newest = consecutive[0]
        edited_since = session.exec(
            select(TaskEvent)
            .where(TaskEvent.task_id == task_id)
            .where(TaskEvent.event_type.in_(["update", "restore", "ai_fix"]))
            .where(TaskEvent.created_at > newest.created_at)
            .limit(1)
        ).first()
        if edited_since is None:
            reasons = "; ".join(failed_reasons)
            return (
                f"This task has failed regeneration 3 consecutive times ({reasons}). "
                "Each attempt costs LLM calls with no result. Edit the Q&A manually "
                "first (resets the cap), or use a different image."
            )
    return None


def _persist_result(
    task_id: int,
    draft: dict,
    difficulty: str,
    figure_type: str,
    complexity: float,
    prev_question: str,
    source_route: str,
) -> None:
    """Persist a successful pipeline result to the tasks table + telemetry."""
    from ..authoring import log_task_event
    from ..authoring._draft_telemetry import log_generation_attempt
    from ..db import get_session
    from ..models import Task
    from ..tracking import classify_difficulty

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return

        auto_difficulty = classify_difficulty(task.qwen_passes, task.gemini_passes)
        if auto_difficulty != difficulty:
            logger.info(
                "orchestrator: auto-classified task_id=%d: requested=%s actual=%s",
                task_id, difficulty, auto_difficulty,
            )
            difficulty = auto_difficulty

        task.question = draft["question"]
        task.answer = draft["answer"]
        task.answer_format = draft.get("answer_format", "number")
        task.task_type = draft.get("task_type", "chart")
        task.difficulty = difficulty
        session.add(task)
        session.commit()
        logger.info("orchestrator: persisted result task_id=%d", task_id)

        usage = draft.get("_usage", {})
        log_generation_attempt(
            figure_id=task.figure_id,
            task_id=task_id,
            attempt_number=1,
            generation_type="regenerate_initial",
            source_route=source_route,
            prompt_template_name=f"{difficulty}_{figure_type}" if figure_type else difficulty,
            prompt_version_id=draft.get("_prompt_version_id", ""),
            prompt_text_hash=draft.get("_prompt_text_hash", ""),
            model_name=draft.get("_model", difficulty),
            difficulty=difficulty,
            figure_type=figure_type,
            complexity_score=complexity,
            previous_question=prev_question,
            raw_response=draft.get("_raw_response", ""),
            reasoning_trace=draft.get("_reasoning_trace", ""),
            generated_question=draft.get("question", ""),
            generated_answer=draft.get("answer", ""),
            generated_answer_format=draft.get("answer_format", ""),
            generated_task_type=draft.get("task_type", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            validation_quality=draft.get("_validation_quality", 0.0),
            validation_is_valid=draft.get("_validation_is_valid", False),
            validation_errors=_json.dumps(draft.get("_validation_errors", [])),
            validation_warnings=_json.dumps(draft.get("_validation_warnings", [])),
            fact_check_errors=_json.dumps(draft.get("_fact_check_errors", [])),
            determinism_errors=_json.dumps(
                draft.get("_determinism_errors", []) or draft.get("_determinism_diverging", [])
            ),
            success=True,
        )

        log_task_event(
            task_id,
            "regenerate",
            details={"model": draft.get("_model", difficulty), "source": source_route},
        )
    except Exception as exc:
        logger.warning("orchestrator: persist failed task_id=%d: %s", task_id, exc)
    finally:
        session.close()


def run_pipeline(
    task_id: int,
    difficulty: str,
    source_route: str = "orchestrator",
    initial_event_type: str = "regeneration_requested",
    issue_report: dict[str, Any] | None = None,
    max_total_seconds: float = 600,
) -> dict[str, Any]:
    """Run the full event-driven agent pipeline for a task.

    This is the single entry point called by the scheduler worker.

    Args:
        task_id:           The task to regenerate.
        difficulty:        Requested difficulty level.
        source_route:      Where this was triggered from (for telemetry).
        initial_event_type: "regeneration_requested" or "issue_reported".
        issue_report:      Issue report dict if triggered by user report.
        max_total_seconds: Wall-clock timeout (default 600s = 10 minutes).

    Returns:
        Result dict with "ok", "question", "answer", etc.
    """
    start_time = time.time()

    # Build context from DB
    from ..db import get_session
    from ..models import Figure, Task

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            return {"error": "Task not found", "ok": False}

        # Consecutive-failure cap: block if 3 failures in a row
        cap_error = _check_consecutive_failures(session, task_id)
        if cap_error:
            return {"error": cap_error, "ok": False}

        figure = session.get(Figure, task.figure_id) if task.figure_id else None
        figure_type = getattr(figure, "figure_type", "") if figure else ""
        complexity = getattr(figure, "complexity_score", 0.0) if figure else 0.0
        image_path = ""
        if task.image_path:
            from ..storage import STORAGE_DIR
            image_path = str(STORAGE_DIR / task.image_path)

        prev_question = task.question
    finally:
        session.close()

    ctx = new_context(
        figure_id=task.figure_id,
        difficulty=difficulty,
        figure_type=figure_type,
    )
    ctx.set_artifact("task_id", task_id)
    ctx.set_artifact("image_path", image_path)
    ctx.set_artifact("complexity_score", complexity)
    ctx.set_artifact("previous_question", prev_question)
    ctx.set_artifact("source_route", source_route)

    # Create fresh event bus and subscribe agents from registry
    bus = EventBus()
    _subscribe_agents(bus)

    # Emit initial event
    if initial_event_type == "issue_reported" and issue_report:
        initial_event = PipelineEvent(
            event_type="issue_reported",
            context=ctx,
            source_agent="user",
            metadata={"issue_report": issue_report},
        )
    else:
        # Build validation context for the prompt
        from ..authoring.validator import validate_task
        v = validate_task(task.question, task.answer, task.answer_format,
                          figure_type=figure_type, task_type=task.task_type, difficulty=difficulty)
        validation_context = ""
        if v.errors:
            validation_context += "Errors: " + "; ".join(v.errors[:3])
        if v.warnings:
            if validation_context:
                validation_context += " | "
            validation_context += "Warnings: " + "; ".join(v.warnings[:3])
        ctx.set_artifact("validation_context", validation_context)

        initial_event = PipelineEvent(
            event_type="regeneration_requested",
            context=ctx,
            source_agent="orchestrator",
        )

    # Drive the pipeline
    produced = bus.emit(initial_event)
    terminal_reached = ctx.pipeline_status in ("completed", "failed")

    max_iterations = 20
    iteration = 0
    while produced and not terminal_reached and iteration < max_iterations:
        iteration += 1
        # Wall-clock timeout check
        if time.time() - start_time > max_total_seconds:
            ctx.add_error(f"Pipeline timed out after {max_total_seconds:.0f}s")
            logger.warning(
                "orchestrator: pipeline timed out task_id=%d after %.0fs",
                task_id, max_total_seconds,
            )
            break
        next_batch: list[PipelineEvent] = []
        for evt in produced:
            if evt.event_type in TERMINAL_EVENTS:
                terminal_reached = True
                break
            results = bus.emit(evt)
            next_batch.extend(results)
        produced = next_batch

    # Collect result
    draft = ctx.get_artifact("draft")
    if draft is None or ctx.pipeline_status == "failed":
        elapsed_ms = int((time.time() - start_time) * 1000)
        errors = ctx.errors or ["Pipeline produced no draft"]
        logger.warning("orchestrator: pipeline failed task_id=%d errors=%s", task_id, errors[:3])
        return {"error": "; ".join(errors[:2]), "ok": False, "elapsed_ms": elapsed_ms}

    # Persist result to DB
    _persist_result(
        task_id=task_id,
        draft=draft,
        difficulty=difficulty,
        figure_type=figure_type,
        complexity=complexity,
        prev_question=prev_question,
        source_route=source_route,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)
    usage = draft.get("_usage", {})
    model_name = draft.get("_model", difficulty)

    return {
        "ok": True,
        "question": draft.get("question", ""),
        "answer": draft.get("answer", ""),
        "answer_format": draft.get("answer_format", "number"),
        "task_type": draft.get("task_type", "chart"),
        "model": model_name,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "elapsed_ms": elapsed_ms,
    }


def _subscribe_agents(bus: EventBus) -> None:
    """Subscribe all pipeline agents to the event bus.

    Uses the global agent registry so that AgentMetadata.status (active/inactive)
    is respected. Falls back to hardcoded list if registry is empty.
    """
    try:
        from .registry import get_registry

        registry = get_registry()
        count = 0
        for meta in registry.list():
            if meta.status == "inactive":
                continue
            agent = meta.instance
            for event_type in agent.subscribe_events:
                bus.subscribe(event_type, agent.process)
            count += 1
        if count > 0:
            logger.info("orchestrator: subscribed %d agents from registry", count)
            return
    except Exception:
        pass

    # Fallback: import and subscribe directly
    from .determinism import DeterminismCheckerAgent
    from .fact_checker import FactCheckerAgent
    from .generator import GeneratorAgent
    from .issue_analyst import IssueAnalystAgent
    from .reviewer import ReviewerAgent
    from .self_critique import SelfCritiqueAgent
    from .verifier import VerifierAgent

    agents = [
        IssueAnalystAgent(),
        GeneratorAgent(),
        SelfCritiqueAgent(),
        FactCheckerAgent(),
        DeterminismCheckerAgent(),
        VerifierAgent(),
        ReviewerAgent(),
    ]

    for agent in agents:
        for event_type in agent.subscribe_events:
            bus.subscribe(event_type, agent.process)

    logger.info("orchestrator: subscribed %d agents (fallback)", len(agents))


# Backward-compatible alias
def run_regeneration(task_id: int, difficulty: str, source_route: str = "api_regenerate_task") -> dict:
    """Backward-compatible wrapper that delegates to run_pipeline."""
    return run_pipeline(task_id, difficulty, source_route=source_route)
