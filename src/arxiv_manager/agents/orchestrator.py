"""Orchestrator — event-driven pipeline planner and executor.

Receives a regeneration or issue-report request, wires up the
agent pipeline via the EventBus, and drives it to completion.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .context import new_context
from .events import TERMINAL_EVENTS, EventBus, PipelineEvent

logger = logging.getLogger(__name__)


def run_pipeline(
    task_id: int,
    difficulty: str,
    source_route: str = "orchestrator",
    initial_event_type: str = "regeneration_requested",
    issue_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full event-driven agent pipeline for a task.

    This is the single entry point called by the scheduler worker.

    Args:
        task_id:           The task to regenerate.
        difficulty:        Requested difficulty level.
        source_route:      Where this was triggered from (for telemetry).
        initial_event_type: "regeneration_requested" or "issue_reported".
        issue_report:      Issue report dict if triggered by user report.

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

        figure = session.get(Figure, task.figure_id) if task.figure_id else None
        figure_type = getattr(figure, "figure_type", "") if figure else ""
        complexity = getattr(figure, "complexity_score", 0.0) if figure else 0.0
        image_path = ""
        if task.image_path:
            from ..storage import STORAGE_DIR
            image_path = str(STORAGE_DIR / task.image_path)
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
    ctx.set_artifact("previous_question", task.question)
    ctx.set_artifact("source_route", source_route)

    # Create fresh event bus and subscribe agents
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
    """Subscribe all pipeline agents to the event bus."""
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

    logger.info("orchestrator: subscribed %d agents", len(agents))


# Backward-compatible alias
def run_regeneration(task_id: int, difficulty: str, source_route: str = "api_regenerate_task") -> dict:
    """Backward-compatible wrapper that delegates to run_pipeline."""
    return run_pipeline(task_id, difficulty, source_route=source_route)
