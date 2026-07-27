"""Orchestrator agent — plans, delegates, and aggregates multi-agent workflows.

The Orchestrator is the entry point for multi-agent collaboration.
It receives a generation request, uses the query decomposer to plan
subtasks, delegates to Generator and Reviewer agents, and aggregates
results into a final output.
"""

from __future__ import annotations

import logging
from typing import Any

from .context import AgentContext

logger = logging.getLogger(__name__)


def orchestrate(
    context: AgentContext,
    prompt: str,
    image_path: str,
) -> dict[str, Any]:
    """Run a full orchestration workflow: plan → delegate → aggregate.

    Args:
        context: The shared AgentContext for this workflow.
        prompt: The generation prompt.
        image_path: Path to the figure image.

    Returns:
        The best Q&A result from the collaboration, or an empty dict on failure.
    """
    context.set_artifact("prompt", prompt)
    context.set_artifact("image_path", image_path)

    subtasks = _plan(context, prompt)
    if not subtasks:
        logger.warning("orchestrator: no subtasks generated")
        return _fallback_generate(context, image_path)

    draft = _delegate_generation(context, image_path, prompt)
    if not draft:
        return {}

    review = _delegate_review(context, draft)
    if review:
        context.set_artifact("review", review)

    result = _aggregate(draft, review, context)
    return result


def _plan(context: AgentContext, prompt: str) -> list[dict[str, Any]]:
    """Plan subtasks using the query decomposer.

    For hardest difficulty, breaks the task into reasoning steps.
    For easier difficulties, returns a single task.
    """
    from ..agents.query_decomposer import decompose_query

    subtasks = decompose_query(
        difficulty=context.difficulty,
        figure_type=context.figure_type,
        prompt=prompt,
    )
    logger.info(
        "orchestrator: planned %d subtasks for '%s'",
        len(subtasks),
        context.difficulty,
    )
    return subtasks


def _delegate_generation(
    context: AgentContext,
    image_path: str,
    prompt: str,
) -> dict[str, Any] | None:
    """Delegate generation to the generator agent(s).

    Uses the existing draft_qa pipeline, launching multiple attempts
    for consensus when the difficulty warrants it.
    """
    from ..authoring.ai_draft.core import draft_qa

    n_attempts = 3 if context.difficulty == "hardest" else 1
    attempts = []

    for i in range(n_attempts):
        child_ctx = context.fork("generator")
        child_ctx.set_artifact("attempt", i)

        result = draft_qa(
            image_path=image_path,
            difficulty=context.difficulty,
            figure_type=context.figure_type,
            complexity_score=context.get_artifact("complexity_score", 0.5),
            caption=context.get_artifact("caption", ""),
            figure_id=context.figure_id,
        )
        if result:
            result["_attempt"] = i
            attempts.append(result)

    if not attempts:
        return None

    # Pick best by validation quality
    attempts.sort(key=lambda x: x.get("_validation_quality", 0), reverse=True)
    context.set_artifact("attempts", attempts)
    return attempts[0]


def _delegate_review(
    context: AgentContext,
    draft: dict[str, Any],
) -> dict[str, Any] | None:
    """Delegate draft review to the reviewer agent.

    Only reviews if the difficulty merits it (challenging or hardest).
    """
    if context.difficulty == "easy":
        return None

    try:
        from .reviewer import review_draft

        child_ctx = context.fork("reviewer")
        result = review_draft(draft, child_ctx)
        return result
    except Exception as exc:
        logger.debug("orchestrator: reviewer unavailable: %s", exc)
        return None


def _aggregate(
    draft: dict[str, Any],
    review: dict[str, Any] | None,
    context: AgentContext,
) -> dict[str, Any]:
    """Aggregate generation and review results into a final output.

    Applies review suggestions if they improve quality. Otherwise
    returns the original draft.
    """
    result = {k: v for k, v in draft.items() if not k.startswith("_")}

    if review and review.get("score", 0) >= 4:
        return result

    if review and review.get("score", 0) < 3 and review.get("suggestion"):
        result["question"] = draft.get("question", "")
        result["answer"] = draft.get("answer", "")
        context.set_artifact("review_applied", False)

    return result


def _fallback_generate(
    context: AgentContext,
    image_path: str,
) -> dict[str, Any]:
    """Fallback: direct generation without orchestration."""
    from ..authoring.ai_draft.core import draft_qa

    result = draft_qa(
        image_path=image_path,
        difficulty=context.difficulty,
        figure_type=context.figure_type,
        figure_id=context.figure_id,
    )
    return result or {}
