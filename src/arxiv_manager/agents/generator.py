"""Generator agent — wraps the existing draft_qa pipeline.

Subscribes to: regeneration_requested
Emits:        draft_generated
"""

from __future__ import annotations

import logging
import os

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)


class GeneratorAgent(Agent):
    name = "generator"
    capabilities = ["draft_qa", "generation"]
    subscribe_events = ["regeneration_requested"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Generate a Q&A draft from the figure image."""
        ctx = event.context
        image_path = ctx.get_artifact("image_path")
        api_key = os.environ.get("OPENCODE_API_KEY")

        # Apply difficulty_delta from IssueAnalyst if present
        delta = ctx.get_artifact("difficulty_delta", 0)
        if delta:
            levels = ["easy", "challenging", "hardest"]
            idx = levels.index(ctx.difficulty) if ctx.difficulty in levels else 1
            ctx.difficulty = levels[max(0, min(2, idx + delta))]

        if not image_path or not api_key:
            ctx.add_error("Generator: missing image_path or OPENCODE_API_KEY")
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"error": "missing inputs"},
            )]

        from ..authoring.ai_draft.core import draft_qa

        feedback = ctx.get_artifact("retry_feedback", "")
        issue_hints = ctx.get_artifact("issue_hints", [])
        if issue_hints:
            feedback = (feedback + " " if feedback else "") + "Issue hints: " + ", ".join(issue_hints)

        try:
            draft = draft_qa(
                image_path=image_path,
                api_key=api_key,
                difficulty=ctx.difficulty,
                figure_type=ctx.figure_type,
                complexity_score=ctx.get_artifact("complexity_score", 0.0),
                previous_question=ctx.get_artifact("previous_question", ""),
                validation_context=ctx.get_artifact("validation_context", ""),
                feedback=feedback,
                figure_id=ctx.figure_id,
                task_id=ctx.get_artifact("task_id"),
            )
        except Exception as exc:
            logger.warning("generator: draft_qa failed: %s", exc)
            ctx.add_error(f"Generator: {exc}")
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"error": str(exc)},
            )]

        if draft is None:
            ctx.add_error("Generator: draft_qa returned None — check image path, API key, and prompt")
            logger.warning("generator: draft_qa returned None for figure_id=%d", ctx.figure_id)
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"error": "draft_qa returned None"},
            )]

        ctx.set_artifact("draft", draft)
        ctx.set_artifact("question", draft.get("question", ""))
        ctx.set_artifact("answer", draft.get("answer", ""))
        ctx.set_artifact("answer_format", draft.get("answer_format", "word"))

        return [PipelineEvent(
            event_type="draft_generated",
            context=ctx,
            source_agent=self.name,
            metadata={"model": draft.get("_model", ""), "quality": draft.get("_validation_quality", 0)},
        )]
