"""Self-Critique agent — scores and optionally rewrites drafts.

Subscribes to: draft_generated
Emits:        draft_validated
"""

from __future__ import annotations

import logging
import os

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)


class SelfCritiqueAgent(Agent):
    name = "self_critique"
    capabilities = ["critique", "self_rewrite"]
    subscribe_events = ["draft_generated"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Run self-critique on the draft (only for challenging/hardest)."""
        ctx = event.context
        difficulty = ctx.difficulty

        # Skip self-critique for easy tasks
        if difficulty == "easy":
            return [PipelineEvent(
                event_type="draft_validated",
                context=ctx,
                source_agent=self.name,
                metadata={"skipped": True, "reason": "easy_difficulty"},
            )]

        image_path = ctx.get_artifact("image_path")
        api_key = os.environ.get("OPENCODE_API_KEY")

        from ..authoring.ai_draft.composition import draft_with_self_critique

        result = draft_with_self_critique(
            image_path=image_path,
            max_rounds=2,
            api_key=api_key,
            difficulty=difficulty,
            figure_type=ctx.figure_type,
            complexity_score=ctx.get_artifact("complexity_score", 0.0),
            previous_question=ctx.get_artifact("previous_question", ""),
            validation_context=ctx.get_artifact("validation_context", ""),
            figure_id=ctx.figure_id,
            task_id=ctx.get_artifact("task_id"),
        )

        if result is None:
            result = ctx.get_artifact("draft")
            if result is None:
                ctx.add_error("Self-critique: no draft available")
                return [PipelineEvent(
                    event_type="pipeline_failed",
                    context=ctx,
                    source_agent=self.name,
                )]

        ctx.set_artifact("draft", result)
        ctx.set_artifact("question", result.get("question", ""))
        ctx.set_artifact("answer", result.get("answer", ""))
        ctx.set_artifact("critique_score", result.get("_validation_quality", 0))

        return [PipelineEvent(
            event_type="draft_validated",
            context=ctx,
            source_agent=self.name,
            metadata={"quality": result.get("_validation_quality", 0)},
        )]
