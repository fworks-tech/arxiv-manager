"""Fact Checker agent — verifies question premises against the image.

Subscribes to: draft_validated
Emits:        fact_checked (pass) or pipeline_failed (fail)
"""

from __future__ import annotations

import logging
import os

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)


class FactCheckerAgent(Agent):
    name = "fact_checker"
    capabilities = ["fact_check", "premise_verification"]
    subscribe_events = ["draft_validated"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Run adversarial premise fact-check on the draft."""
        ctx = event.context
        difficulty = ctx.difficulty

        # Skip fact-check for easy tasks
        if difficulty == "easy":
            return [PipelineEvent(
                event_type="fact_checked",
                context=ctx,
                source_agent=self.name,
                metadata={"skipped": True, "verdict": "pass"},
            )]

        question = ctx.get_artifact("question", "")
        image_path = ctx.get_artifact("image_path")
        api_key = os.environ.get("OPENCODE_API_KEY")

        if not question or not image_path or not api_key:
            ctx.add_error("FactChecker: missing inputs")
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"error": "missing inputs"},
            )]

        from ..authoring.ai_draft._fact_checker import fact_check_draft

        fc = fact_check_draft(
            question=question,
            image_path=image_path,
            api_key=api_key,
            difficulty=difficulty,
        )

        draft = ctx.get_artifact("draft", {})
        draft["_fact_check_checked"] = fc["checked"]
        draft["_fact_check_errors"] = fc["unsupported"]
        draft["_fact_check_claims"] = fc["claims"]
        draft["_fact_check_failed"] = fc["verdict"] == "fail"
        ctx.set_artifact("draft", draft)

        if fc["checked"] and fc["verdict"] == "fail":
            ctx.add_error("Fact-check failed: " + "; ".join(fc["unsupported"][:3]))
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"verdict": "fail", "unsupported": fc["unsupported"]},
            )]

        return [PipelineEvent(
            event_type="fact_checked",
            context=ctx,
            source_agent=self.name,
            metadata={"verdict": "pass", "checked": fc["checked"]},
        )]
