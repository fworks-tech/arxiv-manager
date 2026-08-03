"""Determinism Checker agent — verifies answer consistency across sampled reads.

Subscribes to: fact_checked
Emits:        determinism_checked (pass) or pipeline_failed (fail)
"""

from __future__ import annotations

import logging
import os

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)


class DeterminismCheckerAgent(Agent):
    name = "determinism_checker"
    capabilities = ["determinism_check", "answer_consistency"]
    subscribe_events = ["fact_checked"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Run 3-run sampled determinism check."""
        ctx = event.context
        difficulty = ctx.difficulty

        # Only run for challenging/hardest
        if difficulty not in ("challenging", "hardest"):
            return [PipelineEvent(
                event_type="determinism_checked",
                context=ctx,
                source_agent=self.name,
                metadata={"skipped": True, "verdict": "pass"},
            )]

        question = ctx.get_artifact("question", "")
        answer = ctx.get_artifact("answer", "")
        answer_format = ctx.get_artifact("answer_format", "number")
        image_path = ctx.get_artifact("image_path")
        api_key = os.environ.get("OPENCODE_API_KEY")

        from ..authoring.ai_draft._determinism import check_determinism_for_qa

        try:
            det = check_determinism_for_qa(
                question=question,
                golden=answer,
                answer_format=answer_format,
                image_path=image_path,
                api_key=api_key,
                runs=3,
                difficulty=difficulty,
            )
        except Exception as exc:
            ctx.add_error(f"DeterminismChecker: {exc}")
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"error": str(exc)},
            )]

        draft = ctx.get_artifact("draft", {})
        draft["_determinism_checked"] = det["checked"]
        draft["_determinism_diverging"] = det["diverging"]
        draft["_determinism_failed"] = det["checked"] and not det["deterministic"]
        ctx.set_artifact("draft", draft)

        if det["checked"] and not det["deterministic"]:
            ctx.add_error("Determinism failed: " + "; ".join(str(a) for a in det["diverging"][:3]))
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"diverging": det["diverging"]},
            )]

        return [PipelineEvent(
            event_type="determinism_checked",
            context=ctx,
            source_agent=self.name,
            metadata={"deterministic": det["deterministic"], "runs": len(det.get("runs", []))},
        )]
