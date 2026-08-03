"""Issue Analyst agent — analyzes user-reported issues and generates
structured recommendations for the pipeline.

Subscribes to: issue_reported
Emits:        regeneration_requested (with strategy hints in context.artifacts)
"""

from __future__ import annotations

import logging

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)

# Pattern classification keywords
_STRATEGY_HINTS: dict[str, list[str]] = {
    "too_easy": ["simple_lookup", "single_matchmaking", "counting_only", "chart_comparison"],
    "wrong_answer": ["calculation_error", "ocr_misread", "panel_mismatch"],
    "not_visual": ["text_only_question", "no_image_needed"],
    "not_challenging": ["single_step", "no_reasoning_depth"],
    "unclear": ["ambiguous_question", "multiple_interpretations"],
}


class IssueAnalystAgent(Agent):
    name = "issue_analyst"
    capabilities = ["issue_analysis", "pattern_detection", "strategy_recommendation"]
    subscribe_events = ["issue_reported"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Analyze the issue report and emit a regeneration request with hints."""
        ctx = event.context
        report = event.metadata.get("issue_report", {})

        reason = report.get("reason", "")
        description = report.get("description", "")
        corrected_answer = report.get("corrected_answer", "")

        # Classify strategy hints based on reason + description
        hints: list[str] = []
        text = f"{reason} {description}".lower()
        for strategy, keywords in _STRATEGY_HINTS.items():
            if any(kw in text for kw in keywords) or reason == strategy:
                hints.append(strategy)

        # Determine difficulty adjustment
        difficulty_delta = 0
        if reason == "too_easy":
            difficulty_delta = 1
        elif reason == "too_hard":
            difficulty_delta = -1

        # Store recommendations in context artifacts
        ctx.set_artifact("issue_hints", hints)
        ctx.set_artifact("difficulty_delta", difficulty_delta)
        ctx.set_artifact("corrected_answer", corrected_answer)
        ctx.set_artifact("issue_reason", reason)
        ctx.set_artifact("issue_description", description)

        logger.info(
            "issue_analyst: figure_id=%s reason=%s hints=%s delta=%d",
            ctx.figure_id, reason, hints, difficulty_delta,
        )

        return [PipelineEvent(
            event_type="regeneration_requested",
            context=ctx,
            source_agent=self.name,
            metadata={"hints": hints, "difficulty_delta": difficulty_delta},
        )]
