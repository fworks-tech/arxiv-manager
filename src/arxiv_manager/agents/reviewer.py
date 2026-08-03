"""Reviewer agent — LLM-powered final quality assessment.

Replaces the previous heuristic-only scoring with an LLM call that
evaluates correctness, visual reasoning depth, and difficulty fit.

Subscribes to: answer_verified
Emits:        review_completed
"""

from __future__ import annotations

import json as _json
import logging
import os
import re as _re

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)

_REVIEW_PROMPT = (
    "You are a quality reviewer for visual-reasoning Q&A tasks.\n"
    "Evaluate this draft against the figure.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Answer format: {answer_format}\n"
    "Difficulty: {difficulty}\n"
    "Figure type: {figure_type}\n\n"
    "Score 1-5 on: correctness, clarity, difficulty fit, visual reasoning depth.\n"
    "Return JSON: {{\"score\": int, \"passed\": bool, \"suggestions\": [str], \"strengths\": [str]}}"
)


class ReviewerAgent(Agent):
    name = "reviewer"
    capabilities = ["review", "quality_assessment"]
    subscribe_events = ["answer_verified"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Run LLM-powered review on the draft."""
        ctx = event.context
        draft = ctx.get_artifact("draft", {})
        question = draft.get("question", "")
        answer = draft.get("answer", "")

        if not question or not answer:
            return [PipelineEvent(
                event_type="review_completed",
                context=ctx,
                source_agent=self.name,
                metadata={"score": 1, "passed": False},
            )]

        # Attempt LLM-powered review
        api_key = os.environ.get("OPENCODE_API_KEY")
        review_result = self._llm_review(ctx, draft, api_key) if api_key else None

        # Fall back to heuristic review if LLM unavailable
        if review_result is None:
            review_result = self._heuristic_review(draft)

        ctx.set_artifact("review", review_result)

        score = review_result.get("score", 0)
        if score >= 3:
            ctx.pipeline_status = "completed"
        else:
            ctx.add_error(f"Review rejected: score {score}/5")
            # add_error already sets pipeline_status to "failed"

        return [PipelineEvent(
            event_type="pipeline_completed" if ctx.pipeline_status == "completed" else "pipeline_failed",
            context=ctx,
            source_agent=self.name,
            metadata={"score": score, "passed": review_result.get("passed", False)},
        )]

    def _llm_review(self, ctx, draft, api_key):
        """Call the LLM for a quality review."""
        try:
            from ..authoring._draft_config import CONFIG
            from ..authoring.ai_draft._api_client import _call_opencode as _call

            prompt = _REVIEW_PROMPT.format(
                question=draft.get("question", ""),
                answer=draft.get("answer", ""),
                answer_format=draft.get("answer_format", "word"),
                difficulty=ctx.difficulty,
                figure_type=ctx.figure_type,
            )

            def _parse_review(content, raw_text=""):
                text = content.strip()
                if text.startswith("```"):
                    text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
                    text = _re.sub(r"\n?```\s*$", "", text)
                try:
                    data = _json.loads(text)
                    if "score" in data:
                        return data
                except _json.JSONDecodeError:
                    pass
                return None

            result = _call(api_key, prompt, "", model=CONFIG.text_model, retries=1, parser=_parse_review)
            if result and isinstance(result, dict):
                result["agent"] = "reviewer_llm"
                return result
        except Exception as exc:
            logger.warning("reviewer: LLM review failed: %s", exc)
        return None

    def _heuristic_review(self, draft):
        """Fallback heuristic review."""
        question = (draft.get("question") or "").strip()
        answer = (draft.get("answer") or "").strip()
        quality = draft.get("_validation_quality", 0)

        strengths, suggestions = [], []

        if not question or not answer:
            return {
                "score": 1, "passed": False,
                "suggestions": ["Draft is empty"], "strengths": [],
                "agent": "reviewer_heuristic",
            }

        if quality >= 0.9:
            base_score = 5
            strengths.append("High validation quality")
        elif quality >= 0.7:
            base_score = 4
            strengths.append("Good validation quality")
        elif quality >= 0.5:
            base_score = 3
        else:
            base_score = 2
            suggestions.append("Low validation quality")

        if len(answer) < 2:
            base_score = max(base_score - 1, 1)
            suggestions.append("Answer is very short")
        if answer.lower() in question.lower():
            base_score = max(base_score - 1, 1)
            suggestions.append("Question contains the answer")

        return {
            "score": base_score, "passed": base_score >= 3,
            "suggestions": suggestions, "strengths": strengths,
            "agent": "reviewer_heuristic",
        }
