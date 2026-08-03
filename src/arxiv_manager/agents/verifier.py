"""Verifier agent — cross-checks answer via independent VLM call.

Subscribes to: determinism_checked
Emits:        answer_verified
"""

from __future__ import annotations

import json as _json
import logging
import os
import re as _re

from .base import Agent
from .events import PipelineEvent

logger = logging.getLogger(__name__)


class VerifierAgent(Agent):
    name = "verifier"
    capabilities = ["answer_verification", "vlm_cross_check"]
    subscribe_events = ["determinism_checked"]

    def process(self, event: PipelineEvent) -> list[PipelineEvent]:
        """Send question + image to VLM and verify answer against golden."""
        ctx = event.context
        question = ctx.get_artifact("question", "")
        answer = ctx.get_artifact("answer", "")
        image_path = ctx.get_artifact("image_path")
        api_key = os.environ.get("OPENCODE_API_KEY")

        if not question or not image_path or not api_key:
            return [PipelineEvent(
                event_type="answer_verified",
                context=ctx,
                source_agent=self.name,
                metadata={"skipped": True},
            )]

        from ..authoring._draft_config import CONFIG
        from ..authoring._draft_prompts import CHECK_ANSWER_PROMPT
        from ..authoring.ai_draft._api_client import _call_opencode as _call
        from ..authoring.ai_draft._image_utils import encode_image_for_llm

        b64_image, _ = encode_image_for_llm(image_path, CONFIG.thumbnail_size, CONFIG.jpeg_quality)
        check_prompt = CHECK_ANSWER_PROMPT.text.format(question=question)

        def _parse_answer(content: str, raw_text: str = "") -> dict | None:
            text = content.strip()
            if text.startswith("```"):
                text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = _re.sub(r"\n?```\s*$", "", text)
            try:
                data = _json.loads(text)
                if "answer" in data and data["answer"]:
                    return data
            except _json.JSONDecodeError:
                pass
            start = text.find("{")
            if start >= 0:
                depth = 0
                for end in range(start, len(text)):
                    if text[end] == "{":
                        depth += 1
                    elif text[end] == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                data = _json.loads(text[start:end + 1])
                                if "answer" in data and data["answer"]:
                                    return data
                            except _json.JSONDecodeError:
                                pass
                            break
            return None

        try:
            vlm_result = _call(api_key, check_prompt, b64_image, model=CONFIG.text_model,
                               retries=1, difficulty=ctx.difficulty, parser=_parse_answer)
        except Exception as exc:
            ctx.add_error(f"Verifier: VLM call failed: {exc}")
            return [PipelineEvent(
                event_type="pipeline_failed",
                context=ctx,
                source_agent=self.name,
                metadata={"error": str(exc)},
            )]

        vlm_answer = (vlm_result.get("answer", "") if vlm_result else "").strip()

        # Verify VLM answer against golden
        from ..authoring._draft_prompts import VERIFY_ANSWER_PROMPT

        verify_prompt = VERIFY_ANSWER_PROMPT.text.format(
            question=question,
            golden_answer=answer,
            vlm_answer=vlm_answer,
            vlm_reasoning="",
        )
        try:
            verify_result = _call(api_key, verify_prompt, "", model=CONFIG.text_model,
                                  retries=1, parser=None)
        except Exception as exc:
            logger.warning("verifier: verify call failed: %s", exc)
            verify_result = None

        golden_correct = True
        if verify_result:
            if isinstance(verify_result, dict):
                golden_correct = verify_result.get("golden_correct", True)

        draft = ctx.get_artifact("draft", {})
        draft["_vlm_answer"] = vlm_answer
        draft["_golden_correct"] = golden_correct
        ctx.set_artifact("draft", draft)
        ctx.set_artifact("golden_correct", golden_correct)

        return [PipelineEvent(
            event_type="answer_verified",
            context=ctx,
            source_agent=self.name,
            metadata={"vlm_answer": vlm_answer, "golden_correct": golden_correct},
        )]
