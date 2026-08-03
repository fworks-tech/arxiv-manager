"""Tests for verified model capability cards and dual-target prompt content."""

from arxiv_manager.authoring._draft_prompts import (
    CHALLENGING_PROMPT,
    HARDEST_PROMPT,
    SELF_CRITIQUE_PROMPT,
    SPATIAL_HARDEST_PROMPT,
)


class TestModelCards:
    def test_cards_contain_both_models(self):
        from arxiv_manager.authoring._model_cards import MODEL_CARDS

        assert "qwen3.6-35b-a3b" in MODEL_CARDS
        assert "gemini-3.5-flash" in MODEL_CARDS

    def test_qwen_benchmarks_verified(self):
        from arxiv_manager.authoring._model_cards import MODEL_CARDS

        b = MODEL_CARDS["qwen3.6-35b-a3b"]["vision_benchmarks"]
        assert b["ODInW13 (in-the-wild object detection/counting)"] == 50.8
        assert b["ZEROBench_sub (novel zero-shot formats)"] == 34.4
        assert b["OmniDocBench 1.5 (OCR/document reading)"] == 89.9
        assert b["AI2D (diagram understanding)"] == 92.7

    def test_gemini_strengths_documented(self):
        from arxiv_manager.authoring._model_cards import MODEL_CARDS

        g = MODEL_CARDS["gemini-3.5-flash"]
        assert g["strengths"]
        assert any("reasoning" in s.lower() for s in g["strengths"])


class TestDualTargetPrompts:
    def test_hardest_targets_qwen(self):
        text = HARDEST_PROMPT.text
        assert "Qwen 3.6-35B-A3B" in text
        assert "Realm classifies difficulty by Qwen" in text
        assert "ODInW13 50.8" in text

    def test_hardest_avoids_and_prefers_lists(self):
        text = HARDEST_PROMPT.text
        # Prompt must have clear avoid/prefer guidance for Qwen-targeting strategies
        assert "DO NOT generate these" in text or "AVOID" in text
        assert "PROVEN" in text and "strategies" in text

    def test_spatial_hardest_carries_fact_safety(self):
        assert "FACT SAFETY" in SPATIAL_HARDEST_PROMPT.text

    def test_self_critique_scores_qwen(self):
        text = SELF_CRITIQUE_PROMPT.text
        assert "Qwen 3.6-35B-A3B" in text
        assert "Realm classifies difficulty by Qwen" in text

    def test_challenging_still_targets_qwen_only(self):
        # Challenging = Qwen fails, Gemini passes — must NOT demand both fail.
        assert "BOTH" not in CHALLENGING_PROMPT.text
