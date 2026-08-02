"""Tests for answer determinism checking."""


import pytest

from arxiv_manager.authoring.ai_draft._determinism import (
    check_determinism_for_qa,
    matches_golden,
    normalize_number,
)


class TestNormalizeNumber:
    def test_plain_int(self):
        assert normalize_number("20") == 20.0

    def test_float(self):
        assert normalize_number("30.5") == 30.5

    def test_percent_suffix(self):
        assert normalize_number("20%") == 20.0

    def test_commas(self):
        assert normalize_number("1,234") == 1234.0

    def test_units_suffix(self):
        assert normalize_number("15 ms") == 15.0

    def test_negative(self):
        assert normalize_number("-3") == -3.0

    def test_garbage(self):
        assert normalize_number("cannot determine") is None

    def test_empty(self):
        assert normalize_number("") is None
        assert normalize_number(None) is None


class TestMatchesGolden:
    @pytest.mark.parametrize(
        "model,golden",
        [
            ("20", "20"),
            ("20.0", "20"),
            ("20%", "20"),
            ("20.5", "20.5"),
            ("30.50", "30.5"),
        ],
    )
    def test_numeric_equivalents(self, model, golden):
        assert matches_golden(model, golden, "number") is True

    @pytest.mark.parametrize(
        "model,golden",
        [("21", "20"), ("19", "20"), ("14", "15"), ("20.5", "20")],
    )
    def test_numeric_mismatches(self, model, golden):
        assert matches_golden(model, golden, "number") is False

    def test_word_case_insensitive(self):
        assert matches_golden("DigiLux", "digilux", "word") is True

    def test_word_semantic_fallback(self, monkeypatch):
        from arxiv_manager.authoring.ai_draft import _determinism as det

        monkeypatch.setattr(
            det, "_semantic_equivalent", lambda a, b, api_key: a == "top-left" or b == "top-left"
        )
        assert matches_golden("top left", "top-left", "word", api_key="k") is True

    def test_empty_returns_false(self):
        assert matches_golden("", "20", "number") is False
        assert matches_golden("20", "", "number") is False


class TestCheckDeterminismForQa:
    @staticmethod
    def _answers(*answers):
        seq = list(answers)

        def fake_call(api_key, prompt, b64, **kw):
            a = seq.pop(0) if seq else None
            if a is None:
                return None
            return {"answer": str(a), "reasoning": "observed"}

        return fake_call

    def test_all_runs_match(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _determinism as det

        monkeypatch.setattr(det, "_call_opencode", self._answers("15", "15.0", "15"))
        r = check_determinism_for_qa("Q?", "15", "number", sample_image_chart_path, "key", runs=3)
        assert r["deterministic"] is True
        assert r["diverging"] == []
        assert r["checked"] is True
        assert all(run["match"] for run in r["runs"])

    def test_one_run_diverges(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _determinism as det

        monkeypatch.setattr(det, "_call_opencode", self._answers("15", "17", "15"))
        r = check_determinism_for_qa("Q?", "15", "number", sample_image_chart_path, "key", runs=3)
        assert r["deterministic"] is False
        assert r["diverging"] == ["17"]
        assert [run["match"] for run in r["runs"]] == [True, False, True]

    def test_no_answers_fails_open(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _determinism as det

        monkeypatch.setattr(det, "_call_opencode", self._answers(None, None))
        r = check_determinism_for_qa("Q?", "15", "number", sample_image_chart_path, "key", runs=2)
        assert r["checked"] is False
        assert r["deterministic"] is False

    def test_call_exception_counts_as_no_answer(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _determinism as det

        def boom(*a, **kw):
            raise RuntimeError("network")

        monkeypatch.setattr(det, "_call_opencode", boom)
        r = check_determinism_for_qa("Q?", "15", "number", sample_image_chart_path, "key", runs=2)
        assert r["checked"] is False
        assert r["deterministic"] is False


class TestSemanticPrompt:
    def test_prompt_formats_without_key_error(self):
        """The semantic-equivalence prompt must format cleanly.

        Regression: unescaped JSON braces in the template made .format()
        raise KeyError('\"match\"') on the word-answer semantic fallback,
        crashing determinism checks for word-format tasks.
        """
        from arxiv_manager.authoring.ai_draft._determinism import _VERIFY_SEMANTIC_PROMPT

        prompt = _VERIFY_SEMANTIC_PROMPT.format(a="top left", b="top-left")
        assert "top left" in prompt
        assert "top-left" in prompt
        assert '{"match": true or false' in prompt

    def test_semantic_fallback_used_for_word_mismatch(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _determinism as det

        calls = []
        state = {"n": 0}

        def fake_call(api_key, prompt, b64, **kw):
            calls.append(prompt)
            state["n"] += 1
            if state["n"] == 1:  # the sampled run itself
                return {"answer": "top left", "reasoning": "observed"}
            return {"match": True}  # the semantic-equivalence call

        monkeypatch.setattr(det, "_call_opencode", fake_call)
        r = check_determinism_for_qa("Q?", "top-left", "word", sample_image_chart_path, "key", runs=1)
        assert r["deterministic"] is True
        assert len(calls) == 2
        assert "Answer 1: top left" in calls[1]
