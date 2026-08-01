"""Tests for the adversarial premise fact-checker."""

import json


def _check_result(claims, verdict):
    return {"claims": claims, "verdict": verdict}


class TestParseFactCheck:
    def test_parses_clean_json(self):
        from arxiv_manager.authoring.ai_draft._fact_checker import _parse_fact_check

        text = json.dumps(
            {"claims": [{"claim": "c1", "verdict": "SUPPORTED", "evidence": "e"}], "verdict": "pass"}
        )
        r = _parse_fact_check(text)
        assert r is not None
        assert r["verdict"] == "pass"
        assert r["claims"][0]["verdict"] == "SUPPORTED"

    def test_parses_fenced_json(self):
        from arxiv_manager.authoring.ai_draft._fact_checker import _parse_fact_check

        text = '```json\n{"claims": [], "verdict": "fail"}\n```'
        r = _parse_fact_check(text)
        assert r is not None
        assert r["verdict"] == "fail"

    def test_parses_json_after_prose(self):
        from arxiv_manager.authoring.ai_draft._fact_checker import _parse_fact_check

        text = (
            "Checking the image now... {\"claims\": [{\"claim\": \"x\", \"verdict\": \"NOT_SUPPORTED\", "
            '"evidence": "no"}], "verdict": "fail"} done'
        )
        r = _parse_fact_check(text)
        assert r is not None
        assert r["claims"][0]["verdict"] == "NOT_SUPPORTED"

    def test_none_for_empty(self):
        from arxiv_manager.authoring.ai_draft._fact_checker import _parse_fact_check

        assert _parse_fact_check("") is None
        assert _parse_fact_check(None) is None


class TestFactCheckDraft:
    def test_pass_when_all_supported(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        def fake_call(api_key, prompt, b64, **kw):
            return _check_result(
                [{"claim": "one text overlap panel", "verdict": "SUPPORTED", "evidence": "visible"}],
                "pass",
            )

        monkeypatch.setattr(fc, "_call_opencode", fake_call)
        r = fc.fact_check_draft("Q?", sample_image_chart_path, "key")
        assert r["verdict"] == "pass"
        assert r["unsupported"] == []
        assert r["checked"] is True

    def test_fail_on_not_supported(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        def fake_call(api_key, prompt, b64, **kw):
            return _check_result(
                [{"claim": "two text overlap panels", "verdict": "NOT_SUPPORTED", "evidence": "only one"}],
                "fail",
            )

        monkeypatch.setattr(fc, "_call_opencode", fake_call)
        r = fc.fact_check_draft("Q?", sample_image_chart_path, "key")
        assert r["verdict"] == "fail"
        assert r["unsupported"] == ["two text overlap panels"]

    def test_fail_on_unverifiable(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        def fake_call(api_key, prompt, b64, **kw):
            return _check_result(
                [{"claim": "panel 1 shows 88%", "verdict": "UNVERIFIABLE", "evidence": "text too small"}],
                "fail",
            )

        monkeypatch.setattr(fc, "_call_opencode", fake_call)
        r = fc.fact_check_draft("Q?", sample_image_chart_path, "key")
        assert r["verdict"] == "fail"
        assert "88%" in r["unsupported"][0]

    def test_normalizes_verdicts(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        def fake_call(api_key, prompt, b64, **kw):
            return _check_result(
                [
                    {"claim": "a", "verdict": "not supported", "evidence": ""},
                    {"claim": "b", "verdict": "cannot verify", "evidence": ""},
                ],
                "fail",
            )

        monkeypatch.setattr(fc, "_call_opencode", fake_call)
        r = fc.fact_check_draft("Q?", sample_image_chart_path, "key")
        assert len(r["unsupported"]) == 2

    def test_fail_open_on_call_exception(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        def boom(*a, **kw):
            raise RuntimeError("network down")

        monkeypatch.setattr(fc, "_call_opencode", boom)
        r = fc.fact_check_draft("Q?", sample_image_chart_path, "key")
        assert r["verdict"] == "pass"
        assert r["checked"] is False

    def test_fail_open_on_none_result(self, monkeypatch, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        monkeypatch.setattr(fc, "_call_opencode", lambda *a, **kw: None)
        r = fc.fact_check_draft("Q?", sample_image_chart_path, "key")
        assert r["verdict"] == "pass"
        assert r["checked"] is False

    def test_empty_question_skips(self, sample_image_chart_path):
        from arxiv_manager.authoring.ai_draft import _fact_checker as fc

        r = fc.fact_check_draft("", sample_image_chart_path, "key")
        assert r["verdict"] == "pass"
        assert r["checked"] is False
