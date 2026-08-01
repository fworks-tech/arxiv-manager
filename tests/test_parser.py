"""Tests for AI draft response parser — locks in think-tag handling."""

from arxiv_manager.authoring.ai_draft import (
    _extract_reasoning,
    _parse_critique_response,
    _parse_llm_response,
)


def test_parses_clean_json():
    """Plain JSON in content parses without modification."""
    text = '{"question": "What?", "answer": "x", "answer_format": "word", "task_type": "chart"}'
    r = _parse_llm_response(text)
    assert r is not None
    assert r["question"] == "What?"


def test_parses_think_block_prefixed_json():
    """Think block before JSON is stripped, JSON is parsed."""
    text = (
        "<think>Let me analyze this figure carefully. I see 3 boxes.</think>\n"
        '{"question": "How many boxes?", "answer": "3", "answer_format": "number", "task_type": "chart"}'
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "3"
    assert "think" not in r["question"].lower()


def test_parses_think_block_with_newlines():
    """Multi-line think blocks are stripped correctly."""
    text = (
        "<think>\nThe user wants me to count.\n"
        "Looking at the figure...\n"
        "</think>\n"
        '{"question": "Count the items?", "answer": "8", "answer_format": "number", "task_type": "chart"}'
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "8"


def test_parses_json_in_markdown_fences():
    """Markdown code fences are stripped (existing behavior preserved)."""
    text = '```json\n{"question": "q", "answer": "a", "answer_format": "word", "task_type": "chart"}\n```'
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "a"


def test_parses_json_after_think_and_fences():
    """Think + fences together: both stripped in order."""
    text = (
        "<think>Some reasoning.</think>\n"
        "```json\n"
        '{"question": "q", "answer": "42", "answer_format": "number", "task_type": "chart"}\n'
        "```"
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "42"


def test_parses_json_inside_think_block():
    """JSON embedded within a single think block — no closing </think> before it."""
    text = '<think>Analyzing: {"question": "How many?", "answer": "5", "answer_format": "number", "task_type": "chart"} end</think>'
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "5"


def test_returns_none_for_empty_content():
    """Empty content (model returned no usable text) returns None."""
    assert _parse_llm_response("") is None
    assert _parse_llm_response(None) is None


def test_returns_none_for_think_only():
    """Think block with no JSON after returns None."""
    text = "<think>I cannot answer this question based on the image.</think>"
    assert _parse_llm_response(text) is None


def test_returns_none_for_garbage():
    """Random non-JSON text returns None."""
    assert _parse_llm_response("I think the answer might be 42, but I'm not sure.") is None


def test_think_does_not_consume_json_braces():
    """Think block content containing braces doesn't break the regex."""
    text = (
        "<think>The format {q, a} seems right.</think>"
        '{"question": "q", "answer": "a", "answer_format": "word", "task_type": "chart"}'
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "a"


# ─── New lenient parsing (missing optional fields) ──────────────────


def test_parses_partial_json_missing_answer_format():
    """Missing answer_format defaults to 'number'."""
    text = '{"question": "What is X?", "answer": "42", "task_type": "chart"}'
    r = _parse_llm_response(text)
    assert r is not None
    assert r["question"] == "What is X?"
    assert r["answer"] == "42"
    assert r["answer_format"] == "number"
    assert r["task_type"] == "chart"


def test_parses_partial_json_missing_task_type():
    """Missing task_type defaults to 'chart'."""
    text = '{"question": "What is Y?", "answer": "7", "answer_format": "number"}'
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "7"
    assert r["answer_format"] == "number"
    assert r["task_type"] == "chart"


def test_parses_partial_json_only_q_and_a():
    """Only question + answer: both defaults applied."""
    text = '{"question": "Count the bars?", "answer": "5"}'
    r = _parse_llm_response(text)
    assert r is not None
    assert r["question"] == "Count the bars?"
    assert r["answer"] == "5"
    assert r["answer_format"] == "number"
    assert r["task_type"] == "chart"


# ─── _extract_reasoning ──────────────────────────────────────────────


def test_extract_reasoning_think_block():
    """<think> block is extracted and removed from text."""
    from arxiv_manager.authoring.ai_draft import _extract_reasoning

    cleaned, reasoning = _extract_reasoning('<think>I need to count the bars.</think>{"question": "Q?", "answer": "3"}')
    assert reasoning == "I need to count the bars."
    assert "<think>" not in cleaned
    assert "Q?" in cleaned


def test_extract_reasoning_no_think():
    """No <think> block returns empty reasoning."""
    cleaned, reasoning = _extract_reasoning('{"question": "Q?", "answer": "3"}')
    assert reasoning == ""
    assert cleaned == '{"question": "Q?", "answer": "3"}'


def test_extract_reasoning_multiple_think_blocks():
    """Multiple <think> blocks are extracted and concatenated."""
    cleaned, reasoning = _extract_reasoning('<think>First thought.</think>{"q":1}<think>Second thought.</think>')
    assert "First thought." in reasoning
    assert "Second thought." in reasoning


def test_extract_reasoning_empty_input():
    """Empty string returns empty tuple."""
    cleaned, reasoning = _extract_reasoning("")
    assert reasoning == ""
    assert cleaned == ""


# ─── _parse_critique_response ────────────────────────────────────────


def test_parse_critique_valid():
    """Valid critique response with score and rewrite."""
    text = '{"score": 2, "rewrite_question": "Harder Q?", "rewrite_answer": "7"}'
    r = _parse_critique_response(text)
    assert r is not None
    assert r["score"] == 2
    assert r["rewrite_question"] == "Harder Q?"
    assert r["rewrite_answer"] == "7"


def test_parse_critique_missing_score():
    """Missing score key returns None."""
    text = '{"rewrite_question": "Q?", "rewrite_answer": "A"}'
    r = _parse_critique_response(text)
    assert r is None


def test_parse_critique_empty():
    """Empty input returns None."""
    r = _parse_critique_response("")
    assert r is None


def test_parse_critique_with_think():
    """Think block stripped before parsing critique."""
    text = '<think>This is easy.</think>{"score": 4, "rewrite_question": "", "rewrite_answer": ""}'
    r = _parse_critique_response(text)
    assert r is not None
    assert r["score"] == 4
    assert "_reasoning_trace" in r
    assert r["_reasoning_trace"] == "This is easy."


# ─── _parse_llm_response with raw_text ───────────────────────────────


def test_parse_llm_response_returns_raw_response_key():
    """_raw_response key is present in parsed result."""
    text = '{"question": "Q?", "answer": "A", "answer_format": "word", "task_type": "chart"}'
    r = _parse_llm_response(text, raw_text=text)
    assert r is not None
    assert r["_raw_response"] == text


def test_parse_llm_response_returns_reasoning_trace():
    """_reasoning_trace key is present when think block exists."""
    text = '<think>Count the bars.</think>{"question": "Q?", "answer": "3", "answer_format": "number", "task_type": "chart"}'
    r = _parse_llm_response(text, raw_text=text)
    assert r is not None
    assert r["_reasoning_trace"] == "Count the bars."
    assert r["answer"] == "3"


# ─── Brace-scan regression tests (JSON after brace-containing prose) ──


def test_parses_json_after_prose_with_braces():
    """JSON appearing after prose containing braces is still found."""
    text = (
        "Let me think about the range {0..1} and the subset {a, b} here.\n"
        '{"question": "What is the ratio?", "answer": "0.5", "answer_format": "number", "task_type": "chart"}'
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["question"] == "What is the ratio?"
    assert r["answer"] == "0.5"


def test_parses_json_after_nonmatching_earlier_object():
    """A balanced object without the key earlier in text doesn't block the scan."""
    text = (
        '{"confidence": 0.9, "labels": ["a", "b"]}\n'
        '{"question": "Count them?", "answer": "2", "answer_format": "number", "task_type": "chart"}'
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["question"] == "Count them?"
    assert r["answer"] == "2"


def test_parses_json_after_nested_braces_in_prose():
    """Prose with nested brace pairs before the JSON is handled."""
    text = (
        "The mapping {{a: 1}, {b: 2}} was observed.\n"
        '{"question": "Which is larger?", "answer": "b", "answer_format": "word", "task_type": "chart"}'
    )
    r = _parse_llm_response(text)
    assert r is not None
    assert r["answer"] == "b"


def test_extract_json_after_braced_prose_without_think():
    """_extract_json_from_text directly skips brace-containing prose."""
    from arxiv_manager.authoring.ai_draft._response_parser import _extract_json_from_text

    text = 'ranges {0..5} and sets {x, y} then {"question": "Q?", "answer": "9", "answer_format": "number", "task_type": "chart"}'
    found = _extract_json_from_text(text, '"question"')
    assert found is not None
    assert found["answer"] == "9"
