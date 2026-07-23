"""Tests for AI draft response parser — locks in think-tag handling."""

import json
from arxiv_manager.authoring.ai_draft import _parse_llm_response


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
