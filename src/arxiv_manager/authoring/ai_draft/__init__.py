"""AI-assisted Q&A drafting using LLM — split into sub-modules."""
from __future__ import annotations

from ._api_client import _call_opencode, _get_api_key  # noqa: F401
from ._response_parser import (  # noqa: F401
    _extract_reasoning,
    _parse_critique_response,
    _parse_llm_response,
)
from ._verifier import verify_draft  # noqa: F401
from .composition import draft_qa_consensus, draft_with_self_critique  # noqa: F401
from .core import draft_qa  # noqa: F401
