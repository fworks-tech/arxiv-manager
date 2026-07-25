"""AI-assisted Q&A drafting using LLM — split into sub-modules."""
from __future__ import annotations

from .core import draft_qa  # noqa: F401
from .composition import draft_qa_consensus, draft_with_self_critique  # noqa: F401
from ._verifier import verify_draft  # noqa: F401
from ._api_client import _get_api_key, _call_opencode  # noqa: F401
from ._response_parser import (  # noqa: F401
    _parse_llm_response,
    _parse_critique_response,
    _extract_reasoning,
)
