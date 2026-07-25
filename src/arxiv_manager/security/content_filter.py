"""Content filter — detects PII, sensitive data, and toxic content in model outputs.

Provides lightweight filtering without requiring external classification APIs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# PII patterns
_PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"),  # Credit card
]

# Toxicity keywords (very basic — expand with a real model in production)
_TOXIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(hate|kill|stupid|idiot|dumb)\b", re.IGNORECASE),
]


def filter_output(text: str, content_type: str = "qa_pair") -> dict[str, Any]:
    """Check output text for PII, sensitive data, and toxic content.

    Args:
        text: The text to check.
        content_type: What kind of content this is ("qa_pair", "prompt", "caption").

    Returns:
        dict with:
        - safe: bool — True if passes all filters
        - issues: list[str] — what was found
        - redacted: str | None — text with PII redacted (if applicable)
    """
    issues: list[str] = []
    redacted = text

    # Check PII
    for pattern in _PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            issues.append(f"PII detected: {matches[0][:20]}...")
            redacted = pattern.sub("[REDACTED]", redacted)

    # Check toxicity
    for pattern in _TOXIC_PATTERNS:
        if pattern.search(text):
            issues.append(f"Toxic language detected: {pattern.pattern}")
            return {"safe": False, "issues": issues, "redacted": redacted}

    if content_type == "qa_pair" and not issues:
        pass  # Q&A content is expected to be factual

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted if issues else None,
    }
