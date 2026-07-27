"""Input guard — detects prompt injection attempts and suspicious inputs.

Uses pattern matching and lightweight heuristics to flag potential
prompt injection, jailbreak attempts, and anomalous inputs before
they reach the generation pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns suggesting prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|directions)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)\s+(previous|prior|training)", re.IGNORECASE),
    re.compile(
        r"you\s+(are\s+)?(now|must\s+act\s+as)\s+(\w+\s+){0,5}(free|unconstrained|unrestricted|dan)", re.IGNORECASE
    ),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"print\s+(your|the)\s+(prompt|instructions|system)", re.IGNORECASE),
    re.compile(r"output\s+(your|the)\s+(prompt|instructions|system)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(prompt|instructions|system)", re.IGNORECASE),
    re.compile(
        r"how\s+(can\s+)?(i\s+)?(bypass|override|break)\s+(the\s+)?(rules|restrictions|guardrails)", re.IGNORECASE
    ),
    re.compile(r"role\s*:\s*system", re.IGNORECASE),
]

_SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"OPENCODE_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY", re.IGNORECASE),
    re.compile(r"(api[_-]?key|secret[_-]?key|password|auth[_-]?token)\s*[:=]\s*\S{8,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\b"),  # Potential API keys (20+ alphanumeric + underscore)
]


def check_input(text: str) -> dict[str, Any]:
    """Check input text for prompt injection and sensitive data.

    Returns:
        dict with:
        - safe: bool — True if input passes all checks
        - reasons: list[str] — reasons if flagged
        - risk: str — "safe" | "suspicious" | "blocked"
    """
    reasons: list[str] = []

    # Check for injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            reasons.append(f"Prompt injection pattern detected: {pattern.pattern[:50]}")
            return {"safe": False, "reasons": reasons[:3], "risk": "blocked"}

    # Check for sensitive data
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(text):
            match = pattern.search(text)
            reasons.append(f"Sensitive data pattern detected: {match.group()[:30]}")
            return {"safe": False, "reasons": reasons[:3], "risk": "blocked"}

    # Check for suspicious length
    if len(text) > 5000:
        reasons.append(f"Input exceeds typical length ({len(text)} chars)")

    risk = "suspicious" if reasons else "safe"
    return {"safe": not bool(reasons), "reasons": reasons, "risk": risk}
