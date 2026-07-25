"""Output filter — prevents system prompt leakage and internal data exposure.

Ensures model outputs don't reveal system prompts, configuration,
or internal state information.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that suggest prompt leakage
_LEAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"(you\s+are\s+(an?\s+)?(ai|assistant|expert|model))", re.IGNORECASE),
    re.compile(r"(as\s+(an?\s+)?(ai|assistant|model|system))", re.IGNORECASE),
    re.compile(r"(your\s+(task|job|purpose|role|responsibility|goal)\s+is)", re.IGNORECASE),
    re.compile(r"(i\s+(am|cannot|must|will|shall)\s+(not\s+)?(follow|obey|comply))", re.IGNORECASE),
    re.compile(r"(the\s+(system|instruction|prompt|guideline)\s+(says|states|requires))", re.IGNORECASE),
    re.compile(r"(here\s+(are|is)\s+(my|the)\s+(prompt|instruction|guideline))", re.IGNORECASE),
]


def check_leakage(text: str) -> dict[str, Any]:
    """Check if the model output contains leaked system prompt content.

    Returns:
        dict with:
        - leaked: bool — True if leakage detected
        - matches: list[str] — what patterns matched
    """
    matches: list[str] = []
    for pattern in _LEAK_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern[:40])

    if matches:
        logger.warning("output_filter: prompt leakage detected (%d patterns)", len(matches))

    return {
        "leaked": len(matches) > 0,
        "matches": matches,
    }


def sanitize_output(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Sanitize a generation result dict by removing internal fields.

    Strips all keys prefixed with '_' (internal metadata) before
    returning the result to the user.
    """
    return {k: v for k, v in result.items() if not k.startswith("_")}
