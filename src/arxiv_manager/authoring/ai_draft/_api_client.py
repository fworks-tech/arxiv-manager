"""API client for the OpenCode inference endpoint."""
from __future__ import annotations

import logging
import os
import time

from .._draft_config import CONFIG
from ._response_parser import _parse_llm_response

logger = logging.getLogger(__name__)


def _get_api_key() -> str | None:
    """Get OpenCode API key from environment."""
    return os.environ.get("OPENCODE_API_KEY")


def _call_opencode(
    api_key: str,
    prompt: str,
    b64_image: str,
    model: str | None = None,
    retries: int | None = None,
    difficulty: str = "",
    media_type: str = "image/jpeg",
    parser=None,
) -> dict | None:
    """Call OpenCode Go API (OpenAI-compatible) with image."""
    import httpx

    model_id = model or CONFIG.default_model
    retries = retries or CONFIG.retries
    is_hard = difficulty in ("hardest", "challenging")
    cfg = CONFIG.get_model_config(model_id)
    max_tokens = cfg.max_tokens_hard if is_hard else cfg.max_tokens_easy
    timeout = cfg.timeout_hard if is_hard else cfg.timeout_easy

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)
        try:
            resp = httpx.post(
                CONFIG.api_url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{b64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.text
            try:
                data = resp.json()
            except Exception:
                logger.warning("_call_opencode: non-json response (len=%d, preview=%.200s)", len(body), body[:200])
                raise ValueError(f"API returned non-JSON response: {body[:200]}")
            if "error" in data and isinstance(data["error"], dict):
                err_msg = data["error"].get("message", "") or str(data["error"])
                logger.warning("_call_opencode: API error: %s", err_msg)
                raise ValueError(err_msg)
            choices = data.get("choices")
            if not choices:
                logger.warning("_call_opencode: no choices in response (preview=%.200s)", str(data)[:200])
                raise ValueError("API returned empty response — model may not support image input")
            msg = choices[0].get("message", {})
            content = msg.get("content") or ""
            if not content.strip():
                logger.warning("_call_opencode: empty content on attempt %d", attempt)
                continue
            if "does not support image" in content.lower() or "cannot read" in content.lower():
                logger.warning("_call_opencode: model does not support image input: %.200s", content[:200])
                raise ValueError("Cannot read image — this model does not support image input")
            parse_fn = parser or _parse_llm_response
            parsed = parse_fn(content, raw_text=content)
            if parsed:
                return parsed
            logger.warning("_call_opencode: parsing returned None (len=%d, preview=%.150s)",
                           len(content), content[:150])
            continue
        except ValueError:
            raise
        except Exception:
            if attempt == retries - 1:
                raise

    return None
