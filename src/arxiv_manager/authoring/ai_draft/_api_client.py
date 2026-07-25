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


def _call_api_with_retry(
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float,
    retries: int = 3,
    parser=None,
    method: str = "POST",
) -> dict | None:
    """POST JSON to an API endpoint with retry, backoff, and response parsing.

    Returns parsed dict on success, None after all retries exhausted.
    Raises ValueError on API errors (non-JSON response, error objects, etc.).
    """
    import httpx

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2 ** attempt)
        try:
            if method == "POST":
                resp = httpx.post(url, headers=headers, json=json_body, timeout=timeout)
            else:
                resp = httpx.get(url, headers=headers, params=json_body, timeout=timeout)
            resp.raise_for_status()
            body = resp.text
            try:
                data = resp.json()
            except Exception:
                logger.warning("_call_api_with_retry: non-json response (len=%d, preview=%.200s)", len(body), body[:200])
                raise ValueError(f"API returned non-JSON response: {body[:200]}")
            if "error" in data and isinstance(data["error"], dict):
                err_msg = data["error"].get("message", "") or str(data["error"])
                logger.warning("_call_api_with_retry: API error: %s", err_msg)
                raise ValueError(err_msg)
            if "choices" in json_body:
                # OpenAI-compatible chat completion response format
                choices = data.get("choices")
                if not choices:
                    logger.warning("_call_api_with_retry: no choices (preview=%.200s)", str(data)[:200])
                    raise ValueError("API returned empty response — model may not support image input")
                msg = choices[0].get("message", {})
                content = msg.get("content") or ""
                if not content.strip():
                    logger.warning("_call_api_with_retry: empty content on attempt %d", attempt)
                    continue
                if "does not support image" in content.lower() or "cannot read" in content.lower():
                    logger.warning("_call_api_with_retry: model does not support image input: %.200s", content[:200])
                    raise ValueError("Cannot read image — this model does not support image input")
                parse_fn = parser or _parse_llm_response
                parsed = parse_fn(content, raw_text=content)
                if parsed:
                    usage = data.get("usage", {})
                    if usage:
                        parsed["_usage"] = {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                    return parsed
                logger.warning("_call_api_with_retry: parsing returned None (len=%d, preview=%.150s)",
                               len(content), content[:150])
                continue
            return data
        except ValueError:
            raise
        except Exception:
            if attempt == retries - 1:
                raise

    return None


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
    """Call OpenCode Go API (OpenAI-compatible) with image.

    Builds the request payload and delegates to _call_api_with_retry.
    """
    model_id = model or CONFIG.default_model
    retries = retries or CONFIG.retries
    is_hard = difficulty in ("hardest", "challenging")
    cfg = CONFIG.get_model_config(model_id)
    max_tokens = cfg.max_tokens_hard if is_hard else cfg.max_tokens_easy
    timeout = cfg.timeout_hard if is_hard else cfg.timeout_easy

    return _call_api_with_retry(
        url=CONFIG.api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json_body={
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
        retries=retries,
        parser=parser,
    )
