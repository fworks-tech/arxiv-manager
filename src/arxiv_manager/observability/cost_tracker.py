"""Cost tracking for LLM API usage.

Maps model names to cost-per-token and provides estimation functions.
Costs are in USD per 1K tokens (input and output).
"""

from __future__ import annotations

from typing import Any

# Cost per 1K tokens in USD (input, output)
# Based on OpenCode Zen pricing as of 2026-07
_COST_TABLE: dict[str, tuple[float, float]] = {
    "minimax-m3": (0.15, 0.60),
    "minimax-m2": (0.15, 0.60),
    "gpt-5": (2.50, 10.00),
    "gpt-5-codex": (2.50, 10.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-2.5-flash": (0.075, 0.30),
    "deepseek-v3": (0.27, 1.10),
    "mimo-v2.5": (0.15, 0.60),
}


def get_cost_per_token(model: str) -> tuple[float, float]:
    """Return (input_cost_per_1k, output_cost_per_1k) for a model.

    Falls back to a reasonable default for unknown models.
    """
    for key, costs in _COST_TABLE.items():
        if key in model.lower():
            return costs
    return (0.15, 0.60)  # default fallback


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a generation.

    Args:
        model: Model name (matched against _COST_TABLE keys).
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.

    Returns:
        Estimated cost in USD.
    """
    cost_in, cost_out = get_cost_per_token(model)
    return (input_tokens / 1000 * cost_in) + (output_tokens / 1000 * cost_out)


def format_cost(cost_usd: float) -> str:
    """Format a USD cost for display."""
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    if cost_usd < 1:
        return f"${cost_usd:.3f}"
    return f"${cost_usd:.2f}"


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token and cost data from a list of attempt records.

    Each record should have: model_name, input_tokens, output_tokens, total_tokens.
    """
    total_input = sum(r.get("input_tokens", 0) for r in records)
    total_output = sum(r.get("output_tokens", 0) for r in records)
    total_cost = 0.0
    by_model: dict[str, dict[str, int | float]] = {}

    for r in records:
        model = r.get("model_name", "unknown")
        inp = r.get("input_tokens", 0)
        out = r.get("output_tokens", 0)
        cost = estimate_cost(model, inp, out)

        bucket = by_model.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
        bucket["calls"] += 1
        bucket["input_tokens"] += inp
        bucket["output_tokens"] += out
        bucket["cost"] += cost
        total_cost += cost

    return {
        "total_calls": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost": round(total_cost, 4),
        "total_cost_str": format_cost(total_cost),
        "by_model": by_model,
    }
