"""Metrics dashboard route handler with cost tracking."""

from __future__ import annotations

import json
import logging
import time as time_module
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

from ...observability.cost_tracker import estimate_cost, format_cost, summarize_usage
from ...storage import STORAGE_DIR
from . import TEMPLATES, router

logger = logging.getLogger(__name__)

_metrics_cache: dict | None = None
_metrics_cache_ts: float = 0


def _read_db_usage() -> dict:
    """Read token usage from GenerationAttempt records in the database.

    Returns aggregated cost data when available.
    """
    try:
        from sqlmodel import select
        from ...db import get_session
        from ...models import GenerationAttempt

        session = get_session()
        rows = list(session.exec(
            select(GenerationAttempt).where(GenerationAttempt.total_tokens > 0)
        ).all())
        session.close()

        if not rows:
            return {"total_calls": 0, "total_cost": 0, "total_cost_str": "$0.00", "by_model": {}}

        records = [
            {
                "model_name": r.model_name,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
            }
            for r in rows
        ]
        return summarize_usage(records)
    except Exception:
        return {"total_calls": 0, "total_cost": 0, "total_cost_str": "$0.00", "by_model": {}}


def _compute_metrics() -> dict:
    """Read telemetry JSONL and return aggregated metrics. Cached for 60s."""
    global _metrics_cache, _metrics_cache_ts

    now = time_module.time()
    if _metrics_cache is not None and now - _metrics_cache_ts < 60:
        return _metrics_cache

    telemetry_path = STORAGE_DIR / "_draft_telemetry.jsonl"
    records: list[dict] = []
    if telemetry_path.exists():
        with open(str(telemetry_path)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    total = len(records)
    ok = sum(1 for r in records if r.get("ok"))
    verify_count = sum(1 for r in records if r.get("difficulty") == "verify")
    draft_count = total - verify_count

    draft_latencies = [r["elapsed_s"] for r in records if r.get("difficulty") != "verify" and r.get("elapsed_s")]
    valid_drafts = [r for r in records if r.get("difficulty") != "verify"]

    avg_lat = round(sum(draft_latencies) / len(draft_latencies), 1) if draft_latencies else 0
    min_lat = round(min(draft_latencies), 1) if draft_latencies else 0
    max_lat = round(max(draft_latencies), 1) if draft_latencies else 0
    sorted_lat = sorted(draft_latencies)
    p50_lat = round(sorted_lat[len(sorted_lat) // 2], 1) if sorted_lat else 0

    by_diff: dict[str, dict] = {}
    for r in valid_drafts:
        diff = r.get("difficulty") or "unknown"
        label = diff or "manual"
        bucket = by_diff.setdefault(label, {"total": 0, "ok": 0})
        bucket["total"] += 1
        if r.get("ok"):
            bucket["ok"] += 1

    by_type: dict[str, dict] = {}
    for r in valid_drafts:
        ft = r.get("figure_type") or "unknown"
        bucket = by_type.setdefault(ft, {"total": 0, "ok": 0})
        bucket["total"] += 1
        if r.get("ok"):
            bucket["ok"] += 1

    recent = records[-24:] if len(records) > 24 else records

    # Cost data from DB
    cost_data = _read_db_usage()

    metrics = {
        "total_drafts": draft_count,
        "total_verify": verify_count,
        "success_rate": round(100 * ok / max(draft_count, 1), 1),
        "avg_latency": avg_lat, "min_latency": min_lat,
        "max_latency": max_lat, "p50_latency": p50_lat,
        "by_difficulty": by_diff,
        "by_figure_type": by_type,
        "recent": [dict(r, **{"es": r.get("elapsed_s", 0), "ok_bool": r.get("ok")}) for r in recent[-24:]],
        "cost": cost_data,
    }

    _metrics_cache = metrics
    _metrics_cache_ts = now
    return metrics


@router.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request):
    """AI draft performance dashboard with cost tracking."""
    logger.info("metrics page")
    metrics = _compute_metrics()
    return TEMPLATES.TemplateResponse(request, "metrics.html", {"m": metrics})
