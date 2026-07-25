"""Metrics dashboard route handler — full AI ecosystem observability."""

from __future__ import annotations

import json
import logging
import time as time_module
from collections import Counter
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse

from ...observability.cost_tracker import estimate_cost, format_cost, summarize_usage
from ...storage import STORAGE_DIR
from . import TEMPLATES, router

logger = logging.getLogger(__name__)

_metrics_cache: dict | None = None
_metrics_cache_ts: float = 0


def _read_db_metrics() -> dict:
    """Read all metrics from the database."""
    from sqlmodel import select, desc
    from ...db import get_session
    from ...models import GenerationAttempt, Task, IssueReport

    session = get_session()
    try:
        # Generation attempts with quality data
        attempts = list(session.exec(
            select(GenerationAttempt).where(GenerationAttempt.success == True)
            .order_by(desc(GenerationAttempt.created_at))
            .limit(500)
        ).all())

        # Model performance
        model_perf: dict[str, dict] = {}
        for a in attempts:
            m = a.model_name or "unknown"
            if a.validation_quality > 0:
                bucket = model_perf.setdefault(m, {"qualities": [], "latencies": [], "samples": 0})
                bucket["qualities"].append(a.validation_quality)
                if a.elapsed_ms > 0:
                    bucket["latencies"].append(a.elapsed_ms / 1000)
                bucket["samples"] += 1

        model_performance = {}
        for model, data in model_perf.items():
            if data["samples"] >= 1:
                avg_lat = round(sum(data["latencies"]) / len(data["latencies"]), 1) if data["latencies"] else 0
                model_performance[model] = {
                    "avg_quality": round(sum(data["qualities"]) / len(data["qualities"]), 1),
                    "samples": data["samples"],
                    "avg_latency": avg_lat,
                }

        # Quality trend (by chunks of 20 attempts)
        valid_attempts = [a for a in attempts if a.validation_quality > 0]
        quality_trend = []
        chunk_size = 20
        for i in range(0, len(valid_attempts), chunk_size):
            chunk = valid_attempts[i:i + chunk_size]
            if chunk:
                avg_q = round(sum(a.validation_quality for a in chunk) / len(chunk), 1)
                quality_trend.append({
                    "period": f"#{i + 1}-{i + len(chunk)}",
                    "avg": avg_q,
                    "count": len(chunk),
                    "trend": "up" if i > 0 and avg_q > quality_trend[-1]["avg"] else ("down" if i > 0 and avg_q < quality_trend[-1]["avg"] else "stable"),
                })
        quality_trend = quality_trend[-10:] if len(quality_trend) > 10 else quality_trend

        # Common validation errors
        error_counter: Counter = Counter()
        for a in attempts:
            if a.validation_errors and a.validation_errors.strip() not in ("", "[]"):
                try:
                    errs = json.loads(a.validation_errors)
                    for e in errs:
                        error_counter[e[:80]] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        top_errors = [{"error": e, "count": c} for e, c in error_counter.most_common(10)]

        # Rhea stats
        tasks = list(session.exec(select(Task)).all())
        rhea_total = sum(1 for t in tasks if t.rhea_reviewed)
        rhea_passed = sum(1 for t in tasks if t.rhea_passed)

        # Issue reports
        issues = list(session.exec(select(IssueReport)).all())
        issue_reasons = Counter(i.reason for i in issues)
        issue_top_reason = issue_reasons.most_common(1)[0][0] if issue_reasons else ""
        issue_top_reason_count = issue_reasons.most_common(1)[0][1] if issue_reasons else 0

        # Cost data
        cost_data = summarize_usage([
            {
                "model_name": a.model_name,
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "total_tokens": a.total_tokens,
            }
            for a in attempts if a.total_tokens > 0
        ])

        return {
            "model_performance": model_performance,
            "quality_trend": quality_trend,
            "top_errors": top_errors,
            "rhea_total": rhea_total,
            "rhea_passed": rhea_passed,
            "rhea_failed": rhea_total - rhea_passed,
            "issue_total": len(issues),
            "issue_top_reason": issue_top_reason,
            "issue_top_reason_count": issue_top_reason_count,
            "cost": cost_data,
        }
    finally:
        session.close()


def _compute_metrics() -> dict:
    """Read telemetry JSONL + DB and return aggregated metrics. Cached for 60s."""
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
        bucket = by_diff.setdefault(diff, {"total": 0, "ok": 0})
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

    # DB metrics
    db_metrics = _read_db_metrics()

    metrics = {
        "total_drafts": draft_count,
        "total_verify": verify_count,
        "success_rate": round(100 * ok / max(draft_count, 1), 1),
        "avg_latency": avg_lat, "min_latency": min_lat,
        "max_latency": max_lat, "p50_latency": p50_lat,
        "by_difficulty": by_diff,
        "by_figure_type": by_type,
        "recent": [dict(r, **{"es": r.get("elapsed_s", 0), "ok_bool": r.get("ok")}) for r in recent[-24:]],
        **db_metrics,
    }

    _metrics_cache = metrics
    _metrics_cache_ts = now
    return metrics


@router.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request):
    """AI ecosystem dashboard."""
    logger.info("metrics page")
    metrics = _compute_metrics()
    return TEMPLATES.TemplateResponse(request, "metrics.html", {"m": metrics})
