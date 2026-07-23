"""Image analysis orchestrator for the upload-Q&A page.

Bridges existing tools into a single function call for the web UI."""
from __future__ import annotations

import logging
from pathlib import Path

from ..sourcing.filters import audit_figure
from ..authoring.validator import validate_task

logger = logging.getLogger(__name__)


def analyze_uploaded_image(image_path: Path) -> dict:
    """Run all filters on an uploaded image and determine difficulty potential."""
    logger.info("analyze_uploaded_image entry path=%s", image_path)
    audit = audit_figure(image_path)

    if not audit["is_suitable"]:
        reasons = []
        if audit.get("is_text_only"):
            reasons.append("text-only content")
        if audit.get("is_likely_sparse"):
            reasons.append("sparse/low-content")
        if audit.get("is_logo_or_icon"):
            reasons.append("logo/icon")
        if audit["complexity_score"] < 0.3:
            reasons.append(f"low complexity ({audit['complexity_score']:.2f})")
        if audit["width"] < 200 or audit["height"] < 200:
            reasons.append("too small")
        result = {"audit": audit, "suitability": "REJECTED", "suitability_reason": "; ".join(reasons), "figure_type_label": _type_label(audit)}
        logger.info("analyze result REJECTED reason=%s cs=%.3f dims=%dx%d", result["suitability_reason"], audit["complexity_score"], audit["width"], audit["height"])
        return result

    cs = audit["complexity_score"]
    is_chart = _type_label(audit) == "chart"
    hardest_threshold = 0.85 if is_chart else 0.75
    challenging_threshold = 0.55 if is_chart else 0.45
    easy_threshold = 0.35 if is_chart else 0.25

    if cs >= hardest_threshold and audit["is_dense"]:
        likelihood = "HARDEST"
        reason = f"Very high complexity ({cs:.2f}) + dense — strong HARDEST candidate"
    elif cs >= challenging_threshold and audit["is_dense"]:
        likelihood = "CHALLENGING"
        reason = f"Good complexity ({cs:.2f}) + dense — likely CHALLENGING"
    elif cs >= challenging_threshold:
        likelihood = "CHALLENGING"
        reason = f"Moderate complexity ({cs:.2f}) — potential CHALLENGING (not dense)"
    elif cs >= easy_threshold:
        likelihood = "EASY"
        reason = f"Low complexity ({cs:.2f}) — likely too easy for Qwen"
    else:
        likelihood = "EASY"
        reason = f"Very low complexity ({cs:.2f}) — not suitable"

    result = {"audit": audit, "suitability": likelihood, "suitability_reason": reason, "figure_type_label": _type_label(audit)}
    logger.info("analyze result %s cs=%.3f is_chart=%s dense=%s", likelihood, cs, is_chart, audit.get("is_dense"))
    return result


def validate_draft(draft: dict, figure_type: str = "", task_type: str = "") -> dict:
    """Validate a draft and return a template-friendly result."""
    v = validate_task(
        draft.get("question", ""),
        draft.get("answer", ""),
        draft.get("answer_format", "word"),
        figure_type=figure_type,
        task_type=task_type or draft.get("task_type", ""),
    )
    return {
        "is_valid": v.is_valid,
        "errors": v.errors,
        "warnings": v.warnings,
        "passed_checks": v.passed_checks,
        "quality_score": v.quality_score,
        "summary": v.summary(),
    }


def _type_label(audit: dict) -> str:
    ft = audit.get("figure_type", "")
    return "chart" if ft == "chart_graph_text" else "img"
