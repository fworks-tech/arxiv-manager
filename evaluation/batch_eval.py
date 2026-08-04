"""Batch evaluation script — generates questions from images and runs the full gate chain.

Usage:
    python -m evaluation.batch_eval --dir /path/to/images --difficulty challenging
    python -m evaluation.batch_eval --dir /path/to/images --difficulty hardest --max 10

Takes a directory of images, generates questions at the specified difficulty,
runs the full gate chain (validation + fact-check + determinism), and reports
pass/fail rates per strategy class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


def scan_images(directory: str) -> list[Path]:
    """Scan directory for image files."""
    supported = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    images = []
    for p in Path(directory).iterdir():
        if p.is_file() and p.suffix.lower() in supported:
            images.append(p)
    return sorted(images)


def evaluate_single_image(
    image_path: Path,
    difficulty: str,
    api_key: str,
    n_runs: int = 3,
) -> dict[str, Any]:
    """Generate a question for one image and run the full gate chain.

    Returns a result dict with validation, fact-check, determinism outcomes.
    """
    from arxiv_manager.authoring.ai_draft._determinism import check_determinism_for_qa
    from arxiv_manager.authoring.ai_draft._fact_checker import fact_check_draft
    from arxiv_manager.authoring.ai_draft.core import draft_qa
    from arxiv_manager.authoring.validator import validate_task

    result: dict[str, Any] = {
        "image": str(image_path.name),
        "difficulty": difficulty,
        "generated": False,
        "validation_passed": False,
        "fact_check_passed": False,
        "determinism_passed": False,
        "question": "",
        "answer": "",
        "errors": [],
    }

    # Step 1: Generate draft
    start = time.time()
    try:
        draft = draft_qa(
            image_path=str(image_path),
            api_key=api_key,
            difficulty=difficulty,
            figure_type="chart_graph_text",
            complexity_score=0.5,
            previous_question="",
            validation_context="",
        )
    except Exception as exc:
        result["errors"].append(f"generation failed: {exc}")
        return result

    elapsed = time.time() - start
    result["elapsed_s"] = round(elapsed, 1)

    if draft is None:
        result["errors"].append("draft_qa returned None")
        return result

    result["generated"] = True
    result["question"] = draft.get("question", "")
    result["answer"] = draft.get("answer", "")
    result["model"] = draft.get("_model", "unknown")

    # Step 2: Validation
    v = validate_task(
        result["question"],
        result["answer"],
        draft.get("answer_format", "number"),
        figure_type="chart_graph_text",
        task_type=draft.get("task_type", "chart"),
        difficulty=difficulty,
    )
    result["validation_passed"] = v.is_valid
    result["validation_quality"] = round(v.quality_score, 1)
    result["validation_errors"] = v.errors[:3]

    if not v.is_valid:
        result["errors"].append(f"validation failed: {'; '.join(v.errors[:2])}")
        return result

    # Step 3: Fact-check (skip for easy)
    if difficulty != "easy":
        try:
            fc = fact_check_draft(
                result["question"],
                str(image_path),
                api_key,
                difficulty=difficulty,
            )
            result["fact_check_passed"] = fc["verdict"] == "pass"
            if fc["verdict"] == "fail":
                result["errors"].append(f"fact-check failed: {'; '.join(fc['unsupported'][:2])}")
                return result
        except Exception as exc:
            result["errors"].append(f"fact-check error: {exc}")
            result["fact_check_passed"] = True  # fail-open

    # Step 4: Determinism (challenging/hardest only)
    if difficulty in ("challenging", "hardest"):
        try:
            det = check_determinism_for_qa(
                result["question"],
                result["answer"],
                draft.get("answer_format", "number"),
                str(image_path),
                api_key,
                runs=n_runs,
                difficulty=difficulty,
            )
            result["determinism_passed"] = det["deterministic"]
            result["determinism_diverging"] = det.get("diverging", [])
            if not det["deterministic"]:
                result["errors"].append(f"determinism failed: {det['diverging'][:2]}")
                return result
        except Exception as exc:
            result["errors"].append(f"determinism error: {exc}")

    return result


def run_batch_evaluation(
    directory: str,
    difficulty: str = "challenging",
    max_images: int = 0,
    n_runs: int = 3,
) -> dict[str, Any]:
    """Run batch evaluation on a directory of images.

    Returns aggregate metrics.
    """
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        print("Error: OPENCODE_API_KEY not set")
        return {}

    images = scan_images(directory)
    if not images:
        print(f"No images found in {directory}")
        return {}

    if max_images > 0:
        images = images[:max_images]

    print(f"\n=== Batch Evaluation ===")
    print(f"Directory: {directory}")
    print(f"Images: {len(images)}")
    print(f"Difficulty: {difficulty}")
    print(f"Determinism runs: {n_runs}")
    print()

    results = []
    for i, img in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img.name}...", end=" ", flush=True)
        r = evaluate_single_image(img, difficulty, api_key, n_runs)
        results.append(r)
        status = "PASS" if not r["errors"] else "FAIL"
        print(f"{status} ({r.get('elapsed_s', '?')}s)")
        if r["errors"]:
            print(f"  Errors: {'; '.join(r['errors'][:2])}")

    # Aggregate
    total = len(results)
    generated = sum(1 for r in results if r["generated"])
    valid = sum(1 for r in results if r["validation_passed"])
    fc_pass = sum(1 for r in results if r["fact_check_passed"])
    det_pass = sum(1 for r in results if r["determinism_passed"])
    all_pass = sum(1 for r in results if not r["errors"])

    print(f"\n=== Results ===")
    print(f"Generated: {generated}/{total}")
    print(f"Validation passed: {valid}/{total}")
    print(f"Fact-check passed: {fc_pass}/{total}")
    print(f"Determinism passed: {det_pass}/{total}")
    print(f"All gates passed: {all_pass}/{total}")

    # Save results
    output_path = STORAGE_DIR / f"batch_eval_{difficulty}_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump({
            "directory": directory,
            "difficulty": difficulty,
            "n_runs": n_runs,
            "total": total,
            "all_pass": all_pass,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return {
        "total": total,
        "generated": generated,
        "validation_passed": valid,
        "fact_check_passed": fc_pass,
        "determinism_passed": det_pass,
        "all_pass": all_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluation of Q&A generation")
    parser.add_argument("--dir", required=True, help="Directory of images to evaluate")
    parser.add_argument("--difficulty", default="challenging", help="Difficulty level")
    parser.add_argument("--max", type=int, default=0, help="Max images to evaluate (0=all)")
    parser.add_argument("--runs", type=int, default=3, help="Determinism runs")
    args = parser.parse_args()

    run_batch_evaluation(
        directory=args.dir,
        difficulty=args.difficulty,
        max_images=args.max,
        n_runs=args.runs,
    )


if __name__ == "__main__":
    main()
