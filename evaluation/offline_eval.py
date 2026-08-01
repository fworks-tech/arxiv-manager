"""Offline evaluation script for the AI drafting pipeline.

Usage:
    python -m evaluation.offline_eval --runs 3
    python -m evaluation.offline_eval --quick          # 1 determinism run per example
    python -m evaluation.offline_eval --determinism    # run VLM determinism checks (needs OPENCODE_API_KEY)

Loads the golden dataset and reports per-example:
- handbook validator result (errors/warnings)
- determinism: N sampled minimax-m3 reads must all match the golden answer
  (only for entries with an image_path, only with --determinism)
- known anti-pattern flags recorded in the dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load the golden dataset from JSON."""
    if not GOLDEN_DATASET_PATH.exists():
        print(f"Error: golden dataset not found at {GOLDEN_DATASET_PATH}")
        sys.exit(1)
    with open(GOLDEN_DATASET_PATH) as f:
        data = json.load(f)
    return data["examples"]


def run_evaluation(
    n_runs: int = 3,
    difficulty_filter: str | None = None,
    check_determinism: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run evaluation on the golden dataset.

    Args:
        n_runs: Number of sampled determinism reads per example.
        difficulty_filter: If set, only evaluate examples of this difficulty.
        check_determinism: Run VLM determinism checks (requires OPENCODE_API_KEY).
        verbose: Print per-example results.

    Returns:
        Aggregate metrics dict.
    """
    from arxiv_manager.authoring.validator import validate_task

    examples = load_golden_dataset()
    if difficulty_filter:
        examples = [ex for ex in examples if ex["difficulty"] == difficulty_filter]

    if not examples:
        print("No examples to evaluate.")
        return {}

    api_key = os.environ.get("OPENCODE_API_KEY") if check_determinism else None
    if check_determinism and not api_key:
        print("Error: OPENCODE_API_KEY not set (required for --determinism)")
        return {}

    results: list[dict] = []
    for ex in examples:
        entry: dict[str, Any] = {
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "figure_type": ex["figure_type"],
            "expected_answer": ex["answer"],
            "known_issues": ex.get("known_issues", []),
        }

        image_path = ex.get("image_path") or ""
        if image_path:
            resolved = STORAGE_DIR / image_path
            entry["image_resolved"] = resolved.exists()

        validation = validate_task(
            ex["question"],
            ex["answer"],
            ex.get("answer_format", "word"),
            figure_type=ex.get("figure_type", ""),
            task_type=ex.get("task_type", "chart"),
            difficulty=ex.get("difficulty"),
            image_path=str(STORAGE_DIR / image_path) if image_path else "",
        )
        entry["validation_is_valid"] = validation.is_valid
        entry["validation_quality"] = round(validation.quality_score, 1)
        entry["validation_errors"] = validation.errors[:3]
        entry["validation_warnings"] = validation.warnings[:3]

        if check_determinism and image_path:
            from arxiv_manager.authoring.ai_draft._determinism import check_determinism_for_qa

            resolved = STORAGE_DIR / image_path
            if resolved.exists():
                det = check_determinism_for_qa(
                    ex["question"],
                    ex["answer"],
                    ex.get("answer_format", "word"),
                    resolved,
                    api_key,
                    runs=n_runs,
                    difficulty=ex.get("difficulty", "challenging"),
                )
                entry["determinism_checked"] = det["checked"]
                entry["determinism_pass"] = det["deterministic"]
                entry["diverging"] = det["diverging"]
            else:
                entry["determinism_checked"] = False
                entry["determinism_note"] = f"image not found: {image_path}"

        results.append(entry)

    # Aggregate
    total = len(examples)
    valid = sum(1 for r in results if r["validation_is_valid"])
    det_checked = sum(1 for r in results if r.get("determinism_checked"))
    det_pass = sum(1 for r in results if r.get("determinism_pass"))

    print(f"\n=== Offline Evaluation Report ===")
    print(f"Golden dataset: {total} examples")
    print(f"Handbook-valid: {valid}/{total}")
    if check_determinism:
        print(f"Determinism pass: {det_pass}/{det_checked} checked")
    print()
    for r in results:
        mark = "✅" if r["validation_is_valid"] else "❌"
        det_mark = ""
        if r.get("determinism_checked"):
            det_mark = " | determinism: " + ("PASS" if r["determinism_pass"] else f"FAIL {r.get('diverging')}")
        issues = f" | KNOWN: {r['known_issues']}" if r.get("known_issues") else ""
        print(
            f"  {mark} {r['id']} [{r['difficulty']}] q={r['validation_quality']}"
            f" errs={r['validation_errors']}{det_mark}{issues}"
        )

    return {
        "total_examples": total,
        "handbook_valid": valid,
        "determinism_checked": det_checked,
        "determinism_pass": det_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation of the AI drafting pipeline")
    parser.add_argument("--runs", type=int, default=3, help="Number of determinism reads per example")
    parser.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty")
    parser.add_argument("--quick", action="store_true", help="Quick run (1 determinism read per example)")
    parser.add_argument("--determinism", action="store_true", help="Run VLM determinism checks (costs API calls)")
    args = parser.parse_args()

    n_runs = 1 if args.quick else args.runs
    result = run_evaluation(
        n_runs=n_runs,
        difficulty_filter=args.difficulty,
        check_determinism=args.determinism,
        verbose=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
