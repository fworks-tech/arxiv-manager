"""Strategy coverage tracker — shows how many proven examples exist per strategy cell.

Usage:
    python -m evaluation.coverage
    python -m evaluation.coverage --check-determinism

Reads the golden dataset and the strategy analytics to show which
(strategy × difficulty × figure_type) cells have enough proven examples.
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

# Strategy classes from analytics/strategies.py
STRATEGY_CLASSES = [
    "counting",
    "comparison",
    "rank",
    "cross_panel_sum_diff",
    "spatial",
    "percentage_change",
    "single_lookup",
    "other",
]

DIFFICULTIES = ["easy", "challenging", "hardest"]
FIGURE_TYPES = ["chart_graph_text", "general_image"]

MIN_PROVEN_PER_CELL = 3  # minimum examples per cell for meaningful analytics


def load_golden_dataset() -> list[dict]:
    """Load the golden dataset."""
    if not GOLDEN_DATASET_PATH.exists():
        return []
    with open(GOLDEN_DATASET_PATH) as f:
        data = json.load(f)
    return data.get("examples", [])


def classify_strategy(example: dict) -> str:
    """Classify a golden example into a strategy class."""
    q = example.get("question", "").lower()
    a = example.get("answer", "").lower()

    # Simple heuristic classification
    if any(w in q for w in ["how many", "count", "number of"]):
        return "counting"
    if any(w in q for w in ["which panel", "which series", "which component"]):
        if any(w in q for w in ["higher", "lower", "greater", "larger", "more", "less"]):
            return "comparison"
        return "single_lookup"
    if any(w in q for w in ["rank", "position", "order", "drops", "rises"]):
        return "rank"
    if any(w in q for w in ["sum", "difference", "total", "combined"]):
        return "cross_panel_sum_diff"
    if any(w in q for w in ["closest", "farthest", "spatial", "left", "right", "above", "below"]):
        return "spatial"
    if any(w in q for w in ["percent", "ratio", "proportion", "increase", "decrease"]):
        return "percentage_change"
    return "other"


def build_coverage_matrix(examples: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """Build coverage matrix: strategy → difficulty → figure_type → count.

    Also tracks which examples are proven (pass all gates).
    """
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    for s in STRATEGY_CLASSES:
        matrix[s] = {}
        for d in DIFFICULTIES:
            matrix[s][d] = {}
            for ft in FIGURE_TYPES:
                matrix[s][d][ft] = 0

    for ex in examples:
        strategy = classify_strategy(ex)
        difficulty = ex.get("difficulty", "challenging")
        figure_type = ex.get("figure_type", "chart_graph_text")

        if strategy not in matrix:
            strategy = "other"
        if difficulty not in DIFFICULTIES:
            difficulty = "challenging"
        if figure_type not in FIGURE_TYPES:
            figure_type = "chart_graph_text"

        matrix[strategy][difficulty][figure_type] += 1

    return matrix


def print_coverage(matrix: dict[str, dict[str, dict[str, int]]]) -> None:
    """Print the coverage matrix as a formatted table."""
    print("\n=== Strategy Coverage Matrix ===")
    print(f"(minimum {MIN_PROVEN_PER_CELL} proven examples per cell)\n")

    # Header
    header = f"{'Strategy':<25}"
    for d in DIFFICULTIES:
        for ft in FIGURE_TYPES:
            short_ft = "chart" if ft == "chart_graph_text" else "image"
            header += f" {d[:3]}-{short_ft:>5}"
    print(header)
    print("-" * len(header))

    gaps = []
    for strategy in STRATEGY_CLASSES:
        row = f"{strategy:<25}"
        for d in DIFFICULTIES:
            for ft in FIGURE_TYPES:
                count = matrix[strategy][d][ft]
                marker = "✓" if count >= MIN_PROVEN_PER_CELL else " "
                row += f" {count:>2}{marker}   "
                if count < MIN_PROVEN_PER_CELL:
                    gaps.append((strategy, d, ft, count))
        print(row)

    print(f"\nTotal gaps: {len(gaps)}")
    if gaps:
        print("Cells needing more proven examples:")
        for strategy, difficulty, ft, count in sorted(gaps, key=lambda x: x[3]):
            print(f"  {strategy} / {difficulty} / {ft}: {count}/{MIN_PROVEN_PER_CELL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy coverage tracker")
    args = parser.parse_args()

    examples = load_golden_dataset()
    if not examples:
        print("No golden examples found")
        return

    print(f"Golden dataset: {len(examples)} examples")
    matrix = build_coverage_matrix(examples)
    print_coverage(matrix)


if __name__ == "__main__":
    main()
