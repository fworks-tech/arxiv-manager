"""Verified capability cards for the two models Realm evaluates tasks against.

Sources (fetched 2026-08-01):
- Qwen3.6-35B-A3B: official Hugging Face model card
  (https://huggingface.co/Qwen/Qwen3.6-35B-A3B, Apr 2026)
- Gemini 3.5 Flash: OpenRouter model entry + official Google docs
  (google/gemini-3.5-flash, May 2026)

Gemini 3.5 Flash has no public vision benchmark numbers yet — its vision
profile is estimated from positioning ("near-Pro coding and reasoning")
and must be treated as an assumption until rollout data exists.
"""

from __future__ import annotations

MODEL_CARDS: dict[str, dict] = {
    "qwen3.6-35b-a3b": {
        "id": "openrouter/qwen/qwen3.6-35b-a3b",
        "name": "Qwen3.6 35B A3B",
        "profile": "open-weight 35B/3B-active MoE VLM, thinking optional",
        "vision_benchmarks": {
            "ODInW13 (in-the-wild object detection/counting)": 50.8,
            "ZEROBench_sub (novel zero-shot formats)": 34.4,
            "RefSpatialBench (fine-grained spatial referencing)": 64.3,
            "MMMU (multimodal understanding)": 81.7,
            "MathVista-mini": 86.4,
            "RealWorldQA": 85.3,
            "HallusionBench (hallucination resistance)": 69.8,
            "OmniDocBench 1.5 (OCR/document reading)": 89.9,
            "CC-OCR (text recognition)": 81.9,
            "AI2D (diagram understanding)": 92.7,
            "CharXiv (chart reasoning)": 78.0,
        },
        "strengths": [
            "excellent OCR and document/chart reading (89-93)",
            "strong diagram and chart understanding",
            "strong mathematical visual reasoning (MathVista 86)",
        ],
        "weaknesses": [
            "counting objects in the wild (ODInW13 50.8)",
            "novel/unfamiliar task formats (ZEROBench 34.4)",
            "fine-grained spatial references (RefSpatialBench 64.3)",
        ],
    },
    "gemini-3.5-flash": {
        "id": "openrouter/google/gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "profile": "Google high-efficiency VLM, near-Pro reasoning, thinking mandatory (default medium)",
        "vision_benchmarks": {},  # no public vision numbers yet (2026-08)
        "strengths": [
            "near-Pro-level reasoning at Flash cost (AI index 50.2 vs Qwen 31.6)",
            "excellent coding/agentic intelligence (design arena ELO 1187-1317)",
            "mandatory thinking with tunable effort — reasons carefully by default",
        ],
        "weaknesses": [
            "assumed: struggles with the same perception-level tasks as most VLMs "
            "(counting small/dense elements, fine spatial distinctions) — unverified, needs rollout data",
            "knowledge cutoff Jan 2025 (irrelevant for image tasks)",
        ],
    },
}

# Strategy guidance derived from the cards: what is safe to ask for HARDEST.
# Both models must fail, so avoid anything Gemini's strong reasoning + Qwen's
# strong OCR/diagram reading can solve.
BOTH_FAIL_AVOID = [
    "pure OCR or text extraction (Qwen scores 82-90; Gemini near-Pro reading)",
    "single-value chart reads and simple chart arithmetic (both strong)",
    "text-only reasoning or math on stated values (Gemini strong)",
    "diagram traversal with clear labels (Qwen AI2D 92.7)",
]
BOTH_FAIL_PREFER = [
    "counting small/dense elements in the wild (Qwen ODInW13 50.8; Gemini unverified)",
    "fine-grained spatial distinctions (Qwen RefSpatialBench 64.3)",
    "novel multi-step task formats (Qwen ZEROBench 34.4)",
    "cross-panel arithmetic on many unlabeled visual elements",
]
