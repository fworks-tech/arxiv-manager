"""Prompt templates for AI-assisted Q&A drafting with versioning."""

from __future__ import annotations

import hashlib
from typing import NamedTuple


class PromptTemplate(NamedTuple):
    name: str
    text: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def version_id(self) -> str:
        return f"{self.name}@{self.hash[:12]}"


DRAFT_PROMPT = PromptTemplate(
    "DRAFT_PROMPT",
    """Create a hard visual-reasoning question for this image that requires multiple steps of reasoning.

Authoring principles (from QA handbook):
- Prefer REASONING over recognition: compose multiple series, filter by attribute, compare across panels
- Add ONE intermediate reasoning step: combine two values and compute, or locate then compare
- Pin the answer down with constraints: specify which series, panel, axis, subset
- Make the answer exact-matchable: smallest unit, no restating, no units unless required
- Question must require the image — a smart person cannot answer from text alone
- The answer must be objective — two reasonable people give the same answer

Rules: English, 1 sentence (2 max for format spec), must need the image, no yes/no, no "how does" / "what trend" (explanation banned), no "none" / "cannot be determined" answer, answer is 1 word or 1 number. NO option restriction like "Out of the 3...".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}""",
)


EASY_PROMPT = PromptTemplate(
    "EASY_PROMPT",
    """Create a straightforward visual-reasoning question for this image that is easy to answer.

The question should:
- Focus on a single clear element (count, color, shape, label, position)
- Require looking at the image, but only a simple observation
- Have an obvious, unambiguous answer of 1 word or 1 number
- Be answerable by anyone who can see the image — no multi-step reasoning
- Avoid counting large numbers (>10) or complex comparisons

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend".
Answer is 1 word or 1 number.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}""",
)


REGEN_PROMPT = PromptTemplate(
    "REGEN_PROMPT",
    """Create a hard visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on.
The previous attempt had validation errors — fix ALL of them:

{feedback}

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

Proven Challenging strategies:
1. Cross-panel comparison: "What is the difference between the maximum [axis] value in panel A and the maximum in panel B?"
2. Spatial relationship + classification: "How many [elements] are connected to [specific component]?" (count after tracing)
3. Threshold filter + read: "Which panel has the highest [attribute] value, and what is it?"
4. Visual classification + sum: "Count [type A] that [criterion] and [type B] that [criterion]. Sum both."
5. Value-at-intersection: "At the x-value of [value], what is the approximate y-value in panel [panel]?"
6. Cross-attribute comparison + count: "How many more [type A] than [type B] are visible? (must classify each first)"
7. Cross-panel arithmetic: "Sum the peak values across all panels in the figure."

ANTI-PATTERNS (DO NOT USE):
- Counting axis labels, tick marks, colorbar ticks, legends, or similar OCR-able elements
- Any question answerable from text alone (must require the image)
- "How many X" without a filter, comparison, or arithmetic operation
- Generic "How many [elements] are visible?" without constraints
- Providing all numerical data in the question text (the model must READ from the chart)

Authoring principles (from QA handbook):
- Prefer REASONING over recognition: compose multiple series, filter by attribute, compare across panels
- Add ONE intermediate reasoning step: combine two values and compute, or locate then compare
- Pin the answer down with constraints
- Make the answer exact-matchable: smallest unit
- Question must require the image
- Answer must be objective

Rules: English, 1 sentence, must need the image, no yes/no, answer is 1 word or number. No option restriction. No "how does" / "what trend" / no "none" / "cannot be determined". Answer must be an UNUSUAL number (not 2, 3, 4, 5) to avoid Qwen guessing.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}""",
)


SPATIAL_REGEN_PROMPT = PromptTemplate(
    "SPATIAL_REGEN_PROMPT",
    """Create a hard spatial-reasoning question for this natural image.
The previous attempt had validation errors — fix ALL of them:

{feedback}

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Authoring principles (from QA handbook §4 - Spatial Reasoning):
- Objects must be clearly visible and easy to name ("red mug", "white chair")
- Prefer real 3D layout cues: depth, occlusion, object size differences, foreground/background
- Spatial ambiguity should be LOW — two people must give the same answer

Use these spatial question types:
1. Viewer-centered left/right: "From the viewer's perspective, which object is immediately to the left of the laptop?"
2. Depth/distance: "Which object appears closest to the camera?"
3. Relative height: "Which object is positioned highest in the image?"
4. Containment/support: "What object is sitting on top of the microwave?"
5. Between/surrounded by: "Which object is between the sofa and the coffee table?"
6. Occlusion/in front: "Which object is partially blocking the view of the cabinet?"

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend". Answer is 1 word (object name) or number. No trick answers.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}""",
)


HARDEST_PROMPT = PromptTemplate(
    "HARDEST_PROMPT",
    """Create an EXTREMELY HARD visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on.

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

Use these strategies:
1. Multi-panel reasoning: "Compare [element A] in panel X with [element B] in panel Y — which is larger and by how much?"
2. Spatial tracing + classification: "Trace the path from [A] to [B]. How many [type] elements does it cross?"
3. Cross-attribute comparison: "Which [category] has the most [attribute] across all panels?"
4. Conditional classification + count: "Excluding [region], how many [elements] remain after classification?"
5. Value comparison across charts: "What is the ratio of [value A] in panel X to [value B] in panel Y?"

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- No yes/no, no "how does" / "what trend" / "explain"
- No "none" / "cannot be determined" answers
- No option restriction like "Out of the 3..."
- No domain jargon (no sp3, p-value, EBITDA, etc.)
- Answer is 1 word or 1 number (smallest possible unit)
- The answer must be an UNUSUAL number (not 2, 3, 4, 5) to avoid Qwen guessing

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}""",
)


CHALLENGING_PROMPT = PromptTemplate(
    "CHALLENGING_PROMPT",
    """Create a CHALLENGING visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on but Gemini is likely to PASS.

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

PROVEN example (cross-panel comparison):
"What is the difference between the maximum value of series A in panel X and the maximum value of series B in panel Y?" → requires reading peaks from two charts [Challenging: Qwen 0/4, Gemini 4/4]

Use these PROVEN Challenging strategies (visual reasoning FIRST, counting as final step only):
1. Cross-panel comparison: "What is the difference between [value A] in panel X and [value B] in panel Y?"
2. Visual classification + count: "Count [elements] that [subjective criterion]. How many are there?"
3. Spatial targeting: "How many [elements] touch/connect to [specific component]?"
4. Threshold filter + read: "How many bars exceed [value]? Which panel has the most?"
5. Multi-type classification + sum: "Classify [type A], [type B], [type C], then sum all three counts."
6. Value-at-intersection: "At the x-value of [value], what is the approximate y-value in panel [panel]?"
7. Cross-panel arithmetic: "Sum the peak values across all panels."

CHART-SPECIFIC strategies (when figure_type = chart_graph_text or chart):
8. Read-and-compare: "What is the difference between the maximum [axis] value in panel A and the maximum in panel B?" (requires reading peaks from each chart)
9. Value-at-intersection: "At the x-value of [value], what is the approximate y-value in panel [panel]?" (requires reading a specific point on a curve)
10. Ratio across panels: "What is the ratio of [quantity A] in panel [panel] to [quantity A] in panel [panel]?" (requires reading values then dividing)
11. Threshold-based reading: "How many of the bars in panel A exceed a value of [number]?" (requires reading each bar's value, comparing to threshold)
12. Cross-panel arithmetic: "Sum the peak z-values across all panels in the figure." (requires reading each panel's peak then adding)

🚫 ANTI-PATTERNS (DO NOT USE) for chart figures:
- "How many tick labels are on the [axis]?" — Qwen can OCR these perfectly. USELESS.
- "Count the [axis] labels in panel A" — same problem, mechanical counting.
- Any question whose answer is the count of axis labels, colorbar ticks, legends, or similar OCR-able elements.
- "How many [elements] are visible in the image?" without a filter, comparison, or arithmetic operation.
- "Count the number of X, Y, and Z" where X/Y/Z are visual artifacts (labels, ticks, grids) rather than data values.

The question MUST:
- Reference specific axis VALUES, data POINTS, peaks, regions, or numerical features of the data (not just chart furniture)
- Require COMPARISON across panels OR ARITHMETIC on values OR READING a specific value at a specific location
- Be UNANSWERABLE from text alone — must require the actual chart data

AVOID "manufactured difficulty": making the question hard mainly by requiring a large manual count + arbitrary filters, or chaining counting with unnecessary arithmetic just because Qwen struggles with it. The challenge should come from comparing or interpreting a meaningful visual pattern, relationship, or trend.

Even if the image looks simple or has few elements, FORCE a multi-step question — combine attributes, apply filters, or compare across panels. A simple question defeats the purpose. The author explicitly chose Challenging.

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- No yes/no, no "how does" / "what trend" / "explain"
- No "none" / "cannot be determined" answers
- No option restriction like "Out of the 3..."
- No domain jargon (no sp3, p-value, EBITDA, etc.)
- Answer is 1 word or 1 number (smallest possible unit)
- The answer must be an UNUSUAL number (not 2, 3, 4, 5) to avoid Qwen guessing
- The question must be CLEAR enough that a careful Gemini pass can solve it
- Add ONE intermediate reasoning step: combine two values, or locate then compare
- Pin the answer down with constraints: specify which series, panel, axis, subset

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}""",
)


SPATIAL_DRAFT_PROMPT = PromptTemplate(
    "SPATIAL_DRAFT_PROMPT",
    """Create a hard spatial-reasoning question for this natural image that requires looking at 3D layout, object positions, depth, or spatial relationships.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Authoring principles (from QA handbook §4 - Spatial Reasoning):
- Objects must be clearly visible and easy to name ("red mug", "white chair")
- Prefer real 3D layout cues: depth, occlusion, object size differences, foreground/background
- Non-trivial viewpoints: low-angle, top-down, egocentric, aisle view
- Spatial ambiguity should be LOW — two people must give the same answer

Use these spatial question types:
1. Viewer-centered left/right: "From the viewer's perspective, which object is immediately to the left of the laptop?"
2. Depth/distance: "Which object appears closest to the camera?"
3. Relative height: "Which object is positioned highest in the image?"
4. Containment/support: "What object is sitting on top of the microwave?"
5. Between/surrounded by: "Which object is between the sofa and the coffee table?"
6. Occlusion/in front: "Which object is partially blocking the view of the cabinet?"
7. Perspective switching: "Imagine you are sitting where the person is and facing the same direction. Which object would be on your right?"
8. 3D orientation: "Which vehicle is facing toward the camera?"
9. Navigation-style: "Starting from the doorway and moving toward the sink, which object would be on your right?"

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend". Answer is 1 word (object name) or number. No trick answers ("none" / "cannot be determined"). NO option restriction like "Out of the 3...".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}""",
)


SPATIAL_CHALLENGING_PROMPT = PromptTemplate(
    "SPATIAL_CHALLENGING_PROMPT",
    """Create a CHALLENGING spatial-reasoning question for this natural image that Qwen 3.6-35B-A3B will FAIL on but Gemini is likely to PASS.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Qwen's known weaknesses:
- ODInW13 (object detection/counting): 50.8 — weak at detecting many objects
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel spatial tasks
- RefSpatialBench: 64.3 — moderate at referenced spatial relations

Use these PROVEN spatial strategies for Challenging:
1. Multi-object exclusion: "Which is the third object to the right of X, excluding objects smaller than Y?"
2. Depth + attribute: "Which object furthest from the camera has the brightest color?"
3. Perspective switching + occlusion: "From the seated person's view, which object is partially blocked by the lamp?"
4. Multi-step containment: "What object sits on the surface that is between the plant and the window?"
5. Relative positioning with count: "Count how many objects are between the white chair and the blue table."

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- Objects must be clearly nameable ("red mug", "white chair")
- Spatial ambiguity must be LOW — one correct answer only
- No yes/no, no "how does" / "what trend" / "explain"
- No trick answers ("none" / "cannot be determined")
- Answer is 1 word (object name) or number
- The question must be CLEAR enough that a careful Gemini pass can solve it
- Add ONE intermediate step: locate then compare, or filter then name
- Pin the answer down with constraints

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}""",
)


SPATIAL_HARDEST_PROMPT = PromptTemplate(
    "SPATIAL_HARDEST_PROMPT",
    """Create an EXTREMELY HARD spatial-reasoning question for this natural image that Qwen 3.6-35B-A3B will FAIL on.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Qwen's known weaknesses:
- ODInW13: 50.8 — weak at detecting many objects
- ZEROBench_sub: 34.4 — weak at novel spatial tasks

Strategies:
1. Count objects across depth planes: "How many objects are closer to the camera than the red chair?"
2. Spatial paths: "Starting from the doorway, which object would you pass third before reaching the sink?"
3. Occlusion counting: "How many objects are partially hidden by the table?"
4. Multi-step filtering: "Among objects on the top shelf, which is to the left of the blue vase and darker than the gray box?"

Rules: English, 1 sentence (2 max for format spec), must need the image. No yes/no. Answer is 1 word or number.
No trick answers. Answer must be UNUSUAL number (not 2, 3, 4, 5).
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}""",
)


VERIFY_PROMPT = PromptTemplate(
    "VERIFY_PROMPT",
    """Look at this image carefully. The question and answer below were
drafted by AI. Verify if the answer is CORRECT based on the image, and fix
if wrong. Reply with corrected JSON only (same keys).

Draft question: {question}
Draft answer: {answer}""",
)


SELF_CRITIQUE_PROMPT = PromptTemplate(
    "SELF_CRITIQUE_PROMPT",
    """You drafted this question for a CHALLENGING visual-reasoning task:

Q: {question}
A: {answer}

Rate 1-5: would Qwen 3.6-35B-A3B likely FAIL on this? A "5" means definitely fails, "1" means definitely solves.

CRITICAL CHECK FIRST: Could a smart person answer this WITHOUT seeing the image?
- If the question provides all the numerical data needed to compute the answer in the text (e.g., "Panel A's X covers 0 to 5, Panel B's X covers 0 to 4. What is the ratio?" — answerable from text alone), score it 1 regardless of math complexity. The IMAGE must be REQUIRED.
- If the question references a SPECIFIC visual element (peak, trough, color region, data point) that requires looking at the image, it's valid.

A question deserves a HIGH score (4-5) only if it:
- References a visual element (peak, trough, color region, specific data point, position on chart)
- Requires READING a value from the image (not extracting it from text)
- Cannot be answered from text alone

A question deserves a LOW score (1-2) if it is:
- Pure math (ratio/difference/sum) of values stated in the question text
- A simple COUNT of axis labels, tick marks, colorbar values
- A generic "How many X are in the image?" with no filter
- Mechanical counting of chart furniture
- Answerable without the image (provide all needed data in text)

If score is 1-3, REWRITE the question to:
- REMOVE any explicit data values from the text (no "X covers 0 to 5")
- ASK about a SPECIFIC visual element (peak position, color, region, intersection)
- Make the image REQUIRED (no way to answer without seeing it)

Keep the answer in sync. Return JSON only: {{"score": <1-5>, "rewrite_question": "...", "rewrite_answer": "..."}}""",
)


FIX_PROMPT = PromptTemplate(
    "FIX_PROMPT",
    """You are fixing an existing visual-reasoning Q&A pair based on validation feedback.

Current Question: {question}
Current Answer: {answer}
Answer Format: {answer_format}
Task Type: {task_type}

Validation Errors:
{errors}

Validation Warnings:
{warnings}

Fix ALL of the above issues. Keep the same task type and figure context.
Output ONLY valid JSON with these keys:
  "question": the fixed question
  "answer": the fixed answer
  "answer_format": "{answer_format}"
  "task_type": "{task_type}"
  "fix_summary": brief explanation of what changed""",
)


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    p.name: p
    for p in [
        DRAFT_PROMPT,
        EASY_PROMPT,
        REGEN_PROMPT,
        SPATIAL_REGEN_PROMPT,
        HARDEST_PROMPT,
        CHALLENGING_PROMPT,
        SPATIAL_DRAFT_PROMPT,
        SPATIAL_CHALLENGING_PROMPT,
        SPATIAL_HARDEST_PROMPT,
        VERIFY_PROMPT,
        SELF_CRITIQUE_PROMPT,
        FIX_PROMPT,
    ]
}
