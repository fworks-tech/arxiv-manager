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


_STRICT_JSON = "Do NOT use <think> tags. Output ONLY valid JSON. No explanation before or after.\n"


def _with_strict(prompt: str) -> str:
    """Prepend strict JSON instruction to a prompt."""
    return _STRICT_JSON + prompt


DRAFT_PROMPT = PromptTemplate(
    "DRAFT_PROMPT",
    _with_strict("""Create a hard visual-reasoning question for this image that requires multiple steps of reasoning.

Authoring principles (from QA handbook):
- Prefer REASONING over recognition: compose multiple series, filter by attribute, compare across panels
- Add ONE intermediate reasoning step: locate then compare across panels, or filter then rank
- Pin the answer down with constraints: specify which series, panel, axis, subset
- Make the answer exact-matchable: smallest unit, no restating, no units unless required
- Question must require the image — a smart person cannot answer from text alone
- The answer must be objective — two reasonable people give the same answer
- Do NOT ask for a specific y-axis value that requires tracing from a curve point to the axis (ambiguous answers)
- Prefer comparison/ranking: "which is higher/larger/steeper?" over "what is the value at...?"

Rules: English, 1 sentence (2 max for format spec), must need the image, no yes/no, no "how does" / "what trend" (explanation banned), no "none" / "cannot be determined" answer, answer is 1 word or 1 number. NO option restriction like "Out of the 3...".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""),
)


EASY_PROMPT = PromptTemplate(
    "EASY_PROMPT",
    _with_strict("""Create a straightforward visual-reasoning question for this image that is easy to answer.

The question should:
- Focus on a single clear element (count, color, shape, label, position)
- Require looking at the image, but only a simple observation
- Have an obvious, unambiguous answer of 1 word or 1 number
- Be answerable by anyone who can see the image — no multi-step reasoning
- Avoid counting large numbers (>10) or complex comparisons

Rules: English, 1 sentence, must need the image, no yes/no, no "how does" / "what trend".
Answer is 1 word or 1 number.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""),
)


REGEN_PROMPT = PromptTemplate(
    "REGEN_PROMPT",
    _with_strict("""Create a hard visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on.
The previous attempt had validation errors — fix ALL of them:

{feedback}

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

Proven Challenging strategies (genuine visual REASONING, not mechanical tasks):
1. Cross-panel comparison: "Which panel has the largest value at x=[value]?"
2. Which-is-higher: "At x=[value], which series has the greater y-value: [series A] or [series B]?"
3. Peak comparison: "Which panel has the highest peak value?"
4. Rank change across conditions: "Determine [entity]'s rank among [list all competitors] at [condition A] and at [condition B], then calculate how many positions it drops or rises." — Answer is a small integer.
5. Trend-based comparison: "Between x=[a] and x=[b], which panel shows the steepest increase?"
6. Slope direction: "In panel X, does the curve trend upward or downward between x=[a] and x=[b]?"
7. Visual flow tracing: "Trace the path from [component A] to [component B]. Which intermediate component receives inputs from both?"
8. Attribute comparison: "Which component has the larger value: the one labeled [X] or the one labeled [Y]?"

🚫 DO NOT ask for a specific numeric value that requires tracing from a curve point to the y-axis (e.g., "What is the y-value at x=5?" or "What is the peak value?"). Different readers will get different answers.

🚫 ANTI-PATTERNS (DO NOT USE — these are mechanical, not reasoning):
- "Count the number of X, Y, and Z" where the primary effort is tallying many items — not visual reasoning
- "Classify [A], [B], [C], then sum all counts" — counting with labels, not reasoning
- "How many arrows/connections touch [component]?" — counting instead of interpreting
- Counting chart furniture: axis labels, tick marks, colorbar ticks, legends — OCR-able, useless
- Chaining multiple independent counts with addition/multiplication — manufactured difficulty
- Any question answerable from text alone (must require the image)
- Providing all numerical data in the question text (the model must READ from the chart)
- Asking for a specific y-axis value that requires tracing from a data point to the axis

Authoring principles (from QA handbook):
- Prefer REASONING over recognition: interpret visual patterns, compare values, read specific data points
- Add ONE intermediate reasoning step: combine two values, or locate then compare
- Pin the answer down with constraints
- Make the answer exact-matchable: smallest unit
- Question must require the image
- Answer must be objective

Rules: English, 1 sentence, must need the image, no yes/no, answer is 1 word or number. No option restriction. No "how does" / "what trend" / no "none" / "cannot be determined".
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""),
)


SPATIAL_REGEN_PROMPT = PromptTemplate(
    "SPATIAL_REGEN_PROMPT",
    _with_strict("""Create a hard spatial-reasoning question for this natural image.
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
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""),
)


HARDEST_PROMPT = PromptTemplate(
    "HARDEST_PROMPT",
    _with_strict("""Create an EXTREMELY HARD visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on.

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

Use these strategies (genuine visual REASONING, not mechanical tasks):
1. Multi-panel reasoning: "Compare [element A] in panel X with [element B] in panel Y — which is larger and by how much?"
2. Which-is-higher: "At x=[value], which series has the greater y-value in panel X: [A] or [B]?"
3. Cross-attribute comparison: "Which [category] has the highest [attribute] across all panels?"
4. Peak ranking: "Rank the panels from highest to lowest peak value."
5. Trend comparison across charts: "Between x=[a] and x=[b], which panel's curve increases then decreases?"
6. Visual flow tracing: "Trace the path from [component A] to [component B]. What component sits at the junction?"
7. Attribute comparison: "Which connected component has the largest [attribute]: [X] or [Y]?"

🚫 DO NOT make counting the primary difficulty. The challenge MUST come from interpreting visual patterns and relationships — NOT from tallying items or reading specific axis values that require tracing. Do NOT ask for a specific numeric y-axis value — different readers get different answers.

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- No yes/no, no "how does" / "what trend" / "explain"
- No "none" / "cannot be determined" answers
- No option restriction like "Out of the 3..."
- No domain jargon (no sp3, p-value, EBITDA, etc.)
- Answer is 1 word or 1 number (smallest possible unit)

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""),
)


CHALLENGING_PROMPT = PromptTemplate(
    "CHALLENGING_PROMPT",
    _with_strict("""Create a CHALLENGING visual-reasoning question for this image that Qwen 3.6-35B-A3B will FAIL on but Gemini is likely to PASS.

Qwen's known weaknesses (from benchmarks):
- ODInW13 (object detection/counting): 50.8 — weak at counting many visual elements
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel task formats

WORKFLOW — choose the best pattern for THIS image:
Step 1: ANALYZE the image. Identify its structure: Does it have multiple panels? Grouped bars? Multiple data series? A time axis? A flow/diagram layout? Discrete categories? State what you see.
Step 2: MATCH to a strategy. From the proven strategies below, pick the ONE that best fits the image structure. Explain why this pattern is a natural fit (e.g., "This is a grouped bar chart with 4 categories and 3 conditions → Rank change across conditions applies perfectly").
Step 3: GENERATE the question using ONLY that chosen strategy. Do not mix strategies. Do not default to counting. Pin every answer to specific named entities in the image.

PROVEN example (rank change across conditions):
"Determine [entity]'s rank among [list competitors] at [condition A] and at [condition B], then calculate how many positions it drops." → requires ranking bars by height at two x-values, then computing the ordinal delta. Answer is a small integer (e.g., 2). [Challenging: Qwen fails, Gemini passes]

Use these PROVEN Challenging strategies (genuine visual REASONING, not mechanical tasks):
1. Cross-panel comparison: "Which panel has the largest value at x=[value]?"
2. Which-is-higher: "At x=[value], which series has the greater y-value: [series A] or [series B]?"
3. Peak comparison: "Which panel has the highest peak value?"
4. Rank change across conditions: "Determine [entity]'s rank among [list all competitors] at [condition A] and at [condition B], then calculate how many positions it drops or rises." — Answer is always a small integer (rank delta).
5. Trend-based comparison: "Between x=[a] and x=[b], which panel shows the steepest increase?"
6. Slope direction: "In panel X, does the curve trend upward or downward between x=[a] and x=[b]?"
7. Visual flow tracing: "Trace the path from [component A] to [component B]. Which intermediate component receives inputs from both?"
8. Attribute comparison: "Which component has the larger value: the one labeled [X] or the one labeled [Y]?"

CHART-SPECIFIC strategies (for chart_graph_text or chart figures):
9. Which-is-higher-at-intersection: "At x=[value], which series has the greater y-value: [A] or [B]?"
10. Peak-ranking: "Which panel has the second-highest peak among all panels?"
11. Trend-direction: "In panel X, does the blue curve go up or down as it crosses x=[value]?"

How to match image features → best strategy:
| Image has... | Best fit strategy (#) |
|---|---|
| Grouped/stacked bar chart, discrete categories, multiple conditions | 4 (Rank change) |
| Multiple panels (subplots) with comparable data | 1 (Cross-panel), 3 (Peak comparison), 10 (Peak-ranking) |
| Single panel, multiple data series, shared x-axis | 2 (Which-is-higher), 9 (At-intersection), 5 (Trend-based) |
| Time series with clear slope/flections | 6 (Slope direction), 11 (Trend-direction) |
| Flow diagram / network / tree with labeled nodes | 7 (Flow tracing) |
| Components with labeled values / annotations | 8 (Attribute comparison) |
| Natural image / photo (no chart) | Spatial strategies only (SPATIAL_CHALLENGING) |

🚫 DO NOT make counting the primary difficulty. The challenge MUST come from interpreting visual patterns, relationships, or trends — NOT from how many items need counting. If you must count, it must be the final step AFTER genuine visual reasoning (classification, comparison, tracing, value reading), and the count itself must be small and obvious.

🚫 DO NOT ask for a specific numeric value that requires tracing from a curve point to the y-axis (e.g., "What is the y-value at x=5?" or "What is the peak value?"). Different readers will get different answers — this is ambiguous and defeats exact matching.

🚫 ANTI-PATTERNS (DO NOT USE — these are mechanical, not reasoning):
- "Count the number of X, Y, and Z" where the primary effort is tallying many items
- "Classify [A], [B], [C], then sum all counts" — counting with labels, not reasoning
- "How many arrows/connections touch [component]?" without requiring classification or comparison
- "How many bars exceed [value]?" — counting instead of reading and comparing specific values
- Counting chart furniture: tick labels, axis labels, colorbar ticks, legends — OCR-able, useless
- Chaining multiple independent counts with addition/multiplication — manufactured difficulty
- Any question answerable from text alone without seeing the image
- Asking for a specific y-axis value that requires tracing from a data point to the axis — this has multiple defensible answers
- Questions requiring reading an interpolated value between axis ticks

The question MUST:
- Reference specific VALUES, data points, peaks, regions, or visual features
- Require COMPARISON or RANKING — avoid asking for specific numeric values that require interpolating between axis labels
- Be UNANSWERABLE from text alone — the image must be indispensable
- Present a meaningful visual challenge, not a counting exercise or value-tracing exercise

✅ What a GOOD Challenging question looks like (self-check before answering):
- PATTERN FIT: The chosen strategy matches the image structure (e.g., grouped bars → rank change, not trend-direction)
- Uses ORDINAL reasoning (rank, position, order) not cardinal (specific y-axis value)
- Names ALL competitors/entities explicitly ("among X, Y, Z, W") — no hidden scope
- Compares across TWO conditions (before/after, low/high, panel A/panel B)
- Produces a DISCRETE answer (rank delta, direction, which one) — never interpolated
- Has ONE clear operation: "how many positions it drops" or "which is higher"

Even if the image looks simple or has few elements, FORCE a multi-step question — compare across regions, read specific values, or interpret visual relationships. A simple counting question defeats the purpose. The author explicitly chose Challenging.

QA handbook rules:
- English, 1 sentence (2 max for format spec), must need the image
- No yes/no, no "how does" / "what trend" / "explain"
- No "none" / "cannot be determined" answers
- No option restriction like "Out of the 3..."
- No domain jargon (no sp3, p-value, EBITDA, etc.)
- HARD RULE: Answer is EXACTLY the shortest possible response. If format is "number", answer is ONLY the number (e.g., "6", "2.5"). If format is "word", answer is ONE word (e.g., "Panel A", "increases", "left"). NEVER include explanations, parentheticals, units (no "~36 GW"), or full sentences in the answer field. If the answer is "6 times greater", reduce it to "6" and make the question ask for the factor.
- answer_format must match the answer type: "number" for numeric answers, "word" for text labels, "phrase" only for spatial answers like "red mug"
- The question must be CLEAR enough that a careful Gemini pass can solve it
- Add ONE intermediate reasoning step: combine two values, or locate then compare
- Pin the answer down with constraints: specify which series, panel, axis, subset

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"chart|general_image|spatial"}}"""),
)


SPATIAL_DRAFT_PROMPT = PromptTemplate(
    "SPATIAL_DRAFT_PROMPT",
    _with_strict("""Create a hard spatial-reasoning question for this natural image that requires looking at 3D layout, object positions, depth, or spatial relationships.

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
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""),
)


SPATIAL_CHALLENGING_PROMPT = PromptTemplate(
    "SPATIAL_CHALLENGING_PROMPT",
    _with_strict("""Create a CHALLENGING spatial-reasoning question for this natural image that Qwen 3.6-35B-A3B will FAIL on but Gemini is likely to PASS.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Qwen's known weaknesses:
- ODInW13 (object detection/counting): 50.8 — weak at detecting many objects
- ZEROBench_sub (zero-shot reasoning): 34.4 — weak at novel spatial tasks
- RefSpatialBench: 64.3 — moderate at referenced spatial relations

Use these PROVEN spatial strategies for Challenging (genuine visual REASONING, not mechanical counting):
1. Multi-object exclusion: "Which is the third object to the right of X, excluding objects smaller than Y?"
2. Depth + attribute: "Which object furthest from the camera has the brightest color?"
3. Perspective switching + occlusion: "From the seated person's view, which object is partially blocked by the lamp?"
4. Multi-step containment: "What object sits on the surface that is between the plant and the window?"
5. Spatial comparison: "Which object is closer to the camera: the [A] on the left or the [B] on the right?"

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

Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""),
)


SPATIAL_HARDEST_PROMPT = PromptTemplate(
    "SPATIAL_HARDEST_PROMPT",
    _with_strict("""Create an EXTREMELY HARD spatial-reasoning question for this natural image that Qwen 3.6-35B-A3B will FAIL on.

This is a NATURAL IMAGE (photo or scene) — not a chart or diagram.

Qwen's known weaknesses:
- ODInW13: 50.8 — weak at detecting many objects
- ZEROBench_sub: 34.4 — weak at novel spatial tasks

Strategies (genuine visual REASONING, not mechanical counting):
1. Depth-plane comparison: "Which object appears closer to the camera: the red chair or the blue table?"
2. Spatial paths: "Starting from the doorway and moving toward the sink, which object would you pass third?"
3. Multi-step filtering: "Among objects on the top shelf, which is to the left of the blue vase and darker than the gray box?"
4. Object relationship: "What object sits between the plant and the window, and is it above or below the counter?"

Rules: English, 1 sentence (2 max for format spec), must need the image. No yes/no. Answer is 1 word or number.
No trick answers.
Return JSON only: {{"question":"...","answer":"...","answer_format":"word|number|phrase","task_type":"general_image"}}"""),
)


CHECK_ANSWER_PROMPT = PromptTemplate(
    "CHECK_ANSWER_PROMPT",
    """Answer the question based ONLY on the image. Do not guess.
Think step by step before giving your final answer.

Question: {question}

Return ONLY valid JSON with two keys:
- "reasoning": "<step-by-step reasoning showing what you observed and how you computed the answer>"
- "answer": "<your final answer>"

Return ONLY valid JSON: {{"reasoning": "<your step-by-step reasoning>", "answer": "<your final answer>"}}""",
)


VERIFY_ANSWER_PROMPT = PromptTemplate(
    "VERIFY_ANSWER_PROMPT",
    """You are verifying whether a proposed answer means the same thing as the expected answer for a visual-reasoning question.

Consider:
- Synonyms and equivalent expressions ("more than half" ≈ "majority")
- Numeric equivalence ("10" ≈ "ten", "0.5" ≈ "1/2")
- Unit differences ("50%" ≈ "half", "100cm" ≈ "1m")
- Different phrasings that convey the same meaning
- The question context when interpreting answers

When the answers DIFFER, analyze the VLM's reasoning to identify what it may have miscounted or misinterpreted, and suggest what the correct reasoning should be.

Question: {question}
Expected answer: {golden_answer}
Proposed answer: {vlm_answer}
VLM reasoning: {vlm_reasoning}

Return ONLY valid JSON: {{"match": true/false, "explanation": "<brief reason why they match or differ>", "analysis": "<if match=false: based on the VLM's reasoning, identify what it miscounted or misinterpreted and what the correct approach should be. If match=true: return empty string>"}}""",
)


VERIFY_PROMPT = PromptTemplate(
    "VERIFY_PROMPT",
    _with_strict("""Look at this image carefully. The question and answer below were
drafted by AI. Verify if the answer is CORRECT based on the image, and fix
if wrong. Reply with corrected JSON only (same keys).

Draft question: {question}
Draft answer: {answer}"""),
)


SELF_CRITIQUE_PROMPT = PromptTemplate(
    "SELF_CRITIQUE_PROMPT",
    _with_strict("""You drafted this question for a CHALLENGING visual-reasoning task:

Q: {question}
A: {answer}
{fx_context}
Rate 1-5: would Qwen 3.6-35B-A3B likely FAIL on this? A "5" means definitely fails, "1" means definitely solves.

CRITICAL CHECK FIRST: Could a smart person answer this WITHOUT seeing the image?
- If the question provides all the numerical data needed to compute the answer in the text (e.g., "Panel A's X covers 0 to 5, Panel B's X covers 0 to 4. What is the ratio?" — answerable from text alone), score it 1 regardless of math complexity. The IMAGE must be REQUIRED.
- If the question references a SPECIFIC visual element (peak, trough, color region, data point) that requires looking at the image, it's valid.

A question deserves a HIGH score (4-5) only if it:
- References a visual element (peak, trough, color region, specific data point, position on chart)
- Requires COMPARING values or ranking from the image (not extracting a specific numeric value by tracing to an axis)
- Cannot be answered from text alone

A question deserves a LOW score (1-2) if it is:
- Pure math (ratio/difference/sum) of values stated in the question text
- A simple COUNT of axis labels, tick marks, colorbar values
- A generic "How many X are in the image?" with no filter
- Mechanical counting of chart furniture
- Answerable without the image (provides all needed data in text)
- Asks for a specific y-axis value that requires tracing from a curve point to the axis (ambiguous answers)

If score is 1-3, REWRITE the question to:
- REMOVE any explicit data values from the text (no "X covers 0 to 5")
- ASK about a COMPARISON or ranking (which is higher/larger/steeper, what rank, which direction)
- Make the image REQUIRED (no way to answer without seeing it)

FORMATTING RULES for the rewritten question and answer:
- Question: English only, 1 sentence (2 max for format spec). Must reference visual elements (axis labels, panel names, data series, regions). No yes/no, no "how does"/"what trend"/"explain". No option restrictions like "Out of the 3...". No domain jargon.
- Answer: EXACTLY the shortest possible response — 1 word or 1 number only. NEVER include explanations, parentheticals, units, full sentences, or reasoning. Examples: "6", "Panel A", "increases", "left".
- answer_format: must match the answer type — "number" for digits, "word" for text labels, "phrase" only for spatial answers.
- task_type: "chart" for chart figures, "general_image" for natural images, "spatial" for spatial-reasoning tasks.

Keep the answer in sync. Return JSON only: {{"score": <1-5>, "rewrite_question": "...", "rewrite_answer": "...", "answer_format": "...", "task_type": "..."}}"""),
)


FIX_PROMPT = PromptTemplate(
    "FIX_PROMPT",
    _with_strict("""You are fixing an existing visual-reasoning Q&A pair based on validation feedback.

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
  "fix_summary": brief explanation of what changed"""),
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
        CHECK_ANSWER_PROMPT,
        VERIFY_ANSWER_PROMPT,
        VERIFY_PROMPT,
        SELF_CRITIQUE_PROMPT,
        FIX_PROMPT,
    ]
}
