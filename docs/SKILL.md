# arxiv-manager Challenging Question Generator

## Overview

This skill generates visual reasoning questions for Realm SIQ tasks from scientific figures (arXiv CC0 papers).
The goal is to create questions that exploit Qwen 3.6-35B-A3B weaknesses while being solvable by humans:

- **ODInW13: 50.8** — Weak at object detection / counting many elements
- **ZEROBench_sub: 34.4** — Weak at zero-shot reasoning / novel task formats

Qwen is strong at: diagram reading (AI2D: 92.7), chart reading (CharXiv: 78.0), visual math (Mathvista: 86.4).

## Difficulty Taxonomy

Three tiers, distinguished by which models pass/fail:

| Difficulty | Qwen | Gemini | Strategy |
|---|---|---|---|
| EASY | Passes | Passes | Simple observation, 1 element, no multi-step |
| CHALLENGING | Fails | Passes | Multi-step, needs classification + arithmetic, chart data refs |
| HARDEST | Fails | Fails | 10+ items, subjective classification, rare patterns |

Classification rule: Qwen passes → **EASY**, only Qwen fails → **CHALLENGING**, both fail → **HARDEST**.

### Complexity Thresholds (from `image_analyzer.py`)

Complexity score from `audit_figure()` (0-1 scale):

| Likelihood | Chart (`chart_graph_text`) | General Image |
|---|---|---|
| HARDEST | >= 0.85 + dense | >= 0.75 + dense |
| CHALLENGING | >= 0.55 | >= 0.45 |
| EASY | >= 0.35 | >= 0.25 |
| Rejected | < 0.35 | < 0.25 |

## Task Types & Answer Formats

### Task Types
- **chart** — Scientific charts, graphs, plots (axes, data series, panels)
- **general_image** — Natural images, photos, scenes, diagrams
- **spatial** — 3D layout, object positions, depth, occlusion in natural scenes

### Answer Formats
- `number` — Single numeric value
- `word` — Single word (object name, color, label)
- `phrase` — Short multi-word answer (max 4 words)
- `year` — Year value
- `percent` — Percentage (e.g. "50%")
- `integer` — Whole number

## Figure Type Classification (from `filters.py`)

- **chart_graph_text** — Charts, graphs, plots with axes, data curves, bars. Background tends to be white, high text density, border lines present.
- **general_image** — Natural images, photos, architecture diagrams, networks. Higher color saturation, lower text density.

## Question Patterns by Difficulty

### EASY Patterns
- Focus on a single clear element (count, color, shape, label, position)
- Simple observation — no multi-step reasoning
- Avoid counting large numbers (>10)
- Answer is 1 word or 1 number

### CHALLENGING Patterns (Qwen-fail, Gemini-pass)

PROVEN example (optical computing):
> "Count these three types of elements in the diagram, excluding anything inside the blue dashed boxes: (1) black rectangular mirrors, (2) standalone gray ellipse lenses, and (3) groups of colored filter bars. Sum all three counts." → 18 (6 mirrors + 8 lenses + 4 filter groups)

Strategies:
1. **Multi-type count + sum**: "Count these three types of elements, then give their sum: (1) [type A], (2) [type B], (3) [type C]"
2. **Multi-type count + sum + exclusion**: "Count [type A], [type B], and [type C] excluding anything inside [region]. Sum all three counts."
3. **Subjective classification + count**: "Count [elements] that contain color (not gray or blank). How many are there?"
4. **Spatial targeting + count**: "Count arrows entering red block X, blue block Y, and green block Z combined."
5. **Threshold filter + count**: "Count [elements] with [attribute] greater than [value] across all panels."
6. **Cross-attribute filter + arithmetic**: "How many more [type A] than [type B] are visible in the image?"
7. **Spatial + count**: "How many of the visible [objects] are in the back row vs. the front row?"

**Chart-specific strategies** (when `figure_type = chart_graph_text`):
8. **Read-and-compare**: "What is the difference between the maximum [axis] value in panel A and the maximum in panel B?"
9. **Value-at-intersection**: "At the x-value of [value], what is the approximate y-value in panel [panel]?"
10. **Ratio across panels**: "What is the ratio of [quantity A] in panel [panel] to [quantity A] in panel [panel]?"
11. **Threshold-based reading**: "How many of the bars in panel A exceed a value of [number]?"
12. **Cross-panel arithmetic**: "Sum the peak z-values across all panels in the figure."

### HARDEST Patterns (both models fail)

1. **Multi-type counting + sum**: "Count all [type A] and [type B], then give their sum"
2. **Subjective classification**: "Count [elements] that appear to contain [subjective criterion]"
3. **Exclusion counting**: "Excluding [subset], how many [elements] remain?"
4. **Spatial classification**: "How many [elements] touch/connect to [specific component]?"
5. **Path tracing**: "Trace path from [A] through [B] to [C]. How many [elements] does it cross?"

## Spatial Reasoning (Natural Images)

For natural images (photos, scenes) — NOT charts or diagrams.

### EASY Spatial
- Viewer-centered left/right
- Depth/distance ("closest to camera")
- Relative height ("highest in image")
- Containment/support ("sitting on top of")
- Between/surrounded by
- Occlusion/in front

### CHALLENGING Spatial (Qwen-fail, Gemini-pass)
1. Multi-object exclusion: "Which is the third object to the right of X, excluding objects smaller than Y?"
2. Depth + attribute: "Which object furthest from the camera has the brightest color?"
3. Perspective switching + occlusion: "From the seated person's view, which object is partially blocked by the lamp?"
4. Multi-step containment: "What object sits on the surface that is between the plant and the window?"
5. Relative positioning with count: "Count how many objects are between the white chair and the blue table."

### HARDEST Spatial (both fail)
1. Count across depth planes: "How many objects are closer to the camera than the red chair?"
2. Spatial paths: "Starting from the doorway, which object would you pass third before reaching the sink?"
3. Occlusion counting: "How many objects are partially hidden by the table?"
4. Multi-step filtering: "Among objects on the top shelf, which is to the left of the blue vase and darker than the gray box?"

## Chart Anti-Patterns (NEVER use for charts)

These are mechanical OCR tasks Qwen handles perfectly:
- "How many tick labels are on the [axis]?"
- "Count the [axis] labels in panel A"
- Any question counting axis labels, colorbar ticks, legends, or similar OCR-able elements
- "How many [elements] are visible in the image?" without a filter/comparison/arithmetic
- Counting visual artifacts (labels, ticks, grids) rather than data values

Instead, chart questions MUST: reference specific axis VALUES, data points, peaks, regions, or numerical features of the data.

## Validation Rules (from `_rule_groups.py`)

### Format Checks
- No binary/T-F question ("Is X...?", "Are Y...?")
- Answer format must be specified (number/word/phrase/year/percent/integer)
- Answer max 4 words, max 50 chars
- No trick answers: "none", "cannot be determined", "n/a", "unclear", "none of the above"
- Question must end with ? or .
- Single question only (one ? mark)

### Content Checks
- Max 2 sentences (prefer 1)
- No option restriction: "Out of the 3...", "From the following..."
- No domain jargon: hybridization, sigma/pi bond, LUMO, HOMO, sp2/sp3, EBITDA, WACC, DCF, p-value, chi-square
- Must reference visual content (chart, graph, panel, color, left/right, etc.)
- Answer must match declared format
- No explanation questions: "Explain how...", "What trend...", "How does..."

### Complexity Checks
- Must require multi-step reasoning (comparison, ranking, sum, difference, ratio)
- Chart questions: must reference data values (peaks, axis values, cross-panel), not chart furniture
- No generic count: "How many X are in the image?" without filter/comparison/arithmetic
- No chart math-only: question must require seeing the image, not just computing from stated values

### Handbook Basics
- Answer must be derivable from question
- Capitalize first letter, no double spaces
- Avoid extreme-seeking ("highest", "lowest", "most") — Qwen checks these first
- Prefer threshold filters ("fewer than 10", "greater than 5") over extreme-seeking
- Cross-panel references are good
- Caption must not give away the answer
- Avoid extreme answer values (prefer intermediate)

### Visual-Dependence Tests
- **Test 1**: A smart person should NOT be able to answer without the image
- **Test 2**: The answer must be objective — two reasonable people give the same answer
- The question must NOT provide the data values needed to compute the answer in the text

### Final Checks
- Question should be visuo-spatial, not pure calculation
- Not text-only (no "what does the text say?")
- Not long-winded/awkward
- No noise conditions that don't change the answer
- No list answers (max 3 short elements)
- No watermark/copyright images
- figure_type should match task_type

## Core Rules (All Difficulties)

All questions must:
1. Be a single sentence (2 max for format spec)
2. Require the image to answer
3. Have a single, unambiguous answer
4. Be English
5. Not be yes/no or binary
6. Not be explanation questions ("how does", "what trend")
7. Not use trick answers ("none", "cannot be determined")
8. Not use option restriction ("Out of the 3...")
9. Have an unusual answer number (avoid 2, 3, 4, 5 — Qwen guesses these)
10. No domain jargon

## AI Drafting Pipeline

The system has 3 drafting modes (from `ai_draft.py`):

### Simple Draft (`draft_qa`)
Single LLM call → parse JSON → validate → guardrails → retry with feedback if failed.

### Self-Critique Loop (`draft_with_self_critique`)
1. Generate initial draft
2. Critique with score 1-5 (5 = definitely fails Qwen)
3. If score < 4: rewrite question + answer, repeat up to `max_rounds`
4. Return final draft

### Consensus (`draft_qa_consensus`)
1. Generate N independent drafts (default 3)
2. Score each: quality_score + 50 if valid + 10 if quality >= 80
3. Feed validation errors back as feedback for subsequent attempts
4. Pick best by score
5. Optionally verify (ask model to check answer)
6. Return best verified draft

## Prompt Templates (11 total)

| Template | Purpose |
|---|---|
| DRAFT_PROMPT | Default hard visual-reasoning |
| EASY_PROMPT | Simple observation |
| REGEN_PROMPT | Regeneration with validation feedback |
| HARDEST_PROMPT | Qwen-exploiting (count 10+, multi-step, exclusion) |
| CHALLENGING_PROMPT | Qwen-fail, Gemini-pass (multi-type, chart-specific) |
| SPATIAL_DRAFT_PROMPT | Spatial reasoning for natural images |
| SPATIAL_CHALLENGING_PROMPT | Challenging spatial |
| SPATIAL_HARDEST_PROMPT | Hardest spatial |
| SPATIAL_REGEN_PROMPT | Spatial regeneration with feedback |
| VERIFY_PROMPT | Verification pass |
| SELF_CRITIQUE_PROMPT | Self-critique scoring (1-5) |

All prompts are versioned with SHA-256 hashes. Every generation records `prompt_version_id` for traceability.

## Workflow

### Web UI (Author Page)
1. Upload image or select from arXiv extraction
2. System runs `audit_figure()` → complexity score + figure type + density check
3. Difficulty potential determined (HARDEST/CHALLENGING/EASY/REJECTED)
4. AI drafts Q&A via self-critique loop (for challenging/hardest) or plain draft
5. User reviews, edits, proposes → saved as Figure + Task in DB

### CLI
```bash
arxiv-manager task new --image-id <id> --ai --hardest
arxiv-manager task new --image-id <id> --ai --challenging
arxiv-manager task new --image-id <id> --ai
```
Or batch mode:
```bash
arxiv-manager task new-batch --limit 5 --difficulty hardest
```

### Full Pipeline
```
arXiv search → download PDF → extract figures → audit → filter → store → task create → AI draft → validate → set difficulty → submit
```

## Proven Tasks

| # | Title | Pattern | Answer | Difficulty |
|---|---|---|---|---|
| 3 | CNN Filter and Block Count | Multi-type counting + sum | 69 | HARDEST |
| 5 | CNN Filter Grid Count | Subjective classification | 30 | HARDEST |
| 6 | CNN Element Sum | Multi-type counting + sum | 69 | HARDEST |
| — | Optical Computing Diagram | Multi-type + exclusion + sum | 18 | CHALLENGING |

## What Doesn't Work

- **Simple counting**: "How many cells in this grid?" → Qwen passes easily
- **Chart furniture counting**: axis labels, ticks, colorbars → Qwen OCRs perfectly
- **Questions with text shortcuts**: caption gives away the answer
- **Small numbers**: 2, 3, 4, 5 are guessable by Qwen
- **Objective classification**: clear criteria = easy for Qwen
- **Pure math**: ratio/difference of values stated in question text — image not required
- **Generic count without filter**: "How many X are in the image?"

## Key Insights

1. **Subjective classification beats objective counting** for HARDEST. Qwen fails when it needs to judge "colorful" vs. "gray" because the boundary is fuzzy.
2. **Multi-step is essential**: count → classify → sum (or filter → count → compare). Each step compounds error probability.
3. **Charts must reference data values**, not chart furniture. Read peaks, compare across panels, compute ratios from visual readings.
4. **Use the self-critique loop**: generate, score 1-5, rewrite if below 4, repeat.
5. **Dynamic model selection**: the system queries past generations to pick the best-performing model per (figure_type, difficulty).
