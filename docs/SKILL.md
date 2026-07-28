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

### Manufacturing Difficult — Key Guidance

> **What to avoid**: making a task difficult mainly by requiring a large manual count + arbitrary filters, or chaining counting with unnecessary arithmetic just because Qwen struggles with it. This is "manufactured difficulty" and should be avoided.

> **What's acceptable**: Counting as the **final step** after meaningful visual reasoning — tracing connections, resolving spatial relationships, distinguishing overlapping/obscured elements, or applying a visually meaningful condition. The challenge should come from comparing or interpreting a meaningful visual pattern, relationship, or trend.

**Applies to**: Chart tasks primarily, both rework tasks AND future tasks.

### CHALLENGING Patterns (Qwen-fail, Gemini-pass)

PROVEN example (cross-panel comparison):
> "What is the difference between the maximum value of series A in panel X and the maximum value of series B in panel Y?" → requires reading peaks from two charts

Proven strategies (genuine visual REASONING, not mechanical tasks):
1. **Cross-panel comparison**: "What is the difference between [value A] in panel X and [value B] in panel Y?"
2. **Value-at-specific-location**: "At x=[value], what is the approximate y-value of [series] in panel X?"
3. **Peak/trough reading**: "What is the maximum y-value on the axis of panel A, and which panel has the highest maximum?"
4. **Ratio across panels**: "What is the ratio of the peak value in panel X to the peak value in panel Y?"
5. **Trend-based comparison**: "Between x=[a] and x=[b], which panel shows the steepest increase?"
6. **Cross-panel arithmetic**: "Sum the maximum y-axis values across all panels."
7. **Visual flow tracing**: "Trace the path from [component A] to [component B]. Which intermediate component receives inputs from both?"
8. **Attribute comparison**: "Which component has the larger value: the one labeled [X] or the one labeled [Y]?"

**Chart-specific strategies** (when `figure_type = chart_graph_text`):
9. **Read-and-compare**: "What is the difference between the maximum [axis] value in panel A and the maximum in panel B?"
10. **Value-at-intersection**: "At x=[value], what is the y-value of series [name]?"
11. **Ratio across panels**: "What is the ratio of [quantity A] in panel X to [quantity A] in panel Y?"
12. **Cross-panel arithmetic**: "Sum the peak z-values across all panels in the figure."

### HARDEST Patterns (both models fail)

1. **Multi-panel reasoning**: "Compare [element A] in panel X with [element B] in panel Y — which is larger and by how much?"
2. **Value-at-specific-location**: "At x=[value], what is the y-value of series [name] in panel X?"
3. **Cross-attribute comparison**: "Which [category] has the highest [attribute] across all panels?"
4. **Peak/trough reading**: "What is the y-value at the global maximum of the curve in panel A?"
5. **Value comparison across charts**: "What is the ratio of [value A] in panel X to [value B] in panel Y?"
6. **Visual flow tracing**: "Trace the path from [component A] to [component B]. What component sits at the junction?"
7. **Attribute comparison**: "Which connected component has the largest [attribute]: [X] or [Y]?"

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
5. Spatial comparison: "Which object is closer to the camera: the [A] on the left or the [B] on the right?"

### HARDEST Spatial (both fail)
1. Depth-plane comparison: "Which object appears closer to the camera: the red chair or the blue table?"
2. Spatial paths: "Starting from the doorway, which object would you pass third before reaching the sink?"
3. Multi-step filtering: "Among objects on the top shelf, which is to the left of the blue vase and darker than the gray box?"
4. Object relationship: "What object sits between the plant and the window, and is it above or below the counter?"

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
9. No domain jargon

## AI Drafting Pipeline

The system has 3 drafting modes in `authoring/ai_draft/`:

### Simple Draft (`ai_draft/core.py` `draft_qa()`)
Single LLM call → parse JSON → validate → guardrails → retry with feedback if failed.

### Self-Critique Loop (`ai_draft/composition.py` `draft_with_self_critique()`)
1. Generate initial draft
2. Critique with score 1-5 (5 = definitely fails Qwen)
3. If score < 4: rewrite question + answer, repeat up to `max_rounds`
4. Return final draft

### Consensus (`ai_draft/composition.py` `draft_qa_consensus()`)
1. Query router determines optimal pipeline (simple / RAG-enhanced / consensus / self-critique)
2. Generate N independent drafts (default 3)
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

All prompts are versioned with SHA-256 hashes via `PromptTemplate.version_id` (e.g. `CHALLENGING_PROMPT@a1b2c3d4e5f6`). Every generation records `prompt_template_name`, `prompt_version_id`, and `prompt_text_hash` in the `GenerationAttempt` table for full traceability.

### Supporting Modules

| Module | Location | Purpose |
|--------|----------|---------|
| **History injection** | `authoring/_history_context.py` | Injects past attempts + few-shot examples + previous question into prompts |
| **Guardrails** | `authoring/_guardrails.py` | Quality checks (plausible, extreme, format, diversity) with auto-retry |
| **Telemetry** | `authoring/_draft_telemetry.py` | Logs every generation attempt to DB (38+ fields) + JSONL |
| **Config** | `authoring/_draft_config.py` | Model selection, token limits, timeouts per difficulty |
| **Query router** | `services/query_router.py` | Routes requests to optimal pipeline based on difficulty |
| **Adaptive router** | `agents/adaptive_router.py` | Learns from past performance to improve pipeline selection |
| **Query decomposer** | `agents/query_decomposer.py` | Breaks hard questions into sub-tasks for the prompt |
| **Document grader** | `agents/document_grader.py` | Filters retrieved documents by relevance |
| **CRAG pipeline** | `services/rag_pipeline.py` | CAG (semantic cache) → RAG (hybrid retrieve + rerank) |
| **Hybrid retriever** | `components/hybrid_retriever.py` | ChromaDB + sentence-transformers semantic search |
| **Reranker** | `components/reranker.py` | Cross-encoder re-ranking of retrieval results |
| **Semantic cache** | `services/semantic_cache.py` | CAG layer — caches LLM results keyed by prompt embeddings |
| **Cost tracker** | `observability/cost_tracker.py` | Token/cost estimation per model |
| **Input guard** | `security/input_guard.py` | Prompt injection detection |
| **Content filter** | `security/content_filter.py` | PII detection, toxicity filtering |
| **Output filter** | `security/output_filter.py` | System prompt leakage prevention |
| **Agent registry** | `agents/registry.py` | Agent metadata/capability registry |
| **Agent context** | `agents/context.py` | Shared AgentContext (fork, delegation chain, artifacts) |
| **Orchestrator** | `agents/orchestrator.py` | Multi-agent orchestration (plan → delegate → aggregate) |
| **Reviewer** | `agents/reviewer.py` | Draft scoring (1-5) with quality/format suggestions |
| **Job queue** | `scheduler/queue.py` | DB-backed FIFO queue with priority and retry |
| **Worker** | `scheduler/worker.py` | Subprocess worker with sentinel-based shutdown |
| **Auth** | `personalization/auth.py` | PBKDF2-SHA256 hashing + UUID token auth |
| **Auth middleware** | `personalization/middleware.py` | FastAPI AuthMiddleware (Bearer + X-API-Key) |
| **Personalizer** | `personalization/personalizer.py` | User preference → routing config application |
| **Vision models** | `vision/models.py` | Lazy-loaded ResNet-18 feature extractor |
| **Vision extractor** | `vision/extractor.py` | 512-dim embedding + cosine similarity |
| **Vision classifier** | `vision/classifier.py` | Figure type classification with prototype matching |
| **Prompt registry** | `prompts/` | DB-backed hot-swappable prompt templates |
| **MCP server** | `mcp/` | MCP tools (generate, validate, history, search, health, analytics, orchestrate, enqueue) |

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
- **Objective classification**: clear criteria = easy for Qwen
- **Pure math**: ratio/difference of values stated in question text — image not required
- **Generic count without filter**: "How many X are in the image?"

## Multi-Agent Orchestration

The `agents/orchestrator.py` coordinator extends the drafting pipeline with multi-agent collaboration:

1. **Plan**: Orchestrator uses `query_decomposer` to break hardest tasks into reasoning steps
2. **Delegate**: Generator agent (wraps `draft_qa`) creates N attempts (3 for hardest, 1 otherwise)
3. **Review**: Reviewer agent scores each draft 1-5 on quality, format, and content
4. **Aggregate**: Best draft selected by validation quality; review suggestions applied if score < 3

AgentContext tracks delegation chains and shared artifacts across the workflow.

### Draft Reviewer (`agents/reviewer.py`)

The Reviewer scores drafts across these dimensions:
- **Quality**: Direct mapping of `_validation_quality` (0.9+ = 5, 0.7+ = 4, 0.5+ = 3, else 2)
- **Penalties**: Empty draft → 1, very short answer → -1, answer in question → -1, format mismatch
- **Output**: Score (1-5), passed (bool), suggestions (list), strengths (list)

## Async Job Queue

The `scheduler/` module provides a DB-backed async job queue for generation tasks:

- **No external deps**: Uses existing SQLite (WAL mode for concurrent access)
- **Priority FIFO**: Jobs ordered by priority DESC, created_at ASC
- **Automatic retry**: Configurable max_attempts (default 3), re-queued on failure
- **Subprocess worker**: Runs as isolated subprocess via `subprocess.Popen`
- **Graceful shutdown**: Sentinel file pattern — worker exits when file is deleted
- **Job types**: `generate_qa`, `validate_batch`, `rag_index`

### Scheduler API

| Endpoint | Description |
|----------|-------------|
| `POST /api/scheduler/enqueue` | Enqueue a job `{type, payload, priority}` |
| `GET /api/scheduler/status/{id}` | Poll job status |
| `POST /api/scheduler/cancel/{id}` | Cancel a queued job |
| `GET /api/scheduler/queue` | List queue depth + recent jobs |
| `GET /api/scheduler/worker` | Check if worker is alive |

## Authentication & Personalization

The `personalization/` module provides token-based auth with user-specific preferences:

- **Password hashing**: PBKDF2-SHA256 with random salt (no external deps)
- **Tokens**: UUID-based, stored in DB with optional expiry (default 30 days)
- **Middleware**: FastAPI `AuthMiddleware` checks `Authorization: Bearer` or `X-API-Key`
- **Public routes**: `/auth/login`, `/auth/register`, `/health`, `/mcp/*`, static files
- **User profiles**: Preferred model, difficulty, figure type, prompt style (concise/detailed/default)
- **Key-value preferences**: Arbitrary key-value pairs per user, applied to routing configs

### Auth API

| Endpoint | Description |
|----------|-------------|
| `POST /auth/register` | Create account `{username, password}` |
| `POST /auth/login` | Login, receive Bearer token |
| `GET /auth/profile` | Get profile + preferences (auth required) |
| `PUT /auth/profile` | Update profile fields (auth required) |
| `PUT /auth/preferences/{key}` | Set preference value (auth required) |
| `GET /auth/preferences` | Get all preferences (auth required) |

## Local CNN Vision

The `vision/` module provides optional local figure classification via ResNet-18:

- **Lazy-loaded**: Model only loads on first use — zero import-time overhead
- **Fallback**: If torch/torchvision unavailable, falls back to heuristic `classify_figure_type()` in `filters.py`
- **Feature extraction**: 512-dim embedding vector for similarity comparison
- **Prototype matching**: Store known figure embeddings for few-shot classification
- **Cosine similarity**: Built-in `cosine_similarity()` for embedding comparison

## Key Insights

1. **Subjective classification beats objective counting** for HARDEST. Qwen fails when it needs to judge "colorful" vs. "gray" because the boundary is fuzzy.
2. **Multi-step is essential**: count → classify → sum (or filter → count → compare). Each step compounds error probability.
3. **Charts must reference data values**, not chart furniture. Read peaks, compare across panels, compute ratios from visual readings.
4. **Use the self-critique loop**: generate, score 1-5, rewrite if below 4, repeat.
5. **Dynamic model selection**: the system queries past generations to pick the best-performing model per (figure_type, difficulty).

## Observability & Monitoring

The system provides observability through `src/arxiv_manager/observability/`:

- **Structured JSON logging** (`tracer.py`): All log events written to `storage/_structured_log.jsonl` with trace IDs, span names, and extra fields. Use `log_event()` for structured events, `@contextmanager span()` for timing operations.
- **Trace spans**: Per-request tracing via `with span("operation_name", key=val)` context manager. Automatically captures duration and metadata. Access current trace via `current_trace_id()`.
- **Health endpoint**: `GET /health` returns DB connectivity + API key status. `GET /health?full=true` also checks LLM connectivity. Returns `{"status": "ok", "version": "0.2.0", "checks": {...}}`.
- **Generation attempt telemetry**: Every draft/prompt/critique stored in `GenerationAttempt` DB table (35+ fields) and `storage/_draft_telemetry.jsonl` for historical analysis.
- **Metrics dashboard**: `GET /metrics` serves an HTML dashboard showing success rate, latency (avg/p50/max), breakdown by difficulty and figure type.
