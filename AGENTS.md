# ArXiv Manager — Project Context

## Overview

AI-powered assistant for extracting scientific figures from arXiv PDFs, analyzing them, and generating visual-reasoning Q&A tasks. Used by the AtMyDesk team. Production AI lifecycle with CRAG architecture, observability, security, and MCP integration.

## Tech Stack

- **Language:** Python 3.11+
- **Web:** FastAPI + Jinja2 + HTMX + Tailwind CSS
- **Database:** SQLite + SQLModel (ORM)
- **CLI:** Typer + Rich
- **LLM:** OpenCode API (OpenAI-compatible chat completions)
- **PDF/Images:** PyMuPDF, Pillow, ImageHash
- **RAG:** ChromaDB + sentence-transformers (all-MiniLM-L6-v2) + LangChain
- **Observability:** Structured JSON logging, trace spans, cost tracking

## Key Architecture

```
src/arxiv_manager/
├── authoring/
│   ├── ai_draft/           # LLM integration (client, parser, core, composition)
│   │   ├── _fact_checker.py   # Adversarial premise fact-check (SUPPORTED/NOT_SUPPORTED/UNVERIFIABLE claims)
│   │   ├── _determinism.py    # 3-run sampled answer determinism gate (normalized numeric / semantic match)
│   │   ├── _api_client.py     # LLM API client (OpenCode), token usage capture
│   │   ├── _response_parser.py# Think-block + brace-scan JSON parsing, type coercion
│   │   ├── _image_utils.py    # Shared image encoding for LLM consumption (base64 JPEG)
│   │   ├── core.py            # Prompt building, history injection, guardrails
│   │   └── composition.py     # Self-critique + validation/feedback retry loops
│   ├── _draft_prompts.py   # 14+ prompt templates with SHA-256 versioning (CHECK_ANSWER, VERIFY_ANSWER, FACT_CHECK, HARDEST/CHALLENGING + SPATIAL variants with FACT SAFETY blocks)
│   ├── _model_cards.py     # Verified capability cards for Qwen3.6-35B-A3B + Gemini 3.5 Flash (HF/OpenRouter, 2026-08)
│   ├── _draft_telemetry.py # Generation attempt logging to DB + JSONL
│   ├── _guardrails.py      # Quality checks with auto-retry + feedback
│   ├── _history_context.py # History injection, few-shot, model selection (difficulty-aware ordering)
│   ├── validator.py        # 28+ handbook rule validation
│   └── image_analyzer.py   # Figure suitability classification
├── analytics/
│   └── strategies.py       # Strategy-class × model-verdict aggregation (pluggable data-provider seam)
├── cli/                    # CLI commands (search, task, images, web, check, analytics, index)
├── sourcing/               # arXiv PDF download, figure extraction, filtering
├── web/                    # FastAPI app + 17 Jinja2/HTMX templates
│   ├── routes/             # Route handlers (task, author, arxiv, lifecycle, metrics, health, prompts)
│   └── templates/          # HTML templates with HTMX partials (_determinism.html, strategies.html)
├── components/             # RAG: hybrid retriever, reranker, config
├── services/               # CRAG: RAG pipeline, semantic cache, query router
├── agents/                 # Adaptive router, query decomposer, document grader, tools/
│   ├── registry.py         # Agent metadata/capabilities registry
│   ├── context.py          # AgentContext — shared state across agents
│   ├── orchestrator.py     # Orchestrator — plans, delegates, aggregates
│   └── reviewer.py         # Reviewer — critiques drafts
├── scheduler/              # DB-backed async job queue with subprocess worker
│   ├── models.py           # ScheduledTask (queued | running | done | failed | cancelled)
│   ├── queue.py            # FIFO queue with priority, retry, cancel
│   ├── worker.py           # Standalone worker subprocess (python -m ...worker)
│   ├── manager.py          # Subprocess lifecycle (start/stop via sentinel file)
│   └── routes.py           # API: enqueue, status, cancel, queue depth
├── personalization/        # User accounts, auth, preferences, learning
│   ├── models.py           # User, AuthToken, UserProfile, UserPreference
│   ├── auth.py             # Password hashing (PBKDF2-SHA256), token management
│   ├── middleware.py        # FastAPI AuthMiddleware (Bearer token / X-API-Key)
│   ├── personalizer.py     # Apply preferences to routing configs
│   └── routes.py           # API: login, register, profile, preferences
├── vision/                 # Local CNN-based figure analysis
│   ├── models.py           # Lazy-loaded ResNet-18 feature extractor
│   ├── extractor.py        # 512-dim feature extraction + cosine similarity
│   └── classifier.py       # Figure type classification with prototype matching
├── observability/          # Structured JSON logging, tracing, cost tracker
├── security/               # Input guard, content filter, output filter
├── prompts/                # Hot-swappable prompt registry (DB-backed)
├── mcp/                    # MCP server (8 tools: generate, validate, history, search, health, analytics, orchestrate, enqueue)
├── tracking/               # Task export, platform submission
├── evaluation/             # Golden dataset, offline eval script
├── models.py               # DB models (Paper, Figure, Task, GenerationAttempt, PromptTemplateRecord)
├── db.py                   # SQLite engine + migrations + WAL mode
└── storage.py              # Centralized storage paths
```

## Data Model

- **Paper:** arXiv paper metadata
- **Figure:** Extracted figure image with type, complexity, audit results
- **Task:** Q&A task with question, answer, format, type, difficulty, status
- **GenerationAttempt:** Full trace of every LLM call (38+ fields incl. token usage, Rhea feedback, model run results). Indexed columns: `figure_id`, `task_id`, `model_name`, `validation_quality`
- **IssueReport:** User-reported issues on generation attempts (reason, description, corrected_answer) — fed back into generation prompts
- **PromptTemplateRecord:** DB-backed prompt templates with versioning, rollback support
- **SubmissionLog:** Task submission tracking with review status
- **TaskEvent:** Audit trail for all task state changes (regeneration, update, difficulty_change, rhea_review, issue_report, ai_fix, submit, delete) — used by Task History UI and injected as context during regeneration

## Key Patterns

- **CRAG (CAG + RAG):** Semantic cache check first, then hybrid retrieve + rerank
- **RAG pre-warming:** Embedding model (`all-MiniLM-L6-v2`) and cross-encoder (`ms-marco-MiniLM-L-6-v2`) loaded in background thread at app startup to eliminate cold-start delay
- **History injection:** Past attempts are injected as few-shot examples + "don't repeat" guidance. History injection now uses single DB session to avoid multiple round-trips.
- **Self-critique loop:** LLM scores own draft (1-5) and rewrites, now with 2 rounds
- **Consensus drafting:** Multiple drafts → validate → pick best
- **Guardrails + auto-retry:** Quality checks trigger regeneration with feedback
- **Dynamic model selection:** Best model per (figure_type, difficulty) from historical quality
- **Hot-swappable prompts:** DB-backed templates with API for save/rollback/reload
- **Cost tracking:** Token usage captured from API responses, cost estimated per model
- **IssueReport feedback loop:** User-reported issues (too_easy, wrong_answer) and corrected answers are injected into the generation prompt via `build_figure_history`. Model run results (Qwen/Gemini passes) are copied onto GenerationAttempt on submit and influence few-shot ordering.
- **Arithmetic consistency check:** Validation warns when a question asks for product/sum but the answer may be a simple count.
- **Manufactured difficulty detection:** Validation errors when counting is the primary difficulty source (counting-only chains like "count X and Y then sum" without visual reasoning). Challenge should come from visual reasoning — counting is acceptable only as a final step after meaningful analysis. Now includes multiply/product patterns.
- **Lookup-question detection:** Validation warns when Challenging/Hardest questions are simple lookups ("what is the X of Y?") that don't require multi-step reasoning. Helps prevent questions that are too easy for Qwen.
- **Enhanced reasoning depth check:** `_has_reasoning_depth()` now considers difficulty level — single lookups fail for Challenging/Hardest even if they contain keyword matches.
- **Expanded matchmaking detection:** `_is_single_matchmaking()` now catches "what is the name of X?" and "what is the value of X?" patterns in addition to existing panel/component patterns.
- **Visual-reasoning depth test:** `_requires_complex_visual_reasoning()` checks if question requires multi-panel comparison, spatial relationships, or calculation. Warns for Challenging/Hardest questions that don't meet threshold.
- **Auto-classification:** Regeneration now auto-classifies difficulty based on model rollout results (Qwen/Gemini pass rates). Overrides requested difficulty if significant mismatch detected.
- **TaskEvent audit trail:** Every task action (regeneration, update, difficulty change, Rhea review, issue report, AI fix, submit, delete) is logged to `task_events` table with JSON details. Used by the Task History UI and injected as context during regeneration.
- **Task state injection in regenerate:** Regeneration prompt now includes current task state (question, answer, difficulty, status), Rhea review results, and model performance metrics (Qwen/Gemini pass rates) as additional context.
- **AI Fix with image context:** The AI Fix endpoint now sends the figure image, figure history, and validation context to the LLM for context-aware fixes.
- **Check Answer (VLM verification):** "Check Answer" button sends the task image + question to `mimo-v2.5` (VLM), then verifies the VLM's answer against the golden answer using `mimo-v2.5` (text-only) for unbiased semantic equivalence. Result displayed inline and logged as `TaskEvent(event_type="check_answer")`. Helps authors discover if a task is answerable and whether it's challenging enough to stump VLMs.
- **Premise fact-check gate:** Every regenerate draft that passes text validation is run through an adversarial VLM fact-checker (`FACT_CHECK_PROMPT`): each factual claim is marked SUPPORTED / NOT_SUPPORTED / UNVERIFIABLE and any non-SUPPORTED claim rejects the draft. Failures feed back into the retry loop as feedback; fact-failed drafts are stored in Task History (restorable) but never auto-saved. Fail-open on checker tooling errors. Stored in `generation_attempts.fact_check_errors`.
- **Answer determinism gate:** Challenging/Hardest drafts must also pass a 3-run sampled determinism check (minimax-m3 reads the question independently; every read must match the golden answer — numeric with relative tolerance, words exact + semantic fallback). This is the machine proof of the "two readers give the same answer" rule; diverging reads reject the draft and feed back into regeneration. Stored in `generation_attempts.determinism_errors`. Standalone: `POST /api/task/{id}/determinism-check` button + `arxiv-manager task determinism <id>`.
- **Dual-target Hardest prompts:** HARDEST means BOTH `openrouter/qwen/qwen3.6-35b-a3b` AND `google/gemini-3.5-flash` must fail. `_model_cards.py` holds verified capability cards (Qwen: strong OCR/diagrams/math 82–93, weak ODInW13 counting 50.8 / ZEROBench novel formats 34.4 / RefSpatial 64.3; Gemini: near-Pro reasoner — beatable only via perception failures). HARDEST_PROMPT + SELF_CRITIQUE_PROMPT enforce the both-fail rubric with FAILS-BOTH avoid/prefer lists.
- **Difficulty-aware few-shot ordering:** `get_few_shot_examples` prefers both-model-failed attempts (`qwen_passes=0 AND gemini_passes=0`) for Hardest, and Qwen-failed/Gemini-passed attempts for Challenging.
- **Realm verdict ingestion:** `arxiv-manager task verdict <id> --verdict too_easy|too_hard|approved` (or `POST /api/task/{id}/verdict`) records the outcome on the latest SubmissionLog and auto-adjusts difficulty one tier (warn-only). Logged as `TaskEvent(realm_verdict)` and consumed by strategy analytics.
- **Strategy analytics:** `/analytics/strategies` aggregates auto-classified question strategies (counting, comparison, rank, cross_panel_sum_diff, spatial, percentage_change, single_lookup, other) × every verdict signal (Realm verdicts, manual pass counts, check-answer, determinism) via a pluggable data-provider seam (`analytics/strategies.task_verdict_sources`) for a future rollout engine.
- **Multi-agent orchestration:** Orchestrator plans subtasks → delegates to Generator → delegates to Reviewer → aggregates results. Uses `AgentContext` for shared state and delegation chains.
- **DB-backed task scheduling:** Jobs are enqueued to `scheduled_tasks` table, picked up by a subprocess worker. Priority-based FIFO with automatic retry. No external dependencies (no Redis/Celery).
- **Subprocess worker isolation:** Worker runs in a separate Python process via `subprocess.Popen`, communicates via shared SQLite (WAL mode). Sentinel file for graceful shutdown on all platforms.
- **Token-based authentication:** PBKDF2-SHA256 password hashing, UUID tokens stored in DB. `AuthMiddleware` checks `Authorization: Bearer` or `X-API-Key` headers.
- **User personalization:** `UserProfile` (model preference, difficulty, prompt style) + key-value `UserPreference` applied to routing configs via `personalizer.apply_preferences()`.
- **Local CNN vision:** ResNet-18 feature extractor (lazy-loaded, ~44MB). Falls back to heuristic classifier when torch/torchvision unavailable. Prototype-based few-shot classification.
- **Agent registry:** Dict-based lookup by name or capability. Used by orchestrator to discover available agents and their capabilities.

## Key Files

| File | Purpose |
|------|---------|
| `run.py` | FastAPI entry point |
| `src/arxiv_manager/web/app.py` | App factory, route registration, logging, rate limiting, MCP, auth middleware |
| `src/arxiv_manager/authoring/ai_draft/core.py` | Core generation pipeline with RAG injection |
| `src/arxiv_manager/authoring/ai_draft/_fact_checker.py` | Adversarial premise fact-check (`fact_check_draft`) |
| `src/arxiv_manager/authoring/ai_draft/_determinism.py` | 3-run sampled answer determinism (`check_determinism_for_qa`) |
| `src/arxiv_manager/authoring/_model_cards.py` | Verified Qwen/Gemini capability cards (HF/OpenRouter) |
| `src/arxiv_manager/analytics/strategies.py` | Strategy-class × verdict aggregation + provider seam |
| `src/arxiv_manager/authoring/ai_draft/_api_client.py` | LLM API client (OpenCode), token usage capture |
| `src/arxiv_manager/authoring/ai_draft/_image_utils.py` | Shared image encoding for LLM consumption (base64 JPEG) |
| `src/arxiv_manager/authoring/_history_context.py` | History injection (build_task_history + build_figure_history), few-shot, model selection, task state injection |
| `src/arxiv_manager/authoring/_validation_helpers.py` | Validation patterns: generic count, manufactured difficulty, chart anti-patterns, reasoning indicators |
| `src/arxiv_manager/authoring/validator.py` | Handbook validation rules (now includes manufactured difficulty check) |
| `src/arxiv_manager/authoring/_guardrails.py` | Quality guardrails |
| `src/arxiv_manager/components/hybrid_retriever.py` | ChromaDB + sentence-transformers hybrid search |
| `src/arxiv_manager/services/rag_pipeline.py` | CRAG orchestrator (cache → retrieve → rerank) |
| `src/arxiv_manager/observability/tracer.py` | Structured JSON logging, trace spans |
| `src/arxiv_manager/observability/cost_tracker.py` | Token/cost estimation per model |
| `src/arxiv_manager/security/input_guard.py` | Prompt injection detection |
| `src/arxiv_manager/prompts/__init__.py` | DB-backed prompt registry with versioning |
| `src/arxiv_manager/mcp/__init__.py` | MCP server with 6+ tools |
| `src/arxiv_manager/web/routes/health.py` | Health check endpoint |
| `src/arxiv_manager/web/routes/prompt_routes.py` | Prompt management API |
| `src/arxiv_manager/agents/registry.py` | Agent capability registry |
| `src/arxiv_manager/agents/context.py` | AgentContext — shared state across agents |
| `src/arxiv_manager/agents/orchestrator.py` | Multi-agent orchestrator |
| `src/arxiv_manager/agents/reviewer.py` | Draft reviewer agent |
| `src/arxiv_manager/scheduler/models.py` | ScheduledTask model |
| `src/arxiv_manager/scheduler/queue.py` | DB-backed FIFO job queue |
| `src/arxiv_manager/scheduler/worker.py` | Subprocess worker entry point |
| `src/arxiv_manager/scheduler/manager.py` | Subprocess lifecycle (start/stop) |
| `src/arxiv_manager/scheduler/routes.py` | Scheduler API endpoints |
| `src/arxiv_manager/personalization/auth.py` | Password hashing + token auth |
| `src/arxiv_manager/personalization/middleware.py` | FastAPI auth middleware |
| `src/arxiv_manager/personalization/personalizer.py` | User preference application |
| `src/arxiv_manager/personalization/routes.py` | Auth + profile API endpoints |
| `src/arxiv_manager/vision/models.py` | Lazy ResNet-18 loader |
| `src/arxiv_manager/vision/extractor.py` | 512-dim feature extraction |
| `src/arxiv_manager/vision/classifier.py` | Figure type classification |
| `evaluation/golden_dataset.json` | Golden dataset of 10 Q&A pairs (3 verified against real images, determinism bar) |
| `evaluation/offline_eval.py` | Offline evaluation harness (validator + optional determinism runs) |

## New DB Tables

| Table | Module | Purpose |
|-------|--------|---------|
| `scheduled_tasks` | `scheduler.models` | DB-backed async job queue |
| `users` | `personalization.models` | User accounts |
| `auth_tokens` | `personalization.models` | Bearer token storage |
| `user_profiles` | `personalization.models` | User generation preferences |
| `user_preferences` | `personalization.models` | Key-value preference pairs |
| `task_events` | `models.py` | Unified audit trail for all task state changes (regeneration, update, difficulty_change, rhea_review, issue_report, ai_fix, submit, delete, restore, check_answer, determinism_check, realm_verdict) |

## Key DB Columns (migrations in `db.py`)

| Column | Table | Purpose |
|--------|-------|---------|
| `fact_check_errors` | `generation_attempts` | JSON list of unsupported premise claims (fact-check gate) |
| `determinism_errors` | `generation_attempts` | JSON list of sampled answers that diverged from the golden |
| `review_status` | `submission_logs` | `pending \| approved \| rework \| too_easy \| too_hard` (Realm verdicts via `record_realm_verdict`)

## Storage Layout

```
storage/
├── arxiv-manager.db        # SQLite database
├── figures/                # Extracted figure images
├── papers/                 # Downloaded PDFs
├── _uploads/               # User-uploaded images
├── _draft_telemetry.jsonl  # Generation telemetry
├── _structured_log.jsonl   # Structured JSON logs
└── chroma_db/              # Vector index (ChromaDB, Phase 3)
```

## Available MCP Tools (GET /mcp/tools)

| Tool | Description |
|------|-------------|
| `generate_qa` | Generate visual-reasoning Q&A from an image |
| `validate_qa` | Validate Q&A against handbook rules |
| `figure_history` | Get past generation attempts for a figure |
| `search_figures` | Search indexed figures by caption |
| `health` | Get system health status |
| `analytics` | Get generation performance stats |

## Conventions

- `from __future__ import annotations` in all Python files
- Type hints everywhere
- Docstrings on all public functions (no inline comments)
- Private modules/functions prefixed with `_`
- Tests in `tests/` mirroring source structure
- All new modules under `src/arxiv_manager/`
- Heavy deps (sentence-transformers, torch) use lazy imports
