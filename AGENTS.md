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
│   ├── _draft_prompts.py   # 11 prompt templates with SHA-256 versioning
│   ├── _draft_telemetry.py # Generation attempt logging to DB + JSONL
│   ├── _guardrails.py      # Quality checks with auto-retry + feedback
│   ├── _history_context.py # History injection, few-shot, model selection
│   ├── validator.py        # 28+ handbook rule validation
│   └── image_analyzer.py   # Figure suitability classification
├── cli/                    # CLI commands (search, task, images, web, check, analytics, index)
├── sourcing/               # arXiv PDF download, figure extraction, filtering
├── web/                    # FastAPI app + 15 Jinja2/HTMX templates
│   ├── routes/             # Route handlers (task, author, arxiv, lifecycle, metrics, health, prompts)
│   └── templates/          # HTML templates with HTMX partials
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
- **GenerationAttempt:** Full trace of every LLM call (38+ fields incl. token usage, Rhea feedback, model run results)
- **IssueReport:** User-reported issues on generation attempts (reason, description, corrected_answer) — fed back into generation prompts
- **PromptTemplateRecord:** DB-backed prompt templates with versioning, rollback support
- **SubmissionLog:** Task submission tracking with review status

## Key Patterns

- **CRAG (CAG + RAG):** Semantic cache check first, then hybrid retrieve + rerank
- **History injection:** Past attempts are injected as few-shot examples + "don't repeat" guidance
- **Self-critique loop:** LLM scores own draft (1-5) and rewrites, now with 2 rounds
- **Consensus drafting:** Multiple drafts → validate → pick best
- **Guardrails + auto-retry:** Quality checks trigger regeneration with feedback
- **Dynamic model selection:** Best model per (figure_type, difficulty) from historical quality
- **Hot-swappable prompts:** DB-backed templates with API for save/rollback/reload
- **Cost tracking:** Token usage captured from API responses, cost estimated per model
- **IssueReport feedback loop:** User-reported issues (too_easy, wrong_answer) and corrected answers are injected into the generation prompt via `build_figure_history`. Model run results (Qwen/Gemini passes) are copied onto GenerationAttempt on submit and influence few-shot ordering.
- **Arithmetic consistency check:** Validation warns when a question asks for product/sum but the answer may be a simple count.
- **AI Fix with image context:** The AI Fix endpoint now sends the figure image, figure history, and validation context to the LLM for context-aware fixes.
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
| `src/arxiv_manager/authoring/ai_draft/_api_client.py` | LLM API client (OpenCode), token usage capture |
| `src/arxiv_manager/authoring/_history_context.py` | History injection, few-shot, model selection |
| `src/arxiv_manager/authoring/validator.py` | Handbook validation rules |
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
| `evaluation/golden_dataset.json` | Golden dataset of 7 proven Q&A pairs |
| `evaluation/offline_eval.py` | Offline evaluation harness |

## New DB Tables

| Table | Module | Purpose |
|-------|--------|---------|
| `scheduled_tasks` | `scheduler.models` | DB-backed async job queue |
| `users` | `personalization.models` | User accounts |
| `auth_tokens` | `personalization.models` | Bearer token storage |
| `user_profiles` | `personalization.models` | User generation preferences |
| `user_preferences` | `personalization.models` | Key-value preference pairs |

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
