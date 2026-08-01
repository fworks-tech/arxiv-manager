# ArXiv Manager

AI-powered assistant for creating challenging visual-reasoning Q&A tasks from scientific figures. Features CRAG architecture (CAG + RAG), production observability, hot-swappable prompts, and MCP integration.

## Features

- **Image Upload** — Drag, drop, or paste images; auto-analyze for suitability, complexity score, and figure type
- **arXiv Search** — Search CC0 papers from S3 bucket, extract figures, audit & filter by complexity
- **Task Management** — Full pipeline: draft → proposed → validated → submitted
- **Dashboard** — Pipeline metrics, per-provider draft performance, task status breakdown, cost tracking
- **Health Check** — `GET /health` endpoint for monitoring (DB, API key, LLM connectivity)
- **Multi-Agent Orchestration** — Orchestrator plans subtasks, delegates to Generator and Reviewer agents, aggregates results via weighted voting
- **Token Authentication** — PBKDF2-SHA256 password hashing, Bearer token auth, `AuthMiddleware` on all endpoints
- **User Personalization** — User profiles with model/difficulty/prompt-style preferences, key-value preference pairs
- **Async Job Queue** — DB-backed FIFO queue with subprocess worker, priority ordering, automatic retry
- **Local CNN Vision** — Lazy-loaded ResNet-18 feature extractor with prototype-based figure classification (falls back to heuristic)

### Smart Generation Pipeline

- **AI Drafting** — Generate Q&A at Easy, Challenging, or HARDEST with auto-retry on low quality
- **Self-Critique Loop** — For Challenging/HARDEST difficulties, the model critiques its own draft and rewrites if needed
- **Consensus Drafting** — Multiple independent drafts generated, validated, best one selected
- **CRAG (RAG + CAG)** — Semantic cache check first, then hybrid keyword + vector retrieve, cross-encoder rerank
- **Prompt Versioning** — 11 prompt templates, each content-addressed (SHA-256); every generation records which version produced it
- **Hot-Swappable Prompts** — DB-backed prompt registry; save, rollback, reload via API without code changes
- **Task History** — Every draft, critique, regeneration, update, difficulty change, Rhea review, issue report, and AI fix is logged with full context in a unified audit trail
- **Guardrails** — Answer plausibility, format validation, diversity checks, and quality thresholds catch bad outputs before they reach the user
- **Few-Shot Learning** — Past successful generations (quality >= 80) are injected as examples into new prompts, improving quality over time
- **Query Router** — Routes requests to optimal pipeline based on difficulty (simple / RAG-enhanced / self-critique / consensus)
- **Dynamic Model Selection** — System tracks which model performs best per figure type and difficulty, auto-selects the best one
- **Cost Tracking** — Token usage captured from every API call, cost estimated per model
- **CLI Analytics** — `arxiv-manager task analytics` surfaces success rates, best configurations, and common validation errors

### Observability

- **Structured JSON Logging** — Machine-parseable logs with trace IDs and extra fields, written to `storage/_structured_log.jsonl`
- **Trace Spans** — Per-request timing via `with span("operation_name"):` context manager
- **Metrics Dashboard** — `GET /metrics` shows success rate, latency, cost breakdown by difficulty/figure type
- **Generation Telemetry** — Every LLM call recorded in `GenerationAttempt` DB table (38+ fields) and JSONL

### Security

- **Input Guard** — Prompt injection detection, sensitive data pattern blocking
- **Content Filter** — PII detection (email, phone, SSN), toxicity filtering
- **Output Filter** — System prompt leakage prevention
- **Rate Limiting** — IP-based middleware (120 req/min)

### MCP Integration

- **8 MCP Tools** — `generate_qa`, `validate_qa`, `figure_history`, `search_figures`, `health`, `analytics`, `orchestrate`, `enqueue_generation`
- **Auto-discoverable** via `GET /mcp/tools`
- **Callable** via `POST /mcp/tools/{name}/call`

### Authentication

- **Token-based auth** via `AuthMiddleware` on all non-public routes
- **Public routes**: `/auth/login`, `/auth/register`, `/health`, `/mcp/*`, `/docs`
- **Register**: `POST /auth/register` with `{"username", "password"}`
- **Login**: `POST /auth/login` returns `{"token", "user_id", "username", "role"}`
- **Authenticate**: Pass `Authorization: Bearer <token>` or `X-API-Key` header

## Tech Stack

- **Language:** Python 3.11+
- **Web:** FastAPI + Jinja2 + HTMX + Tailwind CSS
- **Database:** SQLite + SQLModel (ORM)
- **CLI:** Typer + Rich
- **LLM:** OpenCode API (OpenAI-compatible chat completions)
- **PDF/Images:** PyMuPDF, Pillow, ImageHash
- **RAG:** ChromaDB + sentence-transformers (all-MiniLM-L6-v2) + LangChain
- **Vision:** ResNet-18 (lazy-loaded, optional) + torchvision
- **Auth:** PBKDF2-SHA256 password hashing, UUID Bearer tokens
- **Scheduler:** DB-backed job queue + subprocess worker (no external deps)
- **MCP:** FastAPI-based MCP server with 8 tools

## Quick Start

```bash
pip install -r requirements.txt -r requirements-dev.txt
python run.py
```

Environment variables:

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENCODE_API_KEY` | Yes | LLM API key for draft generation |
| `HF_TOKEN` | No | Hugging Face token — set it to raise rate limits and speed up first-time model downloads (embedding model + cross-encoder) |

Development mode:
```bash
uvicorn src.arxiv_manager.web.app:create_app --reload --reload-exclude storage/ --host 0.0.0.0 --port 8000
```

## Testing

```bash
python -m pytest tests/ -v --tb=short                       # All tests
python -m pytest tests/ --cov=src/arxiv_manager              # With coverage
python -m evaluation.offline_eval --quick                    # Offline eval harness
```

## CLI Commands

```bash
# Pre-flight health check
arxiv-manager check

# Search and fetch papers from arXiv
arxiv-manager search papers --terms "neural network" --domain "Computer Science"
arxiv-manager search fetch <paper-id>
arxiv-manager search fetch-many --limit 10 --domain "Computer Science"

# Manage tasks
arxiv-manager task new --image-id 42 --challenging
arxiv-manager task new-batch --limit 5 --difficulty hardest
arxiv-manager task list
arxiv-manager task validate <task-id>
arxiv-manager task analytics
arxiv-manager task determinism <task-id> --runs 3   # 3 sampled VLM reads must match the golden answer
arxiv-manager task verdict <task-id> --verdict too_easy   # record Realm outcome (too_easy|too_hard|approved)

# Image library management
arxiv-manager images list
arxiv-manager images audit
arxiv-manager images clean
arxiv-manager images index    # Batch-index figures into ChromaDB for RAG

# Web UI
arxiv-manager web
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (DB + API key) |
| GET | `/health?full=true` | Full health check (includes LLM) |
| GET | `/metrics` | AI draft performance dashboard |
| GET | `/mcp/tools` | List all MCP tools |
| POST | `/mcp/tools/{name}/call` | Call an MCP tool |
| GET | `/api/prompts` | List prompt templates |
| POST | `/api/prompts/{name}/save` | Save a new prompt version |
| POST | `/api/prompts/{name}/rollback` | Rollback to a previous version |
| POST | `/api/prompts/reload` | Force-reload prompts from DB |
| GET | `/auth/register` | Register a user |
| POST | `/auth/login` | Login and receive Bearer token |
| GET | `/auth/profile` | Get user profile (auth required) |
| PUT | `/auth/profile` | Update user profile (auth required) |
| GET | `/auth/preferences` | Get user preferences (auth required) |
| PUT | `/auth/preferences/{key}` | Set a preference value (auth required) |
| GET | `/api/scheduler/queue` | View job queue depth and list |
| POST | `/api/scheduler/enqueue` | Enqueue a generation job |
| GET | `/api/scheduler/status/{id}` | Poll job status |
| POST | `/api/scheduler/cancel/{id}` | Cancel a queued job |
| GET | `/api/scheduler/worker` | Check worker subprocess status |
| GET | `/` | Dashboard |
| POST | `/api/image/upload` | Upload an image |
| POST | `/api/image/draft` | Generate AI draft |
| POST | `/api/task/{id}/regenerate` | Regenerate with self-critique |
| GET | `/api/task/{id}/task-history` | Unified task history (generations + events) |
| POST | `/api/task/{id}/delete` | Delete a task (cascades to related records) |
| POST | `/api/task/{id}/check-answer` | Send image + question to a VLM, verify against golden |
| POST | `/api/task/{id}/determinism-check` | 3 sampled VLM reads must all match the golden answer |
| POST | `/api/task/{id}/verdict` | Record Realm verdict (too_easy / too_hard / approved) |
| GET | `/analytics/strategies` | Strategy-class × model-verdict dashboard |

Full API docs at `/docs` (Swagger) when the server is running.

## Architecture

```
src/arxiv_manager/
├── authoring/           # AI pipeline: prompts, validation, guardrails, telemetry
├── agents/              # Agent registry, orchestrator, reviewer, adaptive router, tools
├── scheduler/           # DB-backed async job queue + subprocess worker
├── personalization/     # User accounts, auth, profiles, preferences
├── vision/              # Local ResNet-18 CNN figure classification (lazy-loaded)
├── cli/                 # CLI commands (search, task, images, web, check, analytics, index)
├── sourcing/            # arXiv PDF download, figure extraction, filtering
├── web/                 # FastAPI app + templates + AuthMiddleware
├── components/          # RAG: hybrid retriever (ChromaDB + sentence-transformers), reranker
├── services/            # CRAG: RAG pipeline, semantic cache, query router
├── observability/       # Structured logs, trace spans, cost tracker
├── security/            # Input guard, content filter, output filter
├── prompts/             # Hot-swappable prompt registry (DB-backed)
├── mcp/                 # MCP server (8 tools)
├── tracking/            # Task export, platform submission
├── analytics/           # Strategy × model-verdict aggregation
├── evaluation/          # Golden dataset, offline eval script
├── models.py            # DB models
├── db.py                # SQLite engine + migrations + WAL mode
└── storage.py           # Centralized storage paths
```

## Storage Layout

```
storage/
├── arxiv-manager.db        # SQLite database
├── figures/                # Extracted figure images
├── papers/                 # Downloaded PDFs
├── _uploads/               # User-uploaded images
├── _draft_telemetry.jsonl  # Generation telemetry
├── _structured_log.jsonl   # Structured JSON logs
└── chroma_db/              # Vector index (ChromaDB)
```
