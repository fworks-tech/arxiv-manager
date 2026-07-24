# ArXiv Manager

AI-powered assistant for creating challenging visual-reasoning Q&A tasks from scientific figures.

## Features

- **Image Upload** — Drag, drop, or paste images; auto-analyze for suitability, complexity score, and figure type
- **arXiv Search** — Search CC0 papers from S3 bucket, extract figures, audit & filter by complexity
- **Task Management** — Full pipeline: draft → proposed → validated → submitted
- **Dashboard** — Pipeline metrics, per-provider draft performance, task status breakdown

### Smart Generation Pipeline

- **AI Drafting** — Generate Q&A at Easy, Challenging, or HARDEST with auto-retry on low quality
- **Self-Critique Loop** — For Challenging/HARDEST difficulties, the model critiques its own draft and rewrites if needed
- **Prompt Versioning** — Every prompt template is content-addressed (SHA-256); each generation records which version produced it
- **Generation History** — Every draft, critique, and regeneration is logged with reasoning traces, validation scores, and prompt version info — viewable on the task edit page
- **Guardrails** — Answer plausibility, format validation, diversity checks, and quality thresholds catch bad outputs before they reach the user
- **Few-Shot Learning** — Past successful generations are injected as examples into new prompts, improving quality over time
- **Dynamic Model Selection** — System tracks which model performs best per figure type and difficulty, auto-selects the best one
- **CLI Analytics** — `arxiv-manager task analytics` surfaces success rates, best configurations, and common validation errors

## Tech Stack

- **Backend:** FastAPI + SQLModel + SQLite
- **Frontend:** HTMX (partial) + Tailwind CSS (mobile-responsive); AJAX via native fetch()
- **Deployment:** Render

## Quick Start

```bash
pip install -r requirements.txt
python -m uvicorn src.arxiv_manager.web.app:create_app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. **Upload** an image or search arXiv to find figures
2. **Auto-analyze** — system checks suitability, complexity score, and figure type
3. **Draft** — generate AI Q&A at Easy, Challenging, or HARDEST difficulty; override the recommended level
4. **Edit** — refine question, answer, format, and type
5. **Propose** — save as a task in the database
6. **Validate & Submit** — validate and submit from the Tasks page
7. **Review Generation History** — on the task edit page, click "Generation History" to see all past drafts, critiques, and regenerations with quality scores and reasoning traces
8. **CLI Analytics** — run `arxiv-manager task analytics` to see pipeline performance metrics

## CLI Commands

```bash
# Check API and database health
arxiv-manager check

# Search and fetch papers from arXiv
arxiv-manager search papers --terms "neural network" --domain "Computer Science"
arxiv-manager search fetch <paper-id>

# Manage tasks
arxiv-manager task new --image-id 42 --challenging
arxiv-manager task list
arxiv-manager task analytics

# Web UI
arxiv-manager web
```
