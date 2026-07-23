# ArXiv Manager

AI-powered assistant for creating challenging visual-reasoning Q&A tasks from scientific figures.

**Live app:** [arxiv-manager.onrender.com](https://arxiv-manager.onrender.com)

## Features

- **Image Upload** — Drag, drop, or paste images; auto-analyze for suitability, complexity score, and figure type
- **arXiv Search** — Search CC0 papers from S3 bucket, extract figures, audit & filter by complexity
- **AI Drafting** — Generate Q&A drafts at Easy, Challenging, or HARDEST difficulty
- **Task Management** — Full pipeline: draft → proposed → validated → submitted
- **Dashboard** — Pipeline metrics, per-provider draft performance, task status breakdown

## Tech Stack

- **Backend:** FastAPI + SQLModel + SQLite
- **Frontend:** HTMX (partial) + Tailwind CSS (mobile-responsive); AJAX via native fetch()
- **Deployment:** Render

## Quick Start

```bash
pip install -r requirements.txt
uvicorn run:app --host 0.0.0.0 --port 8000
```

## Usage

1. **Upload** an image or search arXiv to find figures
2. **Auto-analyze** — system checks suitability, complexity score, and figure type
3. **Draft** — generate AI Q&A at Easy, Challenging, or HARDEST difficulty; override the recommended level
4. **Edit** — refine question, answer, format, and type
5. **Propose** — save as a task in the database
6. **Validate & Submit** — validate and submit from the Tasks page
