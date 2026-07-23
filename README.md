# ArXiv Manager

AI-powered assistant for creating challenging visual-reasoning Q&A tasks from scientific figures.

**Live app:** [arxiv-manager.onrender.com](https://arxiv-manager.onrender.com)  
**Landing page:** [huggingface.co/spaces/fritzelborges/atmydesk](https://huggingface.co/spaces/fritzelborges/atmydesk)

## Features

- **Image Upload** — Drag, drop, or paste images; get instant suitability analysis
- **arXiv Search** — Search CC0 papers, extract figures, pick the best ones
- **AI Drafting** — Generate Q&A drafts at Easy, Challenging, or HARDEST difficulty
- **Task Management** — Create, edit, validate, and submit tasks
- **Dashboard** — Track pipeline progress and AI draft performance

## Tech Stack

- **Backend:** FastAPI + SQLModel + SQLite
- **Frontend:** HTMX + Tailwind CSS (mobile-responsive)
- **Deployment:** Render (backend) + HuggingFace Spaces (landing page)

## Quick Start

```bash
pip install -r requirements.txt
uvicorn run:app --host 0.0.0.0 --port 8000
```

## Usage

1. **Upload** an image or search arXiv to find figures
2. **Analyze** — the system checks suitability and recommends difficulty
3. **Generate** an AI draft at the recommended difficulty
4. **Edit** the question, answer, format, and type
5. **Propose** as a task — it's saved to the database
6. **Validate** and **submit** from the Tasks page

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENCODE_API_KEY` | For AI drafting | OpenCode Go API key |
| `OPENAI_API_KEY` | Optional | OpenAI API key (alternative provider) |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key (alternative provider) |
