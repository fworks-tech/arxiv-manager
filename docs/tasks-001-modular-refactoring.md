# Tasks: Modular Refactoring of Integrator Modules

Three independent branches, each with its own task list. Branches can be executed in any order.

---

## Branch 1: `refactor/cli-modules`

**Concern:** Split `cli.py` into domain-cohesive sub-modules under `cli/` package.
**Risk:** Low — pure CLI commands, no runtime impact, easy to verify.

### Tasks

- [ ] **build(cli): create cli/ directory and __init__.py skeleton**
  - Create `src/arxiv_manager/cli/` directory
  - Create `cli/__init__.py` with `from ..db import init_db` and `app` creation, `main()` callback, `web_server()` command
  - Register all 5 sub-typers (`search_app`, `task_app`, `images_app`) in `__init__.py`
  - Import sub-module files (to be created next) so Typer decorators fire
  - Add `from arxiv_manager.cli import app` re-export in original `cli.py` (shim)
  - **Acceptance:** `arxiv-manager --help` shows all command groups

- [ ] **feat(cli): extract check command to cli/check.py**
  - Create `cli/check.py` with `check_api()` function and `@app.command()` decorator
  - Move the `check_api()` function body verbatim (lines 37–118 from original)
  - **Acceptance:** `arxiv-manager check` runs without error

- [ ] **feat(cli): extract search commands to cli/search_commands.py**
  - Create `cli/search_commands.py` with `@search_app.command()` decorators
  - Move `search_papers()`, `fetch_paper()`, `fetch_many()` function bodies verbatim
  - Preserve all typer options/arguments
  - **Acceptance:** `arxiv-manager search papers --help`, `arxiv-manager search fetch` work

- [ ] **feat(cli): extract image commands to cli/image_commands.py**
  - Create `cli/image_commands.py` with `@images_app.command()` decorators
  - Move `list_images()`, `audit_images()`, `clean_images()`, `reclassify_images()`, `rescore_images()`, `rank_images()` verbatim
  - **Acceptance:** `arxiv-manager images list --help`, `arxiv-manager images audit` work

- [ ] **feat(cli): extract task commands to cli/task_commands.py**
  - Create `cli/task_commands.py` with `@task_app.command()` decorators
  - Move `create_task()`, `validate_existing()`, `list_tasks()`, `create_task_batch()` verbatim
  - **Acceptance:** `arxiv-manager task new --help`, `arxiv-manager task list` work

- [ ] **feat(cli): extract admin commands to cli/admin_commands.py**
  - Create `cli/admin_commands.py` with `@task_app.command()` decorators
  - Move `set_diff()`, `export_task_cmd()`, `stats()`, `submit_task()`, `analytics()` verbatim
  - **Acceptance:** `arxiv-manager task analytics`, `arxiv-manager task stats` work

- [ ] **test(cli): verify all imports resolve and CLI boots**
  - Run `python -c "from arxiv_manager.cli import app"` — no import errors
  - Run `arxiv-manager --help` — all command groups visible
  - Run full test suite — 211 tests pass

---

## Branch 2: `refactor/routes-modules`

**Concern:** Split `web/routes.py` into domain-cohesive route files.
**Risk:** Medium — routes are the web API surface; any import error causes 500s.

### Tasks

- [ ] **feat(routes): create web/pages.py with all page GET handlers**
  - Create `web/pages.py` with its own `router = APIRouter()`
  - Move `index()`, `images_page()`, `tasks_page()`, `task_form()`, `task_detail()`, `stats_page()`, `author_page()`, `metrics_page()` verbatim
  - **Acceptance:** All GET pages load without error

- [ ] **feat(routes): create web/author_routes.py with upload + draft endpoints**
  - Create `web/author_routes.py` with its own `router = APIRouter()`
  - Move `_save_upload()`, `api_upload_image()`, `api_draft_qa()`, `api_discard_image()`, `api_propose_task()` verbatim
  - **Acceptance:** POST /api/image/upload, POST /api/image/draft work

- [ ] **feat(routes): create web/arxiv_routes.py with search + extract endpoints**
  - Create `web/arxiv_routes.py` with its own `router = APIRouter()`
  - Move `api_arxiv_search()`, `api_arxiv_extract()` verbatim
  - **Acceptance:** GET /api/arxiv/search, POST /api/arxiv/extract work

- [ ] **feat(routes): create web/task_routes.py with task CRUD endpoints**
  - Create `web/task_routes.py` with its own `router = APIRouter()`
  - Move `api_create_task()`, `api_update_task()`, `api_validate_task()`, `api_regenerate_task()`, `api_generation_history()` verbatim
  - **Acceptance:** POST /api/task/create, POST /api/task/{id}/regenerate work

- [ ] **feat(routes): create web/lifecycle_routes.py with lifecycle endpoints**
  - Create `web/lifecycle_routes.py` with its own `router = APIRouter()`
  - Move `update_figure_status()`, `bulk_reject_figures()`, `update_task_difficulty()`, `submit_task()`, `update_rhea()`, `save_rhea_override()` verbatim
  - **Acceptance:** POST /api/figure/{id}/status, POST /api/task/{id}/submit work

- [ ] **feat(routes): create web/metrics.py with telemetry computation**
  - Create `web/metrics.py`
  - Move `_compute_metrics()` verbatim
  - **Acceptance:** GET /metrics loads correctly

- [ ] **fix(routes): update web/app.py to aggregate sub-routers**
  - Replace `from .routes import router` with imports from all 6 sub-modules
  - Call `app.include_router()` for each sub-router
  - Keep `routes.py` as a shim until callers migrate
  - **Acceptance:** Server starts without import errors, all routes respond

- [ ] **test(routes): verify all endpoints respond correctly**
  - Run full test suite — 211 tests pass
  - Run `python -c "from arxiv_manager.web.routes import router"` — shim works

---

## Branch 3: `refactor/ai-draft-modules`

**Concern:** Split `ai_draft.py` into focused sub-modules and fix circular dependency with `_guardrails`.
**Risk:** High — core generation logic; any import error breaks all drafting.

### Tasks

- [ ] **feat(ai-draft): create authoring/ai_draft/ package and _response_parser.py**
  - Create `authoring/ai_draft/` directory
  - Create `authoring/ai_draft/_response_parser.py`
  - Move `_extract_reasoning()`, `_parse_llm_response()`, `_parse_critique_response()` verbatim
  - **Acceptance:** Parser tests pass; `from arxiv_manager.authoring.ai_draft._response_parser import _parse_llm_response` works

- [ ] **feat(ai-draft): create _api_client.py with API call logic**
  - Create `authoring/ai_draft/_api_client.py`
  - Move `_get_api_key()`, `_call_opencode()` verbatim
  - Update `_call_opencode` to import `_parse_llm_response` from `._response_parser`
  - **Acceptance:** API client imports resolve; existing mocks (mock_draft_success) still work

- [ ] **feat(ai-draft): create _verifier.py with verify_draft**
  - Create `authoring/ai_draft/_verifier.py`
  - Move `verify_draft()` verbatim
  - Update imports to use `._api_client._call_opencode` and `._response_parser._parse_llm_response`
  - **Acceptance:** `verify_draft()` works with existing test fixtures

- [ ] **feat(ai-draft): create core.py with draft_qa**
  - Create `authoring/ai_draft/core.py`
  - Move `draft_qa()` verbatim
  - Update imports to use new sub-modules
  - **Acceptance:** `draft_qa()` works; guardrails integration intact

- [ ] **feat(ai-draft): create composition.py with consensus + self-critique**
  - Create `authoring/ai_draft/composition.py`
  - Move `draft_qa_consensus()`, `draft_with_self_critique()` verbatim
  - Update imports to use `core.draft_qa`, `_api_client._call_opencode`, `_response_parser._parse_critique_response`
  - **Acceptance:** Self-critique and consensus flows work

- [ ] **fix(guardrails): break circular dependency with callback injection**
  - Modify `run_guardrails()` in `_guardrails.py` to accept `draft_qa_callback=None` parameter
  - Replace `from .ai_draft import draft_qa` (lazy import) with callback usage
  - Update `core.py:draft_qa()` to pass `draft_qa_callback=self` when calling `run_guardrails`
  - **Acceptance:** `from arxiv_manager.authoring._guardrails import run_guardrails` does not trigger ai_draft import

- [ ] **fix(ai-draft): create __init__.py shim for backward compatibility**
  - Create `authoring/ai_draft/__init__.py`
  - Re-export: `from .core import draft_qa`, `from .composition import draft_qa_consensus, draft_with_self_critique`, `from ._verifier import verify_draft`, `from ._api_client import _get_api_key, _call_opencode`, `from ._response_parser import _parse_llm_response, _parse_critique_response, _extract_reasoning`
  - Remove original `ai_draft.py` file
  - **Acceptance:** All existing imports resolve: `from arxiv_manager.authoring.ai_draft import draft_qa`, etc.

- [ ] **test(ai-draft): verify all imports resolve and tests pass**
  - Run `python -c "from arxiv_manager.authoring.ai_draft import draft_qa, draft_qa_consensus, draft_with_self_critique, verify_draft"`
  - Run full test suite — 211 tests pass
  - Run `python -c "from arxiv_manager.authoring._guardrails import run_guardrails"` — no ai_draft side effects
