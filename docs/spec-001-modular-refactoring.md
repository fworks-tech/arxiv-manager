# Spec: Modular Refactoring of Integrator Modules

## Problem

Three modules (`cli.py`, `web/routes.py`, `authoring/ai_draft.py`) exceed the Warden's maintainability thresholds at 1,264, 1,066, and 652 lines respectively. They contain functions over 80 lines, cyclomatic complexity over 20, and parameter counts over 7. This makes them:
- Hard to read in one sitting (nobody holds 1,200 lines in working memory)
- Risky to modify (a change to one route can accidentally affect another)
- Slow to onboard new contributors (too much context to absorb at once)
- Prone to merge conflicts (multiple people touching the same large file)

Additionally, a circular import exists between `ai_draft.py` and `_guardrails.py` (currently resolved via lazy import), which is fragile and can produce obscure import errors.

## Proposed Solution

Split each integrator module into domain-cohesive sub-modules with a thin shim file for backward compatibility. The refactoring is purely structural — no behavior changes, no new features, no API changes.

### Branch Strategy (stacked)

```
main
 └── refactor/cli-modules          ← splits cli.py → 6 files
 └── refactor/routes-modules       ← splits routes.py → 6 files
 └── refactor/ai-draft-modules     ← splits ai_draft.py → 5 files + fixes circular dep
```

Each branch is independent and can be merged in any order. Each leaves the codebase in a valid, working state.

### Module Boundaries

#### `cli.py` → `cli/` package

```
cli/
├── __init__.py          # app creation, sub-command registration (typer), main(), web_server()
├── check.py             # check_api()
├── search_commands.py   # search_papers(), fetch_paper(), fetch_many()
├── image_commands.py    # list_images(), audit_images(), clean_images(),
│                        #   reclassify_images(), rescore_images(), rank_images()
├── task_commands.py     # create_task(), validate_existing(), list_tasks(), create_task_batch()
└── admin_commands.py    # set_diff(), export_task_cmd(), stats(), submit_task(), analytics()
```

`cli/__init__.py` re-exports `app` for `pyproject.toml` entry point.

#### `web/routes.py` → `web/` package

```
web/
├── routes.py (shim)     # re-exports all route functions; removed after callers migrate
├── pages.py             # index(), images_page(), tasks_page(), task_form(),
│                        #   task_detail(), stats_page(), author_page(), metrics_page()
├── author_routes.py     # _save_upload(), api_upload_image(), api_draft_qa(),
│                        #   api_discard_image(), api_propose_task()
├── arxiv_routes.py      # api_arxiv_search(), api_arxiv_extract()
├── task_routes.py       # api_create_task(), api_update_task(), api_validate_task(),
│                        #   api_regenerate_task(), api_generation_history()
├── lifecycle_routes.py  # update_figure_status(), bulk_reject_figures(),
│                        #   update_task_difficulty(), submit_task(),
│                        #   update_rhea(), save_rhea_override()
└── metrics.py           # _compute_metrics()
```

Each file declares its own `router = APIRouter()` and registers routes. `web/app.py` aggregates them.

#### `authoring/ai_draft.py` → `authoring/ai_draft/` package

```
authoring/ai_draft/
├── __init__.py (shim)   # re-exports public API: draft_qa, draft_qa_consensus,
│                        #   draft_with_self_critique, verify_draft
├── core.py              # draft_qa() — main drafting function
├── _api_client.py       # _get_api_key(), _call_opencode()
├── _response_parser.py  # _parse_llm_response(), _parse_critique_response(),
│                        #   _extract_reasoning()
├── _verifier.py         # verify_draft()
└── composition.py       # draft_qa_consensus(), draft_with_self_critique()
```

### Circular Dependency Fix

`_guardrails.py` currently does `from .ai_draft import draft_qa` (lazy import). Change `run_guardrails()` to accept `draft_qa_callback=None` parameter. The caller (`draft_qa` in `core.py`) passes itself as the callback. This breaks the cycle entirely.

```python
# Before (in _guardrails.py):
from .ai_draft import draft_qa
retry_draft = draft_qa(...)

# After (in _guardrails.py):
def run_guardrails(draft, context, ..., draft_qa_callback=None):
    if draft_qa_callback:
        retry_draft = draft_qa_callback(...)
```

## Out of Scope

- No new features or behavior changes
- No renaming of existing public APIs (function names stay the same)
- No database schema changes
- No dependency upgrades
- No test changes (all existing tests must pass without modification)
- No changes to `pyproject.toml` entry point (the `arxiv-manager` command stays the same)
- No changes to `web/app.py` import paths (shim pattern preserves them)

## Acceptance Criteria

- [ ] All 211 existing tests pass after each branch
- [ ] `arxiv-manager check` completes without error after each branch
- [ ] `arxiv-manager web` starts without import errors after each branch
- [ ] Each sub-module file is under 300 lines
- [ ] No new BLOCKING-level Warden violations introduced
- [ ] No new circular imports introduced
- [ ] `from arxiv_manager.web.routes import router` continues to work (shim)
- [ ] `from arxiv_manager.authoring.ai_draft import draft_qa` continues to work (shim)

## Testing Strategy

| Level | Scope | Approach |
|-------|-------|----------|
| **Unit** | All existing tests must pass | Run full suite after each branch; no test changes expected |
| **Integration** | CLI commands execute without import errors | Run `arxiv-manager check` after each branch |
| **Import validation** | All public imports resolve | `python -c "from arxiv_manager.cli import app; from arxiv_manager.web.routes import router; from arxiv_manager.authoring.ai_draft import draft_qa"` |

No new tests are needed — the refactoring is purely structural. Existing tests verify behavior.

## Open Questions

| Question | Status | Reasoning |
|----------|--------|-----------|
| Should `cli/__init__.py` import sub-modules eagerly or lazily? | **Eager** | Typer uses decorators at module load time to register commands. Lazy imports would miss command registration. |
| Should the `web/app.py` be updated to import from sub-modules directly? | **No** | Shim pattern preserves backward compatibility. `app.py` continues to `from .routes import router`. A future PR can update `app.py` to import from sub-modules and remove the shim. |
| Should function bodies be modified during the split? | **No** | Pure mechanical extraction. No behavioral changes. Any bug introduced is a copy-paste error, not a logic change. |
