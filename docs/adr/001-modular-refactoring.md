# ADR-001: Modular Refactoring of Integrator Modules

**Date:** 2026-07-24
**Status:** Proposed

## Context

The Warden scan (Issue #7) identified three integrator modules that exceed maintainability thresholds:

| Module | Lines | Threshold | Violations |
|--------|-------|-----------|------------|
| `cli.py` | 1,264 | >500 (BLOCKING) | 7 functions >40 lines, 3 functions >80 lines, 4 functions with complexity >20 |
| `web/routes.py` | 1,066 | >500 (BLOCKING) | 3 functions >80 lines, 3 functions with complexity >20, 2 functions with 8+ params |
| `authoring/ai_draft.py` | 652 | >500 (BLOCKING) | 5 functions >80 lines, 4 functions with complexity >20, 3 functions with 8+ params |

These modules are the primary integrators — they import from 10–12 internal modules each and orchestrate cross-cutting workflows. Their size is a consequence of organic growth: all three have accumulated route handlers, CLI commands, and drafting logic without structural decomposition.

The tightest coupling: `ai_draft.py` ↔ `_guardrails.py` (circular import, resolved by lazy import).

## Decision

Split each integrator module into domain-cohesive sub-modules. Use a shim file pattern for backward compatibility.

### Pattern

```
Parent file → split into domain files + thin shim re-exporting the public API
```

The shim ensures that all existing `from X import Y` statements continue to work without modification. New code can import directly from the sub-modules.

### `cli.py` → 6 files

| File | Content | Risk |
|------|---------|------|
| `cli/__init__.py` | App creation, sub-command registration, `main()`, `web_server()` | Low |
| `cli/check.py` | `check_api()` | Low |
| `cli/search_commands.py` | `search_papers()`, `fetch_paper()`, `fetch_many()` | Low |
| `cli/image_commands.py` | `list_images()`, `audit_images()`, `clean_images()`, `reclassify_images()`, `rescore_images()`, `rank_images()` | Low |
| `cli/task_commands.py` | `create_task()`, `validate_existing()`, `list_tasks()`, `create_task_batch()` | Medium |
| `cli/admin_commands.py` | `set_diff()`, `export_task_cmd()`, `stats()`, `submit_task()`, `analytics()` | Low |

### `web/routes.py` → 6 files

| File | Content | Risk |
|------|---------|------|
| `web/pages.py` | All page GET handlers (8 routes) | Low |
| `web/author_routes.py` | Upload + draft endpoints (5 routes) | Medium |
| `web/arxiv_routes.py` | arXiv search + extract (2 routes) | Low |
| `web/task_routes.py` | Task CRUD + regenerate + history (5 routes) | High |
| `web/lifecycle_routes.py` | Status, difficulty, submit, Rhea (6 routes) | Low |
| `web/metrics.py` | `_compute_metrics()` | Low |

### `authoring/ai_draft.py` → 5 files

| File | Content | Risk |
|------|---------|------|
| `authoring/ai_draft/core.py` | `draft_qa()` — prompt building, history injection, guardrails | High |
| `authoring/ai_draft/_api_client.py` | `_get_api_key()`, `_call_opencode()` | Medium |
| `authoring/ai_draft/_response_parser.py` | `_parse_llm_response()`, `_parse_critique_response()`, `_extract_reasoning()` | Low |
| `authoring/ai_draft/_verifier.py` | `verify_draft()` | Low |
| `authoring/ai_draft/composition.py` | `draft_qa_consensus()`, `draft_with_self_critique()` | Medium |

### Circular dependency resolution

The `ai_draft` ↔ `_guardrails` circular dependency (currently resolved via lazy import) will be resolved by injecting `draft_qa` as a callback parameter to `run_guardrails()`:

```python
# _guardrails.py
def run_guardrails(draft, context, api_key=None, image_path="",
                   max_retries=2, draft_qa_callback=None):
    ...
    if api_key and image_path and draft_qa_callback:
        retry_draft = draft_qa_callback(
            image_path=image_path,
            feedback=feedback,
            ...
        )
```

This makes the data flow explicit and breaks the cycle entirely.

## Alternatives Considered

| Option | Pros | Cons | Why Rejected |
|--------|------|------|-------------|
| **No refactor** | Zero effort | Bugs compound; new features become harder to add; onboarding impairs | Rejected — the Warden thresholds exist for a reason |
| **Split everything at once** | Clean slate | Unreviewable 2,000-line diff; high merge conflict risk; hard to revert | Rejected — one branch per module |
| **Shim-only (no sub-modules)** | Minimal diff | Just renaming; doesn't reduce cognitive load per file | Rejected — doesn't solve the maintainability problem |
| **Shim + sub-modules (chosen)** | Incremental, revertible, reviewable per module | More files; shims add indirection | Accepted — the indirection is transparent to callers |

## Consequences

**Easier:**
- Each sub-module has a single responsibility and is readable in one sitting
- New features can be added to the correct sub-module without touching unrelated code
- Code review per PR becomes predictable (reviewer reads one domain)
- Parallel work possible (multiple people can refactor different modules simultaneously)

**Harder:**
- More files to navigate (though directory structure mirrors the domain boundaries)
- Import paths change for new code (though shims preserve backward compat for existing)
- Build process unchanged (file-based, no bundler)

**New risks:**
- Shim files must be maintained as the API evolves
- Need to ensure no cyclic dependencies emerge between sub-module files

## References

- Issue #7 — Warden code health baseline
- Warden scan commit `3d5b96a`
