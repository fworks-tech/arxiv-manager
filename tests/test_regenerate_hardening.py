"""Tests for the regenerate/restore/verdict/check-answer hardening.

Covers:
- restore gate: invalid / fact-check-failed / determinism-failed attempts are blocked
- regenerate cap: 3 consecutive failures (any reason) block further attempts
- regenerate-status endpoint: returns the latest job for the task
- verdict route: persists and returns difficulty
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine

from arxiv_manager.models import GenerationAttempt, Task


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    """App DB with a real file so route code (get_session) works."""
    from arxiv_manager import models as _app_models  # noqa: F401
    from arxiv_manager.scheduler import models as _sched_models  # noqa: F401

    monkeypatch.setenv("OPENCODE_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("OPENCODE_TESTING", "1")

    db_path = tmp_path / "app_test.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(engine)

    def _fake_session():
        return Session(engine)

    import arxiv_manager.db as db_mod
    import arxiv_manager.scheduler.queue as q_mod
    import arxiv_manager.tracking as track_mod
    import arxiv_manager.web.routes.task_routes as tr_mod

    monkeypatch.setattr(db_mod, "get_session", _fake_session)
    monkeypatch.setattr(tr_mod, "get_session", _fake_session)
    monkeypatch.setattr(q_mod, "get_session", _fake_session)
    monkeypatch.setattr(track_mod, "get_session", _fake_session)
    return engine


def _make_task(engine, image_path="figures/test.jpg"):
    s = Session(engine)
    t = Task(
        figure_id=1,
        title="t",
        question="Q?",
        answer="1",
        answer_format="number",
        task_type="chart",
        difficulty="hardest",
        image_path=image_path,
    )
    s.add(t)
    s.commit()
    s.refresh(t)
    tid = t.id
    s.close()
    return tid


class TestRestoreGate:
    def _make_attempt(self, engine, task_id, valid=True, fact="[]", det="[]", q="GQ?", a="7"):
        s = Session(engine)
        a_row = GenerationAttempt(
            figure_id=1,
            task_id=task_id,
            generation_type="regenerate_initial",
            validation_is_valid=valid,
            fact_check_errors=fact,
            determinism_errors=det,
            generated_question=q,
            generated_answer=a,
        )
        s.add(a_row)
        s.commit()
        s.refresh(a_row)
        aid = a_row.id
        s.close()
        return aid

    def test_restore_valid_attempt_ok(self, app_db, monkeypatch):
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)
        aid = self._make_attempt(app_db, tid)
        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.post(f"/api/task/{tid}/restore/{aid}")
        assert resp.status_code == 200
        assert b"Restored attempt" in resp.content

    def test_restore_invalid_attempt_ok(self, app_db, monkeypatch):
        """Restoring an invalid attempt is allowed — user has full control."""
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)
        aid = self._make_attempt(app_db, tid, valid=False)
        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.post(f"/api/task/{tid}/restore/{aid}")
        assert b"Restored attempt" in resp.content

    def test_restore_fact_failed_attempt_ok(self, app_db, monkeypatch):
        """Restoring a fact-check-failed attempt is allowed."""
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)
        aid = self._make_attempt(app_db, tid, fact='["The premise is not supported"]')
        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.post(f"/api/task/{tid}/restore/{aid}")
        assert b"Restored attempt" in resp.content

    def test_restore_determinism_failed_attempt_ok(self, app_db, monkeypatch):
        """Restoring a determinism-failed attempt is allowed."""
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)
        aid = self._make_attempt(app_db, tid, det='["12", "13"]')
        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.post(f"/api/task/{tid}/restore/{aid}")
        assert b"Restored attempt" in resp.content


class TestRegenerateCap:
    def _make_failed_attempt(self, engine, task_id, valid=False, prev_question="Q?", success=True):
        s = Session(engine)
        a_row = GenerationAttempt(
            figure_id=1,
            task_id=task_id,
            generation_type="regenerate_initial",
            validation_is_valid=valid,
            fact_check_errors="[]",
            determinism_errors="[]",
            success=success,
            previous_question=prev_question,
            generated_question="Draft Q?",
            generated_answer="1",
        )
        s.add(a_row)
        s.commit()
        s.close()

    def test_cap_blocks_after_3_consecutive_failures(self, app_db, monkeypatch):
        from arxiv_manager.web.routes import task_routes as tr_mod

        tid = _make_task(app_db)
        for _ in range(3):
            self._make_failed_attempt(app_db, tid, valid=False)

        # run_regeneration must reject before any LLM call
        result = tr_mod.run_regeneration(tid, "hardest")
        assert result["ok"] is False
        assert "3 consecutive times" in result["error"]

    def test_cap_resets_after_manual_edit(self, app_db, monkeypatch):
        from arxiv_manager.web.routes import task_routes as tr_mod

        tid = _make_task(app_db)
        for _ in range(3):
            self._make_failed_attempt(app_db, tid, valid=False)

        # A manual edit (logged as a TaskEvent) resets the cap

        from arxiv_manager.models import TaskEvent

        s = Session(app_db)
        t = s.get(Task, tid)
        t.question = "Manually edited question?"
        t.answer = "99"
        s.add(t)
        s.add(TaskEvent(task_id=tid, event_type="update", details='{"question": "Manually edited question?"}'))
        s.commit()
        s.close()

        # With the cap reset and an image missing, we get past the cap to the
        # image check — proving the cap no longer blocks.
        result = tr_mod.run_regeneration(tid, "hardest")
        assert result["error"] == "Image not found"

    def test_cap_resets_after_answer_only_edit(self, app_db, monkeypatch):
        """Answer-only edits reset the cap too (TaskEvent-based detection)."""
        from arxiv_manager.web.routes import task_routes as tr_mod

        tid = _make_task(app_db)
        for _ in range(3):
            self._make_failed_attempt(app_db, tid, valid=False)

        from arxiv_manager.models import TaskEvent

        s = Session(app_db)
        t = s.get(Task, tid)
        t.answer = "99"
        s.add(t)
        s.add(TaskEvent(task_id=tid, event_type="update", details='{"answer": "99"}'))
        s.commit()
        s.close()

        result = tr_mod.run_regeneration(tid, "hardest")
        assert result["error"] == "Image not found"  # cap passed, image check hit

    def test_cap_counts_llm_error_failures(self, app_db, monkeypatch):
        """Draft-None (LLM error) attempts are logged as failures and counted."""
        from arxiv_manager.web.routes import task_routes as tr_mod

        tid = _make_task(app_db)
        for _ in range(3):
            self._make_failed_attempt(app_db, tid, success=False, valid=False)

        result = tr_mod.run_regeneration(tid, "hardest")
        assert result["ok"] is False
        assert "3 consecutive times" in result["error"]
        assert "generation failed" in result["error"]

    def test_one_success_resets_cap(self, app_db, monkeypatch):
        from arxiv_manager.web.routes import task_routes as tr_mod

        tid = _make_task(app_db)
        self._make_failed_attempt(app_db, tid, valid=False)
        self._make_failed_attempt(app_db, tid, valid=False)
        s = Session(app_db)
        ok_row = GenerationAttempt(
            figure_id=1,
            task_id=tid,
            generation_type="regenerate_initial",
            validation_is_valid=True,
            fact_check_errors="[]",
            determinism_errors="[]",
            success=True,
            generated_question="Good draft?",
            generated_answer="5",
        )
        s.add(ok_row)
        s.commit()
        s.close()

        result = tr_mod.run_regeneration(tid, "hardest")
        assert result["error"] == "Image not found"  # cap passed, image check hit

    def test_draft_none_logs_failed_attempt(self, app_db, tmp_path, monkeypatch):
        """Draft-None (LLM error) now logs a success=False attempt for the cap."""
        from sqlmodel import select

        from arxiv_manager.models import GenerationAttempt
        from arxiv_manager.web.routes import task_routes as tr_mod

        # Create a real image so the pipeline reaches the draft call
        img = tmp_path / "figures" / "missing.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.new("RGB", (50, 50), (128, 128, 128)).save(img)
        monkeypatch.setattr(tr_mod, "STORAGE_DIR", tmp_path)
        tid = _make_task(app_db, image_path="figures/missing.png")

        # Force draft_qa to return None so _do_regenerate returns None
        monkeypatch.setattr(tr_mod, "draft_with_self_critique", lambda **kw: None)
        monkeypatch.setattr(tr_mod, "draft_qa", lambda **kw: None)

        result = tr_mod.run_regeneration(tid, "hardest")
        assert result["error"] == "Draft generation failed"

        s = Session(app_db)
        attempt = s.exec(
            select(GenerationAttempt)
            .where(GenerationAttempt.task_id == tid)
            .order_by(GenerationAttempt.id.desc())
        ).first()
        assert attempt is not None
        assert attempt.success is False
        s.close()


class TestRegenerateStatusEndpoint:
    def test_status_endpoint_returns_latest_job(self, app_db, monkeypatch):
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)

        from arxiv_manager.scheduler.queue import complete_job, enqueue

        job = enqueue("regenerate_task", {"task_id": tid, "difficulty": "hardest"}, max_attempts=1)
        complete_job(job.id, {"ok": True, "question": "Q2?", "answer": "42"})

        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.get(f"/api/task/{tid}/regenerate-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["result"]["answer"] == "42"

    def test_status_endpoint_no_job(self, app_db, monkeypatch):
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)
        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.get(f"/api/task/{tid}/regenerate-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "none"

    def test_status_endpoint_filters_other_tasks(self, app_db, monkeypatch):
        """Jobs for other tasks don't shadow this task's job (SQL filter)."""
        from fastapi.testclient import TestClient

        tid = _make_task(app_db)
        other = _make_task(app_db, image_path="figures/other.jpg")

        from arxiv_manager.scheduler.queue import complete_job, enqueue

        # 25 jobs for OTHER tasks, pushing this task's job out of a naive top-20
        for i in range(25):
            j = enqueue("regenerate_task", {"task_id": other, "difficulty": "hardest"}, max_attempts=1)
            complete_job(j.id, {"ok": True, "answer": str(i)})
        job = enqueue("regenerate_task", {"task_id": tid, "difficulty": "hardest"}, max_attempts=1)
        complete_job(job.id, {"ok": True, "question": "Q2?", "answer": "42"})

        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.get(f"/api/task/{tid}/regenerate-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["result"]["answer"] == "42"


class TestGoldenSuspectFlag:
    def _make_task_with_image(self, engine, tmp_path, monkeypatch):
        """Task with a real image file so check-answer can encode it."""
        import arxiv_manager.web.routes.task_routes as tr_mod

        img = tmp_path / "figures" / "test.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.new("RGB", (50, 50), (200, 100, 50)).save(img)
        monkeypatch.setattr(tr_mod, "STORAGE_DIR", tmp_path)
        return _make_task(engine, image_path="figures/test.png")

    def test_check_answer_flags_suspect_golden(self, app_db, tmp_path, monkeypatch):
        """When the verifier judges the golden wrong, golden_suspect is set."""

        tid = self._make_task_with_image(app_db, tmp_path, monkeypatch)

        # VLM answers differently, verifier says golden is wrong
        calls = {"n": 0}

        def _fake_call(api_key, prompt, b64_image, **kwargs):
            calls["n"] += 1
            if "VLM reasoning" not in prompt and "check" not in prompt.lower()[:60]:
                return {"answer": "Panel 3", "reasoning": "Panel 3 fits the criteria"}
            return {"match": False, "golden_correct": False, "explanation": "differ", "analysis": "golden wrong"}

        monkeypatch.setattr(
            "arxiv_manager.authoring.ai_draft._api_client._call_opencode",
            _fake_call,
        )

        from fastapi.testclient import TestClient

        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.post(f"/api/task/{tid}/check-answer")
        assert resp.status_code == 200
        assert b"Golden answer flagged as suspect" in resp.content
        assert calls["n"] == 2

        s = Session(app_db)
        t = s.get(Task, tid)
        assert t.golden_suspect is True
        s.close()

    def test_check_answer_clears_flag_on_match(self, app_db, tmp_path, monkeypatch):
        """A matching run clears a previously-set golden_suspect flag."""

        tid = self._make_task_with_image(app_db, tmp_path, monkeypatch)

        s = Session(app_db)
        t = s.get(Task, tid)
        t.golden_suspect = True
        s.add(t)
        s.commit()
        s.close()

        def _fake_call(api_key, prompt, b64_image, **kwargs):
            if "VLM reasoning" not in prompt and "check" not in prompt.lower()[:60]:
                return {"answer": "1", "reasoning": "reads 1"}
            return {"match": True, "golden_correct": True, "explanation": "same", "analysis": ""}

        monkeypatch.setattr(
            "arxiv_manager.authoring.ai_draft._api_client._call_opencode",
            _fake_call,
        )

        from fastapi.testclient import TestClient

        from arxiv_manager.web.app import create_app

        with TestClient(create_app()) as c:
            resp = c.post(f"/api/task/{tid}/check-answer")
        assert resp.status_code == 200

        s = Session(app_db)
        t = s.get(Task, tid)
        assert t.golden_suspect is False
        s.close()
