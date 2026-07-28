"""Tests for all HTTP endpoints.

NOTE on monkeypatch: When monkeypatching a function imported via
``from X import Y``, patch the consumer module's reference, not the source.
Example: ``routes.py`` does ``from ..ai_draft import draft_with_self_critique``.
In tests, patch ``routes_mod.draft_with_self_critique``, NOT
``ai_draft.draft_with_self_critique``, because the consumer already holds a
direct reference to the original function object.
"""

import io
import re

import pytest
from PIL import Image
from sqlmodel import select

from arxiv_manager.models import Figure, Paper, Task

# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------


class TestGetEndpoints:
    def test_get_dashboard(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200

    def test_get_tasks(self, test_client):
        resp = test_client.get("/tasks")
        assert resp.status_code == 200

    def test_get_author_page(self, test_client):
        resp = test_client.get("/author")
        assert resp.status_code == 200

    def test_get_images(self, test_client):
        resp = test_client.get("/images")
        assert resp.status_code == 200

    def test_get_stats(self, test_client):
        resp = test_client.get("/stats")
        assert resp.status_code == 200

    def test_get_metrics(self, test_client):
        resp = test_client.get("/metrics")
        assert resp.status_code == 200

    def test_get_task_edit(self, test_client, sample_task):
        resp = test_client.get(f"/task/{sample_task.id}")
        assert resp.status_code == 200
        assert sample_task.title.encode() in resp.content

    def test_get_task_edit_not_found(self, test_client):
        resp = test_client.get("/task/99999")
        assert resp.status_code == 404

    def test_get_task_new_with_figure(self, test_client, sample_figure):
        resp = test_client.get(f"/task/new/{sample_figure.id}")
        assert resp.status_code == 200

    def test_get_task_new_not_found(self, test_client):
        resp = test_client.get("/task/new/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST: Upload image
# ---------------------------------------------------------------------------


class TestUploadImage:
    def test_upload_image(self, test_client):
        """Upload a synthetic JPEG image."""
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        resp = test_client.post(
            "/api/image/upload",
            files={"image": ("test.jpg", buf, "image/jpeg")},
        )
        # Returns 200 with analysis HTML
        assert resp.status_code == 200
        assert b"data-upload-id" in resp.content

    def test_upload_no_file(self, test_client):
        """POST without file returns 200 with error (route accepts optional params)."""
        resp = test_client.post("/api/image/upload")
        assert resp.status_code in (200, 422)

    def test_upload_non_image_bytes_rejected(self, test_client, override_storage):
        """Uploading non-image bytes with non-image extension returns error."""
        resp = test_client.post(
            "/api/image/upload",
            files={"image": ("malware.exe", b"not an image", "application/octet-stream")},
        )
        # Route catches exception and returns 200 with error message in response
        assert resp.status_code == 200
        assert b"cannot identify" in resp.content.lower() or b"upload" in resp.content.lower()
        # No .exe file should exist in uploads directory (security fix)
        uploads = list(override_storage.glob("_uploads/*.exe"))
        assert len(uploads) == 0


# ---------------------------------------------------------------------------
# POST: Draft QA
# ---------------------------------------------------------------------------


class TestDraftQA:
    def test_draft_qa_needs_upload_first(self, test_client):
        """Draft without an existing upload returns error HTML."""
        resp = test_client.post(
            "/api/image/draft",
            data={"upload_id": "nonexistent", "difficulty": "challenging"},
        )
        assert resp.status_code == 200
        assert b"Upload not found" in resp.content

    def test_draft_qa_no_api_key(self, test_client_no_key):
        """Draft without API key returns error (on a fresh upload)."""
        # Upload using test_client_no_key (no API key — but upload doesn't need it)
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        up = test_client_no_key.post(
            "/api/image/upload",
            files={"image": ("t.jpg", buf, "image/jpeg")},
        )
        assert up.status_code == 200
        import re

        match = re.search(rb'data-upload-id="([^"]+)"', up.content)
        assert match, "No upload_id in response"
        upload_id = match.group(1).decode()

        # Now draft without API key
        resp = test_client_no_key.post(
            "/api/image/draft",
            data={"upload_id": upload_id, "difficulty": "challenging"},
        )
        assert resp.status_code == 200
        assert b"OPENCODE_API_KEY" in resp.content or b"api key" in resp.content.lower()

    def test_draft_qa_with_mock(self, test_client, mock_draft_success):
        """Draft with mocked API returns draft HTML."""
        # Upload first
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        up = test_client.post(
            "/api/image/upload",
            files={"image": ("t.jpg", buf, "image/jpeg")},
        )
        import re

        match = re.search(rb'data-upload-id="([^"]+)"', up.content)
        assert match
        upload_id = match.group(1).decode()

        resp = test_client.post(
            "/api/image/draft",
            data={"upload_id": upload_id, "difficulty": "challenging"},
        )
        assert resp.status_code == 200
        # Should contain the draft question from our mock
        assert b"peak value" in resp.content


# ---------------------------------------------------------------------------
# POST: Propose task
# ---------------------------------------------------------------------------


class TestProposeTask:
    def test_propose_creates_task(self, test_client):
        """Propose creates a Task and redirects."""
        # Upload first
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        up = test_client.post(
            "/api/image/upload",
            files={"image": ("t.jpg", buf, "image/jpeg")},
        )
        import re

        match = re.search(rb'data-upload-id="([^"]+)"', up.content)
        assert match
        upload_id = match.group(1).decode()

        resp = test_client.post(
            "/api/image/propose",
            data={
                "upload_id": upload_id,
                "question": "How many distinct colored regions are visible in this image?",
                "answer": "2",
                "answer_format": "number",
                "task_type": "general_image",
                "domain": "Physics",
                "title": "Test",
            },
        )
        # Should redirect (303) to the task edit page
        assert resp.status_code in (200, 303, 302)
        if resp.status_code in (303, 302):
            assert "/task/" in resp.headers.get("location", "")

    def test_propose_missing_fields(self, test_client):
        """Propose with missing fields returns error."""
        resp = test_client.post(
            "/api/image/propose",
            data={
                "upload_id": "nonexistent",
                "question": "",
                "answer": "",
                "answer_format": "",
                "task_type": "",
                "domain": "",
                "title": "",
            },
        )
        assert resp.status_code in (200, 422, 400)


# ---------------------------------------------------------------------------
# POST: Discard upload
# ---------------------------------------------------------------------------


class TestDiscardUpload:
    def test_discard_existing(self, test_client):
        """Discard an existing upload."""
        # Upload first
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        up = test_client.post(
            "/api/image/upload",
            files={"image": ("t.jpg", buf, "image/jpeg")},
        )
        import re

        match = re.search(rb'data-upload-id="([^"]+)"', up.content)
        assert match
        upload_id = match.group(1).decode()

        resp = test_client.post(
            "/api/image/discard",
            data={"upload_id": upload_id},
        )
        assert resp.status_code == 200

    def test_discard_nonexistent(self, test_client):
        """Discard of non-existent upload returns 200 (idempotent)."""
        resp = test_client.post(
            "/api/image/discard",
            data={"upload_id": "nonexistent"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST: Task regenerate
# ---------------------------------------------------------------------------


class TestTaskRegenerate:
    def test_regenerate_with_mock(self, test_client, override_storage, monkeypatch):
        """Regenerate with mocked API returns new Q&A."""
        import arxiv_manager.web.routes.task_routes as tr_mod

        def _fake_draft(**kw):
            return {"question": "Mock Q?", "answer": "99", "answer_format": "number", "task_type": "chart"}

        monkeypatch.setattr(tr_mod, "draft_with_self_critique", _fake_draft)
        monkeypatch.setattr(tr_mod, "draft_qa", _fake_draft)

        # Upload an image to get a valid upload_id
        import io

        from PIL import Image

        img = Image.new("RGB", (100, 100), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        up = test_client.post("/api/image/upload", files={"image": ("t.jpg", buf, "image/jpeg")})
        assert up.status_code == 200
        match = re.search(rb'data-upload-id="([^"]+)"', up.content)
        assert match
        upload_id = match.group(1).decode()

        # Propose the task
        prop = test_client.post(
            "/api/image/propose",
            data={
                "upload_id": upload_id,
                "question": "Q?",
                "answer": "1",
                "answer_format": "number",
                "task_type": "chart",
                "domain": "Physics",
                "title": "Test",
            },
        )
        task_id = None
        if prop.status_code in (303, 302):
            loc = prop.headers.get("location", "")
            m2 = re.search(r"/task/(\d+)", loc)
            if m2:
                task_id = int(m2.group(1))
        if not task_id:
            from arxiv_manager.db import get_session
            from arxiv_manager.models import Task

            s = get_session()
            tasks = s.exec(select(Task).order_by(Task.id.desc())).first()
            if tasks:
                task_id = tasks.id
            s.close()
        if not task_id:
            pytest.skip("Could not create a task via propose endpoint")

        resp = test_client.post(
            f"/api/task/{task_id}/regenerate",
            data={"difficulty": "challenging"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "question" in data
        assert data["answer_format"] in ("number", "word", "phrase")

    def test_regenerate_not_found(self, test_client):
        """Regenerate for non-existent task returns error."""
        resp = test_client.post(
            "/api/task/99999/regenerate",
            data={"difficulty": "challenging"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "error" in data

    def test_regenerate_no_api_key(self, test_client_no_key, sample_task):
        """Regenerate without API key returns error."""
        resp = test_client_no_key.post(
            f"/api/task/{sample_task.id}/regenerate",
            data={"difficulty": "challenging"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "OPENCODE_API_KEY" in data.get("error", "")

    def test_regenerate_no_image(self, test_client, sample_task):
        """Regenerate when task has no image returns error."""
        # Update the task's image_path to a non-existent file
        from sqlmodel import select

        from arxiv_manager.db import get_session
        from arxiv_manager.models import Task

        s = get_session()
        t = s.exec(select(Task).where(Task.id == sample_task.id)).first()
        if t:
            t.image_path = "figures/nonexistent.png"
            s.add(t)
            s.commit()
        s.close()

        resp = test_client.post(
            f"/api/task/{sample_task.id}/regenerate",
            data={"difficulty": "challenging"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False


# ---------------------------------------------------------------------------
# POST: Task update
# ---------------------------------------------------------------------------


class TestTaskUpdate:
    def test_update_task(self, test_client, sample_task):
        resp = test_client.post(
            f"/api/task/{sample_task.id}/update",
            data={
                "title": "Updated Title",
                "question": "Updated question?",
                "answer": "99",
                "answer_format": "number",
                "task_type": "chart",
                "domain": "Physics",
            },
        )
        assert resp.status_code == 200
        assert b"Updated" in resp.content

    def test_update_task_not_found(self, test_client):
        resp = test_client.post(
            "/api/task/99999/update",
            data={
                "title": "X",
                "question": "Q?",
                "answer": "A",
                "answer_format": "word",
                "task_type": "chart",
                "domain": "CS",
            },
        )
        assert resp.status_code == 200  # Returns HTML with error or new-task form


# ---------------------------------------------------------------------------
# POST: Task delete
# ---------------------------------------------------------------------------


class TestTaskDelete:
    def test_delete_task(self, test_client, sample_task):
        resp = test_client.post(f"/api/task/{sample_task.id}/delete")
        # Redirects to /tasks (303 → TestClient follows to 200)
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        t = s.get(Task, sample_task.id)
        assert t is None
        s.close()

    def test_delete_not_found(self, test_client):
        resp = test_client.post("/api/task/99999/delete")
        assert resp.status_code == 404

    def test_delete_cascades_to_related(
        self, test_client, sample_task, sample_figure, db_session
    ):
        """Deleting a task also deletes its GenerationAttempts, IssueReports, and SubmissionLogs."""
        from arxiv_manager.db import get_session as _get_session
        from arxiv_manager.models import GenerationAttempt, IssueReport, SubmissionLog

        db_session.add(
            GenerationAttempt(
                figure_id=sample_figure.id,
                task_id=sample_task.id,
                generation_type="draft",
                difficulty="easy",
                generated_question="Q?",
                generated_answer="A",
                success=True,
            )
        )
        db_session.add(
            IssueReport(
                task_id=sample_task.id,
                figure_id=sample_figure.id,
                reason="too_easy",
            )
        )
        db_session.add(
            SubmissionLog(
                task_id=sample_task.id,
            )
        )
        db_session.commit()

        resp = test_client.post(f"/api/task/{sample_task.id}/delete")
        assert resp.status_code in (200, 303, 302)

        # Use a fresh session to avoid stale identity map
        fresh = _get_session()
        try:
            assert fresh.get(Task, sample_task.id) is None
            assert (
                fresh.exec(select(GenerationAttempt).where(GenerationAttempt.task_id == sample_task.id)).first()
                is None
            )
            assert (
                fresh.exec(select(IssueReport).where(IssueReport.task_id == sample_task.id)).first() is None
            )
            assert (
                fresh.exec(select(SubmissionLog).where(SubmissionLog.task_id == sample_task.id)).first() is None
            )
        finally:
            fresh.close()


# ---------------------------------------------------------------------------
# POST: Task submit
# ---------------------------------------------------------------------------


class TestTaskSubmit:
    def test_submit_task(self, test_client, sample_task):
        resp = test_client.post(f"/api/task/{sample_task.id}/submit")
        assert resp.status_code in (200, 303, 302)
        # Verify status changed
        from arxiv_manager.db import get_session

        s = get_session()
        t = s.get(Task, sample_task.id)
        assert t.status == "submitted"
        s.close()

    def test_submit_not_found(self, test_client):
        resp = test_client.post("/api/task/99999/submit")
        # Returns redirect (303) which TestClient follows to /tasks (200)
        assert resp.status_code in (200, 303, 302)


# ---------------------------------------------------------------------------
# POST: Rhea review
# ---------------------------------------------------------------------------


class TestRheaReview:
    def test_rhea_review_passed(self, test_client, sample_task):
        resp = test_client.post(
            f"/api/task/{sample_task.id}/rhea",
            data={"rhea_reviewed": "true", "rhea_passed": "true", "rhea_notes": "Good"},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        t = s.get(Task, sample_task.id)
        assert t.rhea_reviewed is True
        assert t.rhea_passed is True
        s.close()

    def test_rhea_review_failed(self, test_client, sample_task):
        resp = test_client.post(
            f"/api/task/{sample_task.id}/rhea",
            data={"rhea_reviewed": "true", "rhea_passed": "false", "rhea_notes": "Too easy"},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        t = s.get(Task, sample_task.id)
        assert t.rhea_reviewed is True
        assert t.rhea_passed is False
        s.close()

    def test_rhea_override(self, test_client, sample_task):
        resp = test_client.post(
            f"/api/task/{sample_task.id}/rhea-override",
            data={"rhea_passed": "true", "rhea_override_notes": "Disagree with Rhea"},
        )
        assert resp.status_code == 200
        try:
            data = resp.json()
            assert data.get("ok") is True
        except Exception:
            # May return HTML redirect on success
            pass
        from arxiv_manager.db import get_session

        s = get_session()
        t = s.get(Task, sample_task.id)
        assert t.rhea_passed is True
        assert t.rhea_override_notes == "Disagree with Rhea"
        s.close()


# ---------------------------------------------------------------------------
# POST: Figure status
# ---------------------------------------------------------------------------


class TestFigureStatus:
    def test_update_figure_status(self, test_client, sample_figure):
        resp = test_client.post(
            f"/api/figure/{sample_figure.id}/status",
            data={"status": "rejected"},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        f = s.get(Figure, sample_figure.id)
        assert f.status == "rejected"
        s.close()


# ---------------------------------------------------------------------------
# POST: Bulk reject
# ---------------------------------------------------------------------------


class TestBulkReject:
    def test_bulk_reject(self, test_client, db_session):
        paper = Paper(id="1111.11111", title="Test")
        db_session.add(paper)
        db_session.commit()
        f1 = Figure(paper_id=paper.id, page=1, image_path="figures/a.png", image_hash="aaa")
        f2 = Figure(paper_id=paper.id, page=2, image_path="figures/b.png", image_hash="bbb")
        db_session.add(f1)
        db_session.add(f2)
        db_session.commit()

        resp = test_client.post(
            "/api/figures/bulk-reject",
            data={"figure_ids": [f1.id, f2.id]},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        ff1 = s.get(Figure, f1.id)
        ff2 = s.get(Figure, f2.id)
        assert ff1.status == "rejected"
        assert ff2.status == "rejected"
        s.close()


# ---------------------------------------------------------------------------
# POST: Task difficulty
# ---------------------------------------------------------------------------


class TestTaskDifficulty:
    def test_update_difficulty(self, test_client, sample_task):
        resp = test_client.post(
            f"/api/task/{sample_task.id}/difficulty",
            data={"difficulty": "hardest", "qwen": "2", "gemini": "4"},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        t = s.get(Task, sample_task.id)
        assert t.difficulty == "hardest"
        assert t.qwen_passes == 2
        assert t.gemini_passes == 4
        s.close()


# ---------------------------------------------------------------------------
# Edge-case tests — not-found and missing-entity behavior
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_rhea_review_not_found(self, test_client):
        """Rhea review of non-existent task returns redirect (then 404 at target)."""
        resp = test_client.post(
            "/api/task/99999/rhea",
            data={"rhea_reviewed": "true", "rhea_passed": "true", "rhea_notes": ""},
        )
        # Returns 303 redirect; TestClient follows to /task/99999 → 404
        assert resp.status_code == 404 or resp.status_code in (200, 303, 302)

    def test_rhea_override_not_found(self, test_client):
        """Rhea override of non-existent task — redirects to task page (which gives 404)."""
        resp = test_client.post(
            "/api/task/99999/rhea-override",
            data={"rhea_passed": "true", "rhea_override_notes": "test"},
        )
        assert resp.status_code == 404 or resp.status_code in (200, 303, 302)

    def test_difficulty_not_found(self, test_client):
        """Difficulty update for non-existent task."""
        resp = test_client.post(
            "/api/task/99999/difficulty",
            data={"difficulty": "easy", "qwen": "0", "gemini": "0"},
        )
        # Returns 303 redirect to /task/99999 → 404
        assert resp.status_code == 404 or resp.status_code in (200, 303, 302)

    def test_figure_status_not_found(self, test_client):
        """Figure status update for non-existent figure returns redirect."""
        resp = test_client.post(
            "/api/figure/99999/status",
            data={"status": "rejected"},
        )
        assert resp.status_code in (200, 303, 302)


# ─── Generation History ──────────────────────────────────────────────


class TestTaskHistory:
    def test_history_not_found(self, test_client):
        """GET /api/task/99999/task-history returns 404 for non-existent task."""
        resp = test_client.get("/api/task/99999/task-history")
        assert resp.status_code == 404

    def test_history_empty(self, test_client, sample_task):
        """History for a task with no data returns empty-state partial."""
        resp = test_client.get(f"/api/task/{sample_task.id}/task-history")
        assert resp.status_code == 200
        assert "No task history yet" in resp.text

    def test_history_with_attempts(self, test_client, sample_task, sample_figure, db_session):
        """History renders generation attempt records."""
        from arxiv_manager.models import GenerationAttempt

        db_session.add(
            GenerationAttempt(
                figure_id=sample_figure.id,
                task_id=sample_task.id,
                generation_type="draft",
                difficulty="challenging",
                generated_question="Test historical question?",
                generated_answer="42",
                validation_quality=85.0,
                success=True,
                model_name="minimax-m3",
            )
        )
        db_session.commit()
        resp = test_client.get(f"/api/task/{sample_task.id}/task-history")
        assert resp.status_code == 200
        assert "Test historical question?" in resp.text
        assert "draft" in resp.text
        assert "challenging" in resp.text
        assert "42" in resp.text

    def test_history_with_task_events(self, test_client, sample_task, db_session):
        """History renders TaskEvent records."""
        from arxiv_manager.models import TaskEvent

        db_session.add(
            TaskEvent(
                task_id=sample_task.id,
                event_type="update",
                details='{"changed_fields": ["question"], "old_values": {"question": "Old Q?"}, "new_values": {"question": "New Q?"}}',
            )
        )
        db_session.commit()
        resp = test_client.get(f"/api/task/{sample_task.id}/task-history")
        assert resp.status_code == 200
        assert "Task Update" in resp.text
        assert "Old Q?" in resp.text
        assert "New Q?" in resp.text


# ─── TaskEvent logging ─────────────────────────────────────────────


class TestTaskEventLogging:
    def test_submit_creates_task_event(self, test_client, sample_task):
        """Submitting a task logs a TaskEvent."""
        from arxiv_manager.models import TaskEvent

        resp = test_client.post(f"/api/task/{sample_task.id}/submit")
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        event = s.exec(select(TaskEvent).where(TaskEvent.task_id == sample_task.id)).first()
        s.close()
        assert event is not None
        assert event.event_type == "submit"

    def test_difficulty_change_creates_task_event(self, test_client, sample_task):
        """Updating difficulty logs a TaskEvent."""
        from arxiv_manager.models import TaskEvent

        resp = test_client.post(
            f"/api/task/{sample_task.id}/difficulty",
            data={"difficulty": "hardest", "qwen": "2", "gemini": "4"},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        event = s.exec(
            select(TaskEvent).where(TaskEvent.task_id == sample_task.id, TaskEvent.event_type == "difficulty_change")
        ).first()
        s.close()
        assert event is not None
        assert event.event_type == "difficulty_change"

    def test_rhea_review_creates_task_event(self, test_client, sample_task):
        """Rhea review logs a TaskEvent."""
        from arxiv_manager.models import TaskEvent

        resp = test_client.post(
            f"/api/task/{sample_task.id}/rhea",
            data={"rhea_reviewed": "true", "rhea_passed": "true", "rhea_notes": "Good"},
        )
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        event = s.exec(
            select(TaskEvent).where(TaskEvent.task_id == sample_task.id, TaskEvent.event_type == "rhea_review")
        ).first()
        s.close()
        assert event is not None

    def test_delete_creates_task_event(self, test_client, sample_task):
        """Deleting a task logs a TaskEvent."""
        from arxiv_manager.models import TaskEvent

        resp = test_client.post(f"/api/task/{sample_task.id}/delete")
        assert resp.status_code in (200, 303, 302)
        from arxiv_manager.db import get_session

        s = get_session()
        event = s.exec(
            select(TaskEvent).where(TaskEvent.task_id == sample_task.id, TaskEvent.event_type == "delete")
        ).first()
        s.close()
        assert event is not None
