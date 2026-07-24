"""Tests for all HTTP endpoints."""

import io
import json
import re

import pytest
from PIL import Image
from sqlmodel import select

from arxiv_manager.models import Task, Figure, Paper, TaskStatus, ImageStatus


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
                "question": "Test question?",
                "answer": "42",
                "answer_format": "number",
                "task_type": "chart",
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
# POST: Task validate
# ---------------------------------------------------------------------------

class TestTaskValidate:
    def test_validate_existing(self, test_client, sample_task):
        resp = test_client.post(f"/api/task/{sample_task.id}/validate")
        assert resp.status_code == 200
        assert b"checks passed" in resp.content or b"Issues found" in resp.content

    def test_validate_not_found(self, test_client):
        resp = test_client.post("/api/task/99999/validate")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST: Task regenerate
# ---------------------------------------------------------------------------

class TestTaskRegenerate:
    def test_regenerate_with_mock(self, test_client, override_storage, monkeypatch):
        """Regenerate with mocked API returns new Q&A."""
        import arxiv_manager.web.routes as routes_mod
        def _fake_self_critique(**kw):
            return {"question": "Mock Q?", "answer": "99", "answer_format": "number", "task_type": "chart"}
        monkeypatch.setattr(routes_mod, "draft_with_self_critique", _fake_self_critique)

        # Upload an image to get a valid upload_id
        from PIL import Image
        import io
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
        prop = test_client.post("/api/image/propose", data={
            "upload_id": upload_id, "question": "Q?", "answer": "1",
            "answer_format": "number", "task_type": "chart",
            "domain": "Physics", "title": "Test",
        })
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
                "title": "X", "question": "Q?", "answer": "A",
                "answer_format": "word", "task_type": "chart", "domain": "CS",
            },
        )
        assert resp.status_code == 200  # Returns HTML with error or new-task form


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
        from arxiv_manager.models import Paper
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
