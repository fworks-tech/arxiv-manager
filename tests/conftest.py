"""Shared fixtures for all tests."""

import io
import os
import random
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from sqlmodel import SQLModel, Session, create_engine


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _make_image(width=200, height=200, content="blank") -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    if content == "blank":
        pass
    elif content == "chart":
        for y in range(height):
            for x in range(width):
                img.putpixel((x, y), (
                    (x * 255 // width) % 256,
                    (y * 255 // height) % 256,
                    128,
                ))
    elif content == "text_wall":
        for y in range(20, height - 20, 12):
            for x in range(10, width - 10):
                if random.random() < 0.45:
                    img.putpixel((x, y), (0, 0, 0))
    elif content == "sparse_banner":
        img = Image.new("RGB", (600, 50), (255, 255, 255))
    elif content == "photo":
        for y in range(height):
            for x in range(width):
                img.putpixel((x, y), (
                    (x * 200 // width) % 200 + 55,
                    (y * 200 // height) % 200 + 55,
                    100 + (x * y * 7) % 100,
                ))
    return img


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def sample_image_chart_path(tmp_path):
    """Returns path to a synthetic chart-like JPEG image."""
    img = _make_image(300, 300, "chart")
    path = tmp_path / "chart.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def sample_image_blank_path(tmp_path):
    """Returns path to a synthetic blank (mostly white) JPEG image."""
    img = _make_image(200, 200, "blank")
    path = tmp_path / "blank.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def sample_image_text_wall_path(tmp_path):
    """Returns path to a synthetic text-wall image."""
    img = _make_image(400, 400, "text_wall")
    path = tmp_path / "text_wall.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def sample_image_chart_bytes() -> bytes:
    """Returns raw JPEG bytes of a synthetic chart image."""
    return _image_to_bytes(_make_image(300, 300, "chart"))


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    """Creates a temp directory and returns the path for a test SQLite DB."""
    return tmp_path / "test.db"


@pytest.fixture
def db_engine(tmp_db_path):
    """Creates a file-based SQLite engine with all tables (thread-safe)."""
    from arxiv_manager.models import Task, Figure, Paper
    engine = create_engine(f"sqlite:///{tmp_db_path}", echo=False, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Provides a scoped DB session with automatic cleanup."""
    session = Session(db_engine)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_figure(db_session):
    """Creates and returns a Figure saved to the test DB."""
    from arxiv_manager.models import Figure
    fig = Figure(
        paper_id="9999.99999",
        page=1,
        width=600,
        height=400,
        figure_type="chart_graph_text",
        complexity_score=0.75,
        image_path="figures/test_figure.png",
        image_hash="abc123",
        is_dense=True,
        status="selected",
    )
    db_session.add(fig)
    db_session.commit()
    db_session.refresh(fig)
    return fig


@pytest.fixture
def sample_task(db_session, sample_figure, override_storage):
    """Creates and returns a Task saved to the test DB.
    A real image file is written to override_storage/figures/ so
    endpoints that need to read the image (regenerate, draft) work."""
    from arxiv_manager.models import Task
    # Create a real image file at the expected path
    from PIL import Image
    img_path = override_storage / "figures" / "test_figure.png"
    Image.new("RGB", (200, 200), (128, 128, 128)).save(img_path)
    task = Task(
        title="Test Task",
        figure_id=sample_figure.id,
        question="What is the ratio of the bar heights in panel A?",
        answer="3.5",
        answer_format="number",
        task_type="chart",
        domain="Computer Science",
        difficulty="challenging",
        status="draft",
        image_path="figures/test_figure.png",
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


@pytest.fixture
def sample_paper(db_session):
    """Creates and returns a Paper saved to the test DB."""
    from arxiv_manager.models import Paper
    paper = Paper(
        id="9999.99999",
        title="Test Paper for Unit Testing",
        authors="Test Author",
    )
    db_session.add(paper)
    db_session.commit()
    db_session.refresh(paper)
    return paper


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_api_key(monkeypatch):
    """Sets OPENCODE_API_KEY for tests that need it."""
    monkeypatch.setenv("OPENCODE_API_KEY", "test-api-key-12345")
    return "test-api-key-12345"


@pytest.fixture
def mock_no_api_key(monkeypatch):
    """Clears OPENCODE_API_KEY for tests that expect no key."""
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Storage path override
# ---------------------------------------------------------------------------

@pytest.fixture
def override_storage(tmp_path, monkeypatch):
    """Monkeypatches STORAGE_DIR and DB_PATH to point at temp directory."""
    from arxiv_manager import storage
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(storage, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(storage, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path / "_uploads")
    (tmp_path / "papers").mkdir(exist_ok=True)
    (tmp_path / "figures").mkdir(exist_ok=True)
    (tmp_path / "_uploads").mkdir(exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(override_storage, mock_api_key, tmp_db_path, monkeypatch):
    """Provides a FastAPI TestClient with temp storage, mocked API key,
    and a file-based test database."""
    from fastapi.testclient import TestClient
    # Point DATABASE_URL to the temp DB file before app starts
    import arxiv_manager.db as db_mod
    monkeypatch.setattr(db_mod, "DATABASE_URL", f"sqlite:///{tmp_db_path}")
    # Re-create engine with the new URL (needed before init_db)
    from sqlmodel import create_engine
    new_engine = create_engine(f"sqlite:///{tmp_db_path}", echo=False, connect_args={"check_same_thread": False})
    monkeypatch.setattr(db_mod, "engine", new_engine)
    from arxiv_manager.web.app import create_app
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_client_no_key(override_storage, mock_no_api_key, tmp_db_path, monkeypatch):
    """Provides a TestClient without API key set (shared DB)."""
    from fastapi.testclient import TestClient
    import arxiv_manager.db as db_mod
    monkeypatch.setattr(db_mod, "DATABASE_URL", f"sqlite:///{tmp_db_path}")
    from sqlmodel import create_engine
    new_engine = create_engine(f"sqlite:///{tmp_db_path}", echo=False, connect_args={"check_same_thread": False})
    monkeypatch.setattr(db_mod, "engine", new_engine)
    from arxiv_manager.web.app import create_app
    app = create_app()
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Mock for _call_opencode (routes tests that need AI draft)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_draft_success(monkeypatch):
    """Monkeypatches _call_opencode to return a valid draft dict."""
    def _fake_call(*args, **kwargs):
        return {
            "question": "What is the peak value in panel A?",
            "answer": "42",
            "answer_format": "number",
            "task_type": "chart",
        }
    import arxiv_manager.authoring.ai_draft as draft_mod
    monkeypatch.setattr(draft_mod, "_call_opencode", _fake_call)


@pytest.fixture
def mock_draft_empty(monkeypatch):
    """Monkeypatches _call_opencode to return None (generation failure)."""
    def _fake_call(*args, **kwargs):
        return None
    import arxiv_manager.authoring.ai_draft as draft_mod
    monkeypatch.setattr(draft_mod, "_call_opencode", _fake_call)
