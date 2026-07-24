"""Tests for database initialization and operations."""

import pytest
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy import text


@pytest.fixture
def fresh_engine():
    """Creates a fresh in-memory engine with no tables."""
    engine = create_engine("sqlite://", echo=False)
    return engine


def test_init_db_creates_tables(fresh_engine, monkeypatch):
    """init_db creates all tables."""
    monkeypatch.setattr("arxiv_manager.db.engine", fresh_engine)
    from arxiv_manager.db import init_db
    init_db()
    # Check tables exist
    tables = SQLModel.metadata.tables.keys()
    assert "tasks" in tables
    assert "figures" in tables
    assert "papers" in tables


def test_get_session_crud(fresh_engine, monkeypatch):
    """Basic CRUD operations via get_session."""
    monkeypatch.setattr("arxiv_manager.db.engine", fresh_engine)
    from arxiv_manager.db import init_db, get_session
    init_db()
    from arxiv_manager.models import Task, Figure

    # Create
    s1 = get_session()
    fig = Figure(paper_id="9999.99999", page=1, image_path="figures/t.png", image_hash="abc")
    s1.add(fig)
    s1.commit()
    s1.refresh(fig)

    t = Task(question="Test?", answer="42", answer_format="number", figure_id=fig.id)
    s1.add(t)
    s1.commit()
    task_id = t.id
    s1.close()

    # Read
    s2 = get_session()
    read = s2.get(Task, task_id)
    assert read is not None
    assert read.question == "Test?"
    assert read.answer == "42"
    s2.close()

    # Update
    s3 = get_session()
    upd = s3.get(Task, task_id)
    upd.answer = "43"
    s3.add(upd)
    s3.commit()
    s3.close()

    # Verify update
    s4 = get_session()
    verified = s4.get(Task, task_id)
    assert verified.answer == "43"
    s4.close()


def test_migration_adds_column(fresh_engine, monkeypatch):
    """_migrate adds rhea_override_notes column if missing."""
    monkeypatch.setattr("arxiv_manager.db.engine", fresh_engine)
    from arxiv_manager.db import init_db
    init_db()

    # Verify the column exists after init_db
    with fresh_engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(tasks)")).fetchall()]
        assert "rhea_override_notes" in cols


def test_migration_idempotent(fresh_engine, monkeypatch):
    """_migrate is safe to run multiple times."""
    monkeypatch.setattr("arxiv_manager.db.engine", fresh_engine)
    from arxiv_manager.db import init_db, _migrate
    init_db()       # Creates tables
    _migrate()      # First migration (should be no-op)
    _migrate()      # Second call should not raise


def test_task_query_by_status(fresh_engine, monkeypatch):
    """Filter tasks by status field."""
    monkeypatch.setattr("arxiv_manager.db.engine", fresh_engine)
    from arxiv_manager.db import init_db, get_session
    from arxiv_manager.models import Task, Figure, Paper
    from sqlmodel import select
    init_db()

    s = get_session()
    paper = Paper(id="9999.99999", title="Test")
    s.add(paper)
    s.commit()
    fig = Figure(paper_id=paper.id, page=1, image_path="figures/t.png", image_hash="abc")
    s.add(fig)
    s.commit()

    s.add(Task(question="Q1", answer="a1", answer_format="word", status="draft", figure_id=fig.id))
    s.add(Task(question="Q2", answer="a2", answer_format="number", status="submitted", figure_id=fig.id))
    s.add(Task(question="Q3", answer="a3", answer_format="word", status="draft", figure_id=fig.id))
    s.commit()

    drafts = s.exec(select(Task).where(Task.status == "draft")).all()
    assert len(drafts) == 2
    submitted = s.exec(select(Task).where(Task.status == "submitted")).all()
    assert len(submitted) == 1
    s.close()


def test_figure_query_by_paper(fresh_engine, monkeypatch):
    """Filter figures by paper_id."""
    monkeypatch.setattr("arxiv_manager.db.engine", fresh_engine)
    from arxiv_manager.db import init_db, get_session
    from arxiv_manager.models import Figure, Paper
    from sqlmodel import select
    init_db()

    s = get_session()
    paper1 = Paper(id="1111.11111", title="P1")
    paper2 = Paper(id="2222.22222", title="P2")
    s.add(paper1)
    s.add(paper2)
    s.commit()

    s.add(Figure(paper_id=paper1.id, page=1, image_path="figures/a.png", image_hash="aaa"))
    s.add(Figure(paper_id=paper1.id, page=2, image_path="figures/b.png", image_hash="bbb"))
    s.add(Figure(paper_id=paper2.id, page=1, image_path="figures/c.png", image_hash="ccc"))
    s.commit()

    figs = s.exec(select(Figure).where(Figure.paper_id == paper1.id)).all()
    assert len(figs) == 2
    figs2 = s.exec(select(Figure).where(Figure.paper_id == paper2.id)).all()
    assert len(figs2) == 1
    s.close()
