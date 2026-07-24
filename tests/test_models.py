"""Tests for model instantiation and defaults."""

from arxiv_manager.models import Task, Figure, Paper, ImageStatus, TaskStatus


def test_task_defaults():
    """Task() has all default values."""
    t = Task()
    assert t.title == ""      # Has default ""
    assert t.question is None  # No default — must be provided
    assert t.answer is None    # No default
    assert t.answer_format == "word"  # Default from AnswerFormat.WORD
    assert t.task_type == "chart"     # Default from TaskType.CHART
    assert t.domain == "Computer Science"  # Default
    assert t.difficulty == ""
    assert t.status == TaskStatus.DRAFT.value
    assert t.image_path == ""
    assert t.rhea_override_notes == ""
    assert t.figure_id is None
    assert t.status == TaskStatus.DRAFT.value
    assert t.image_path == ""
    assert t.rhea_override_notes == ""
    assert t.figure_id is None  # FK, optional default


def test_task_custom_values():
    """Task() accepts and stores custom values."""
    t = Task(
        title="My Task",
        question="What is X?",
        answer="42",
        answer_format="number",
        task_type="chart",
        domain="Physics",
        difficulty="challenging",
        status="submitted",
        figure_id=1,
    )
    assert t.title == "My Task"
    assert t.answer == "42"
    assert t.status == "submitted"
    assert t.figure_id == 1


def test_figure_defaults():
    """Figure() has default values (image_path and paper_id required)."""
    f = Figure(paper_id="9999.99999", image_path="figures/t.png")
    assert f.paper_id == "9999.99999"
    assert f.image_path == "figures/t.png"
    assert f.figure_type == ""
    assert f.complexity_score == 0.0
    assert f.status == "new"
    assert f.is_dense is False
    assert f.width == 0
    assert f.height == 0


def test_figure_custom_values():
    """Figure() accepts and stores custom values."""
    f = Figure(
        paper_id="1234.56789",
        image_path="figures/chart.png",
        image_hash="abc123",
        page=2,
        figure_type="chart_graph_text",
        complexity_score=0.85,
        is_dense=True,
    )
    assert f.paper_id == "1234.56789"
    assert f.complexity_score == 0.85
    assert f.is_dense is True


def test_paper_defaults():
    """Paper() has required id and title fields."""
    p = Paper(id="9999.99999", title="Test Paper")
    assert p.id == "9999.99999"
    assert p.title == "Test Paper"
    assert p.license == "CC0"
    assert p.is_suitable is False


def test_image_status_enum():
    """ImageStatus enum values match expected string values."""
    assert ImageStatus.NEW == "new"
    assert ImageStatus.REVIEWED == "reviewed"
    assert ImageStatus.USED == "used"
    assert ImageStatus.REJECTED == "rejected"


def test_task_status_enum():
    """TaskStatus enum values match expected string values."""
    assert TaskStatus.DRAFT == "draft"
    assert TaskStatus.VALIDATED == "validated"
    assert TaskStatus.SUBMITTED == "submitted"
    assert TaskStatus.APPROVED == "approved"
    assert TaskStatus.REWORK == "rework"
