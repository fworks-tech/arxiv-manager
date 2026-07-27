"""Tests for model instantiation and defaults."""

from arxiv_manager.models import Figure, GenerationAttempt, ImageStatus, Paper, Task, TaskStatus


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
    assert ImageStatus.USED == "used"
    assert ImageStatus.REJECTED == "rejected"


def test_task_status_enum():
    """TaskStatus enum values match expected string values."""
    assert TaskStatus.DRAFT == "draft"
    assert TaskStatus.SUBMITTED == "submitted"


# ─── GenerationAttempt ───────────────────────────────────────────────


def test_generation_attempt_defaults():
    """GenerationAttempt() with minimum fields has sensible defaults."""
    a = GenerationAttempt(figure_id=1)
    assert a.figure_id == 1
    assert a.task_id is None
    assert a.attempt_number == 0
    assert a.generation_type == ""
    assert a.prompt_text_hash == ""
    assert a.prompt_version_id == ""
    assert a.success is False
    assert a.validation_quality == 0.0


def test_generation_attempt_custom_values():
    """GenerationAttempt stores custom values including new prompt fields."""
    a = GenerationAttempt(
        figure_id=1,
        task_id=2,
        attempt_number=3,
        generation_type="draft",
        prompt_template_name="CHALLENGING_PROMPT",
        prompt_text_hash="a1b2c3d4e5f6a1b2c3d4",
        prompt_version_id="CHALLENGING_PROMPT@a1b2c3d4e5f6",
        difficulty="challenging",
        success=True,
        validation_quality=90.0,
    )
    assert a.figure_id == 1
    assert a.task_id == 2
    assert a.attempt_number == 3
    assert a.prompt_text_hash == "a1b2c3d4e5f6a1b2c3d4"
    assert a.prompt_version_id == "CHALLENGING_PROMPT@a1b2c3d4e5f6"
    assert a.success is True
    assert a.validation_quality == 90.0


def test_generation_attempt_nullable_figure_id():
    """figure_id and task_id are nullable for draft-before-propose flow."""
    a = GenerationAttempt()
    assert a.figure_id is None
    assert a.task_id is None
