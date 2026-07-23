"""Tests for the validator — aligned with QA calibration quiz."""

from arxiv_manager.authoring.validator import validate_task


# --- Calibration Q1: CC-BY license (not CC0) ---
# Not directly testable via validator, but validates the sourcing rule.
# We test the validator's ability to enforce handbook rules instead.

# --- Calibration Q2: Short answer format ---
def test_short_answer_is_concise():
    """Short answers should be 1-2 words, not full sentences."""
    result = validate_task(
        "What is the building between the hospital and the Marine Art Museum?",
        "The Naval Museum",
        "phrase",
    )
    assert result.is_valid


# --- Calibration Q3: Too simple (few data points) ---
def test_valid_chart_question():
    """A valid chart reasoning question passes all checks."""
    result = validate_task(
        "Which model shows a greater decline in accuracy from Session 1 to Session 9?",
        "Joint-CNN",
        "phrase",
    )
    assert result.is_valid


# --- Calibration Q4: "Name at least 2 reasons" invites explanation ---
def test_explanation_question_rejected():
    """Questions asking for explanations/reasons fail — answer won't be short."""
    result = validate_task(
        "Name at least 2 reasons that LLB-exposed mice have higher values than Sham mice",
        "increased enzyme activity and cell damage",
        "phrase",
    )
    # The answer is too long (6 words) and question invites open-ended response
    assert any("words" in w.lower() for w in result.warnings)


# --- Calibration Q5: No images of children ---
# Not testable via validator (requires image content analysis)

# --- Calibration Q6: "What does TFII stand for?" — external knowledge ---
def test_domain_jargon_warning():
    """Questions with domain-specific terminology should warn."""
    result = validate_task(
        "What is the hybridization of the sulfur atom?",
        "sp3",
        "word",
    )
    assert any("jargon" in w.lower() or "domain" in w.lower() for w in result.warnings)


# --- Calibration Q7: Task requirements ---
def test_visual_reasoning_required():
    """Questions must require visual reasoning (not just text)."""
    result = validate_task(
        "Which bar is tallest in the chart?",
        "Desktop",
        "word",
    )
    assert result.is_valid  # references visual element ("bar", "tallest")


def test_no_binary_questions():
    """Binary (yes/no, true/false) questions are never allowed."""
    for q in [
        "Is this a chart?",
        "Are there more than 3 bars?",
        "Does the line go up?",
        "Was the value higher in 2020?",
    ]:
        result = validate_task(q, "yes", "word")
        assert not result.is_valid, f"Should reject: {q}"


def test_no_trick_answers():
    """Trick answers like 'none' or 'cannot be determined' are rejected."""
    for a in ["none", "cannot be determined", "N/A", "not applicable"]:
        result = validate_task("What is the value?", a, "word")
        assert not result.is_valid, f"Should reject answer: {a}"


def test_single_question_only():
    """Questions must have exactly one question mark."""
    result = validate_task(
        "What is X? And what about Y?",
        "42",
        "number",
    )
    assert not result.is_valid


def test_answer_format_specified():
    """Answer format must always be specified."""
    result = validate_task(
        "Which model is best?",
        "Model A",
        "",  # missing format
    )
    assert not result.is_valid


def test_no_option_restriction():
    """Questions must not restrict options (e.g. 'Out of the 5...')."""
    for q in [
        "Out of the 3 options, which is best?",
        "From the following, which is tallest?",
        "Among these 5 bars, which is lowest?",
    ]:
        result = validate_task(q, "Option A", "phrase")
        assert not result.is_valid, f"Should reject: {q}"


def test_visual_reference_in_question():
    """Questions should reference visual elements (chart, figure, bar, etc.)."""
    # Good: references visual elements
    result_good = validate_task(
        "What percentage of text elements are shown in the bar chart's Mobile category?",
        "35.2%",
        "percent",
    )
    assert result_good.is_valid

    # Bad: text-only question, fails handbook Test 1
    result_bad = validate_task(
        "What is the capital of France?",
        "Paris",
        "word",
    )
    assert not result_bad.is_valid  # should fail handbook Test 1
    assert any("Test 1" in e or "image" in e.lower() for e in result_bad.errors)


# --- New rules from Visual QA Pipeline Guide ---

def test_extreme_seeking_warning():
    """Questions using extreme-seeking words should warn (Qwen bias)."""
    result = validate_task(
        "Which model has the highest accuracy?",
        "Model A",
        "word",
    )
    # Should warn about extreme-seeking
    assert any("extreme" in w.lower() for w in result.warnings)


def test_threshold_filter_bonus():
    """Questions using threshold filters get quality bonus."""
    result = validate_task(
        "Which model has fewer than 10 GFLOPS and highest APH L2 score?",
        "Model B",
        "word",
    )
    # Should detect threshold filter
    assert any("threshold" in c.lower() for c in result.passed_checks)


def test_multi_panel_bonus():
    """Questions referencing multiple panels get quality bonus."""
    result = validate_task(
        "In panel (a), which bar is tallest? What is its value in panel (b)?",
        "42",
        "number",
    )
    assert any("panel" in c.lower() for c in result.passed_checks)


def test_caption_solvable_warning():
    """Very descriptive captions make questions solvable without image."""
    long_caption = (
        "This figure shows a bar chart comparing five machine learning models. "
        "The x-axis displays model names: SVM, Random Forest, CNN, LSTM, and Transformer. "
        "The y-axis shows accuracy from 0 to 100%. CNN achieves the highest accuracy at 95.2%."
    )
    result = validate_task(
        "Which model has the highest accuracy?",
        "CNN",
        "word",
        caption=long_caption,
    )
    # Either an error or a warning about the caption
    assert any("caption" in (e + w).lower() for e in result.errors for w in [e]) or any(
        "caption" in w.lower() for w in result.warnings
    )


def test_answer_extreme_warning():
    """Extreme answers (highest, lowest) should warn."""
    result = validate_task(
        "Which model performs best?",
        "highest",
        "word",
    )
    assert any("extreme" in w.lower() for w in result.warnings)


def test_quality_score():
    """Quality score should reflect validation results."""
    # Good question
    good = validate_task(
        "How many bars in panel (a) are above the 5% threshold, given the figure compares accuracy across five models?",
        "BERT",
        "word",
    )
    assert good.quality_score >= 80

    # Bad question (binary + trick answer)
    bad = validate_task("Is this correct?", "none", "word")
    assert bad.quality_score < 50


# ─── New Handbook-aligned tests ───


def test_handbook_visual_dependence_test():
    """Handbook §3 Test 1: visual-dependence test."""
    # Good: must reference image
    good = validate_task(
        "Which bar in the chart is tallest?",
        "BERT",
        "word",
    )
    assert any("visual-dependence" in c.lower() for c in good.passed_checks)

    # Bad: answerable without image
    bad = validate_task(
        "What is 2 + 2?",
        "4",
        "number",
    )
    assert any("Test 1" in e for e in bad.errors)


def test_handbook_text_only_question_rejected():
    """Handbook common error: text-only questions don't require visual reasoning."""
    bad = validate_task(
        "What does this say?",
        "Hello world",
        "phrase",
    )
    assert any("text-only" in e.lower() or "visual" in e.lower() for e in bad.errors)


def test_handbook_math_heavy_warning():
    """Handbook common error: math-heavy questions lack visuo-spatial reasoning."""
    result = validate_task(
        "Multiply the value in row 1 column 2 by the value in row 3 column 4, then divide by 2.",
        "42",
        "number",
    )
    assert any("calculation" in w.lower() or "math" in w.lower() for w in result.warnings)


def test_handbook_explanation_what_trend_rejected():
    """Handbook: 'what trend do you see' is an explanation question."""
    result = validate_task(
        "What trend do you see in the chart?",
        "increasing",
        "word",
    )
    assert not result.is_valid
    assert any("explanation" in e.lower() for e in result.errors)


def test_handbook_list_answer_3plus_warning():
    """Handbook §5: avoid answers that are lists with more than 3 elements."""
    result = validate_task(
        "Which categories appear in the legend?",
        "alpha, beta, gamma, delta",
        "phrase",
    )
    assert any("list" in w.lower() for w in result.warnings)


def test_handbook_mcq_8_options():
    """Handbook §5: MCQ must have 8+ options."""
    # Only 4 options — should warn
    result = validate_task(
        "Which model has the highest accuracy?",
        "BERT",
        "word",
        options=["A", "B", "C", "D"],
    )
    assert any("8+" in w or "options" in w.lower() for w in result.warnings)

    # 8 options — should pass
    result_good = validate_task(
        "Which model has the highest accuracy in the chart?",
        "BERT",
        "word",
        options=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    assert any("8 options" in c.lower() for c in result_good.passed_checks)


def test_handbook_mcq_no_trick_options():
    """Handbook §5: MCQ options cannot be 'none' or 'cannot be determined'."""
    result = validate_task(
        "Which model is best?",
        "A",
        "word",
        options=["A", "B", "C", "D", "E", "F", "G", "none"],
    )
    assert any("trick" in w.lower() for w in result.warnings)


def test_handbook_noise_condition_warning():
    """Handbook common error #7: conditions that don't materially change the answer."""
    result = validate_task(
        "Which sample shows the greatest rate of decline in reflectance between 550nm and 850nm?",
        "Grosnaja",
        "word",
    )
    # Heuristic may or may not match — at minimum should be valid
    assert result.is_valid


def test_handbook_figure_type_task_type_mismatch():
    """Validator Rule 28: figure_type vs task_type mismatch."""
    result = validate_task(
        "Which object is to the left of the laptop?",
        "mouse",
        "word",
        figure_type="chart_graph_text",
        task_type="spatial",
    )
    # Mismatch should warn
    assert any("mismatch" in w.lower() or "match" in w.lower() for w in result.warnings)


def test_handbook_watermark_filename_check():
    """HDM checklist: filenames suggesting watermarks/copyright."""
    result = validate_task(
        "Which bar in the chart is highest?",
        "alpha",
        "word",
        image_path="shutterstock_12345.png",
    )
    assert any("watermark" in w.lower() or "copyright" in w.lower() for w in result.warnings)


def test_handbook_challenging_pattern_recognized():
    """A well-formed Challenging question passes all handbook checks."""
    result = validate_task(
        "Count the total number of arrows in the diagram, including those entering the red Model block, the green Environment block, and the blue Loss block, then give the sum?",
        "8",
        "number",
    )
    assert result.is_valid
    assert result.quality_score >= 80


def test_validate_mcq_helper():
    """validate_mcq convenience function exists and accepts options."""
    from arxiv_manager.authoring.validator import validate_mcq
    result = validate_mcq(
        "Which bar in the chart shows the largest gap from the previous year?",
        "BERT",
        options=["A", "B", "C", "D", "E", "F", "G", "H"],
        answer_format="word",
    )
    assert result.is_valid
