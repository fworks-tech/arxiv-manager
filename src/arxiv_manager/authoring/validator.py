"""Handbook rule validation engine — aligned with Rhea review criteria.

Sources:
  - QA Expert Handbook
  - arxiv-manager SKILL.md (challenging-question-generator)
  - docs/qwen_weaknesses.md (Qwen 3.6-35B-A3B exploitation strategies)
  - docs/figure_suggestions.md (figure-type patterns)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a task against handbook rules."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100, higher = better Rhea pass chance

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  ❌ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        if not lines:
            lines.append("  ✅ All checks passed")
        lines.append(f"\n  Quality score: {self.quality_score:.0f}/100")
        return "\n".join(lines)

    def rhea_summary(self) -> str:
        """Rhea-style review summary."""
        lines = []
        for check in self.passed_checks:
            lines.append(f"  ✅ {check}")
        for e in self.errors:
            lines.append(f"  ❌ {e}")
        for w in self.warnings:
            lines.append(f"  ⚠️  {w}")
        lines.append(f"\n  Quality score: {self.quality_score:.0f}/100")
        return "\n".join(lines)


# --- Binary / T-F detection ---

BINARY_PATTERNS = [
    r"^(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\s+.+\?$",
    r"^(true|false)\s",
    r"^(yes|no)\s",
    r"\b(true or false)\b",
    r"\b(yes or no)\b",
]

# "How does X work?" / "What trend do you see?" — explanation questions
EXPLANATION_PATTERNS = [
    r"^(explain|describe|how does|how do|why does|why do|what is the mechanism|what trend|how can you|how would you)",
    r"\b(explain|describe|discuss|elaborate)\b",
    r"\bwhat trend\b",
    r"\bhow (?:does|do|can|would)\b",
]

TRICK_ANSWERS = {"none", "cannot be determined", "n/a", "not applicable", "no answer", "unclear", "none of the above"}

RESTRICT_PATTERNS = [
    r"out of (?:the )?\d+",
    r"from (?:the )?(?:following|these|below)",
    r"(?:which|among) (?:of )?(?:the )?(?:following|these|below)",
]

# Math-heavy: focus on calculation instead of visuo-spatial reasoning
MATH_HEAVY_PATTERNS = [
    r"\b(?:multiply|divide|subtract|add)\b",
    r"\b(?:calculate|compute|solve)\b",
    r"\b(?:sum|product|quotient|difference)\s+of\s+",
    r"\b\d+\s*[\*\/×÷]\s*\d+\b",
]

# Text-heavy: focuses on text reading rather than visual reasoning
TEXT_HEAVY_PATTERNS = [
    r"^what does (?:this|the|it) say",
    r"^what is written",
    r"^what (?:does|is) the (?:text|caption|quote|title|heading|label) say",
]

# Long-winded/awkward wording (handbook error)
LONG_WINDED_INDICATORS = [
    r"\bthe (?:one|item) (?:that|who) (?:is|has|contains) .*? and (?:is|has|contains)",
    r"^(?:the|a|an)\s+\w+\s+(?:that|who|which)\s+\w+\s+\w+",
]

# Tables-only (handbook limits to 1-2 submissions)
TABLE_LIMITS = "Tables (text + arithmetic, little visuospatial reasoning) limited to 1-2 submissions"

DOMAIN_JARGON = [
    r"\bhybridization\b", r"\bsigma bond\b", r"\bpi bond\b",
    r"\bLUMO\b", r"\bHOMO\b", r"\bsp[23]\b",
    r"\bEBITDA\b", r"\bWACC\b", r"\bDCF\b",
    r"\bp-value\b", r"\bchi.?square\b",
]

REASONING_INDICATORS = [
    r"\b(highest|lowest|most|least|fewest|greatest)\b.*\b(highest|lowest|most|least|fewest|greatest)\b",
    r"\b(also|and|while|whereas|compared|between|versus|vs)\b",
    r"\b(rank|order|sort|compare|relative|ratio|factor)\b",
    r"\b(first|second|third|top|bottom)\b.*\b(first|second|third|top|bottom)\b",
    r"\b(sum|total|average|difference|ratio|factor|proportion)\b",
    r"\b(larger|smaller|greater|less|fewer)\s+than\b",
    r"\b(magnitude)\b",
]

# Conditions that may not materially change the answer (handbook error #7)
NOISE_CONDITION_PATTERNS = [
    r"\bbetween\s+[\d.]+\s*(?:nm|mm|cm|m|kg|lb|s|sec|min|hr|h|°[CFK])\s+and\s+[\d.]+",
    r"\bover\s+all\s+wavelengths\b",
    r"\bacross\s+all\b",
]

# Extreme-seeking words (Qwen bias to exploit)
EXTREME_SEEKING = [
    r"\b(highest|lowest|largest|smallest|most|least|fewest|greatest)\b",
    r"\b(maximum|minimum|max|min)\b",
    r"\b(leftmost|rightmost|topmost|bottommost)\b",
    r"\b(best|worst|strongest|weakest)\b",
]

# Threshold filter patterns (good for creating difficulty)
THRESHOLD_PATTERNS = [
    r"\b(fewer|less|more|greater|above|below|over|under|exactly|closest to|nearest to)\s+(?:than\s+)?\d",
    r"\b(between|within|around)\s+\d",
    r"\b(?:less|more)\s+than\s+\d",
]

# Multi-panel reference (good practice)
MULTI_PANEL_PATTERNS = [
    r"\bpanel\s*\([a-z]\)",
    r"\bfigure\s*\d",
    r"\b(a)\b.*\b(b)\b",
    r"\bleft\b.*\bright\b",
    r"\btop\b.*\bbottom\b",
]

# Chart anti-patterns: questions that count chart FURNITURE (labels, ticks, colorbars)
# rather than DATA. These are mechanical OCR tasks that Qwen 3.6 solves easily.
# Note: pattern is "X labels/marks/ticks/numbers" where X is a chart furniture word.
# "z-axis value" or "axis at x=5" are NOT matched (those are data questions).
CHART_FURNITURE_ANTI_PATTERNS = [
    (r"\b(?:tick|axis|colorbar|legend)\s+(?:labels?|marks?|ticks?|numbers?)\b", "axis/label/tick counting"),
    (r"\b(?:count|how\s+many)\s+(?:the\s+)?(?:tick|axis|colorbar|legend)\b", "axis counting"),
    (r"\b(?:labeled|numerical)\s+(?:tick|values?|labels?|numbers?)\b.*\b(?:axis|colorbar|legend|tick)\b", "labeled value counting"),
]

# Chart data references: phrases that indicate the question is about actual data
CHART_DATA_REFS = [
    r"\b(?:peak|maximum|minimum|max|min|valley|trough)\b",
    r"\b(?:x|y|z)\s*(?:-\s*axis|axis)?\s*(?:value|coordinate|position|at)\b",
    r"\b(?:surface|curve|bar|line|series|column|histogram)\b",
    r"\b(?:panel|figure)\s*[ab]\b.*\b(?:panel|figure)\s*[ab]\b",
    r"\b(?:ratio|difference|sum|total|average|mean)\b.*\b(?:panel|figure|between|across)\b",
    r"\b(?:exceed|above|below|greater|less|threshold|over|under)\s+(?:than\s+)?-?\d",
    r"\b(?:at\s+(?:the\s+)?(?:x|y|t)\s*=\s*-?\d)",
    r"\b(?:steepest|flattest|highest|lowest)\s+(?:point|value|region|peak)\b",
    r"\b(?:gradient|slope|derivative)\b",
]

# Generic simple-count questions: "How many X are in the image?" without
# filter, comparison, or arithmetic. Always too easy for Qwen.
GENERIC_COUNT_PATTERNS = [
    r"^how\s+many\s+[\w\s]+\s+(?:are|appear|exist|visible)\s+(?:in\s+)?(?:the\s+)?(?:image|figure|chart|diagram|plot)\s*[\?\.]?$",
    r"^count\s+(?:the\s+)?(?:total\s+)?(?:number\s+of\s+)?[\w\s]+(?:in|across)\s+(?:the\s+)?(?:image|figure|chart|diagram)\s*[\?\.]?$",
]

# Spatial reasoning patterns (general image type)
SPATIAL_PATTERNS = [
    r"\bto the (?:left|right) of\b",
    r"\bclosest to (?:the )?camera\b",
    r"\bhighest in the image\b",
    r"\bsitting on top of\b",
    r"\bbetween\b.*\band\b",
    r"\bblocking\b",
    r"\bimag(?:ine|ining) (?:you|yourself)\b",
    r"\bfacing (?:toward|towards)\b",
    r"\bfrom (?:the )?doorway\b",
]


# ─── The Two Tests (handbook §3) ───────────────────────────────────


def _passes_visual_dependence_test(q: str) -> bool:
    """Handbook §3 Test 1: Could a smart person answer without the image?

    Simple heuristics: question must reference visual elements
    AND not be solvable from general knowledge / question text alone.
    """
    q_lower = q.lower()
    # If question only mentions text/captions and no visual elements, fails
    if TEXT_HEAVY_PATTERNS and any(re.search(p, q_lower) for p in TEXT_HEAVY_PATTERNS):
        return False
    # If no visual reference at all, fails
    visual_refs = [
        "chart", "graph", "figure", "image", "diagram", "plot",
        "table", "panel", "bar", "line", "pie", "color", "shape",
        "object", "left", "right", "top", "bottom", "above", "below",
        "next to", "between", "behind", "front", "show", "display",
        "axis", "label", "legend", "title", "y-axis", "x-axis",
        "grid", "row", "column", "cell", "tile", "circle", "square",
        "highlighted", "marked", "circled", "indicated", "pointed",
    ]
    return any(ref in q_lower for ref in visual_refs)


def _passes_one_answer_test(q: str, a: str) -> bool:
    """Handbook §3 Test 2: Could two reasonable people give different answers?

    Heuristic: answer must be specific (number, single word, exact phrase).
    """
    a_lower = a.strip().lower()
    # Open-ended answers indicate ambiguity
    if a_lower in {"varies", "maybe", "it depends", "unclear"}:
        return False
    # Long explanations = ambiguous
    if len(a.split()) > 4:
        return False
    return True


def _is_caption_solvable(caption: str, q: str) -> bool:
    """Handbook §3 + Common Error #5: Caption should not be the only source."""
    if not caption or len(caption) < 150:
        return False
    # If question asks "what does caption say" — caption is the only source
    if re.search(r"caption", q.lower()):
        return True
    return False


# ─── MCQ format checks (handbook §5) ──────────────────────────────


def _check_mcq_options(options: list[str] | None) -> list[str]:
    """Check MCQ options meet handbook rules.

    Rules:
      - 8+ options
      - all same format (number/word/phrase)
      - plausible distractors
      - no 'none' / 'cannot be determined' options
    """
    issues = []
    if not options:
        return issues
    if len(options) < 8:
        issues.append(f"MCQ has {len(options)} options; handbook requires 8+")
    for opt in options:
        if opt.strip().lower() in TRICK_ANSWERS:
            issues.append(f"MCQ option '{opt}' is a trick answer (handbook ban)")
    return issues


def _answer_is_list_of_three_plus(a: str) -> bool:
    """Handbook §5: avoid answers that are lists with more than 3 short elements."""
    # Detect comma-separated lists, "and" lists
    parts = re.split(r",|\band\b", a)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 3:
        return True
    return False


# ─── Copyright / watermark hint (handbook HDM checklist) ───────────


WATERMARK_HINTS = [
    r"shutterstock", r"getty", r"©", r"all rights reserved",
    r"istock", r"alamy", r"watermark", r"stock photo",
]


def validate_task(
    question: str,
    answer: str,
    answer_format: str = "",
    image_path: str = "",
    caption: str = "",
    options: list[str] | None = None,
    figure_type: str = "",
    task_type: str = "",
) -> ValidationResult:
    """Validate a task against all handbook rules."""
    logger.info("validate_task entry q_len=%d a_len=%d fmt=%s ftype=%s ttype=%s",
                len(question), len(answer), answer_format, figure_type, task_type)
    result = ValidationResult()
    q = question.strip()
    a = answer.strip().lower()

    # --- Rule 1: No binary/T-F questions ---
    if _is_binary_question(q):
        result.errors.append("Binary/T-F question is not allowed")
    else:
        result.passed_checks.append("Question is not binary/T-F")

    # --- Rule 2: Answer format should be specified ---
    if not answer_format:
        result.errors.append("Answer format not specified (e.g. number, word, phrase)")
    else:
        result.passed_checks.append("Answer format is specified")

    # --- Rule 3: Answer should be short ---
    word_count = len(a.split())
    if word_count > 4:
        result.warnings.append(f"Answer has {word_count} words — prefer 1-2 words (max 4)")
    else:
        result.passed_checks.append(f"Answer is concise ({word_count} word{'s' if word_count != 1 else ''})")
    if len(a) > 50:
        result.errors.append(f"Answer too long ({len(a)} chars)")

    # --- Rule 4: No trick answers ---
    if a in TRICK_ANSWERS:
        result.errors.append(f"Trick answer '{a}' is not allowed")
    else:
        result.passed_checks.append("Answer is not a trick answer")

    # --- Rule 5: Single question (not multiple ?) ---
    stripped = q.rstrip(".!?")
    ending = q[len(stripped):] if len(stripped) < len(q) else ""
    if q.count("?") > 1:
        result.errors.append(f"Multiple questions detected ({q.count('?')} question marks)")
    elif not ending:
        result.errors.append("Question must end with punctuation ('?' or '.')")
    else:
        result.passed_checks.append(f"Ends with '{ending}'")

    # --- Rule 6: Question length ---
    sentences = _count_sentences(q)
    if sentences > 2:
        result.warnings.append(f"Question has {sentences} sentences — prefer 1-2")
    else:
        result.passed_checks.append("Question is concise")

    # --- Rule 7: No option restriction in question ---
    if _restricts_options(q):
        result.errors.append("Don't restrict options in question (e.g. 'Out of the 3...')")
    else:
        result.passed_checks.append("No option restriction in question")

    # --- Rule 8: No domain-specific jargon ---
    if _has_domain_jargon(q):
        result.warnings.append("Contains domain-specific terminology — rewrite for general audience")
    else:
        result.passed_checks.append("No domain-specific jargon")

    # --- Rule 9: Question references image content ---
    if not _references_visual_content(q):
        result.warnings.append("Question may not require the image to answer")
    else:
        result.passed_checks.append("Question references visual content")

    # --- Rule 10: Answer format consistency ---
    if answer_format == "number" and not _is_number(a):
        result.warnings.append(f"Answer format is 'number' but answer '{a}' doesn't look numeric")
    elif answer_format == "number":
        result.passed_checks.append("Answer matches declared format")
    if answer_format == "percent" and "%" not in a:
        result.warnings.append("Answer format is 'percent' but answer missing '%'")

    # --- Rule 11: No explanation questions (handbook "how/what trend") ---
    if _is_explanation_question(q):
        result.errors.append("Explanation questions ('Explain how...' / 'What trend...') are not allowed")
    else:
        result.passed_checks.append("Not an explanation question")

    # --- Rule 12: Question complexity (multi-step reasoning) ---
    if _has_reasoning_depth(q):
        result.passed_checks.append("Question requires multi-step reasoning")
    else:
        result.warnings.append("Question may be too simple — consider adding comparison or ranking")

    # --- Rule 12b: Chart-specific anti-patterns (counting chart furniture) ---
    if figure_type in ("chart_graph_text", "chart") or task_type == "chart":
        anti_pattern_hits = _matches_chart_anti_pattern(q)
        if anti_pattern_hits:
            for hit in anti_pattern_hits:
                result.errors.append(
                    f"Chart anti-pattern: '{hit}' — chart questions must reference data values, not just chart furniture (labels/ticks/colorbars)"
                )
        else:
            if _references_chart_data(q):
                result.passed_checks.append("References chart data (axis values, peaks, regions) — not just furniture")
            else:
                result.warnings.append(
                    "Chart question may not reference actual data — consider referencing axis values, peaks, regions, or cross-panel comparisons"
                )

    # --- Rule 12c: Generic simple-count check (no filter/comparison) ---
    if _is_generic_count_question(q):
        result.errors.append(
            "Question is a generic count ('How many X in the image?') without a filter, comparison, or arithmetic — too easy for Qwen"
        )

    # --- Rule 13: Answer derivability ---
    if _answer_seems_derivable(q, a):
        result.passed_checks.append("Answer appears derivable from question")
    else:
        result.warnings.append("Answer may not be clearly derivable from the question")

    # --- Rule 14: Grammar basics ---
    grammar_issues = _check_grammar(q)
    if grammar_issues:
        for issue in grammar_issues:
            result.warnings.append(issue)
    else:
        result.passed_checks.append("Basic grammar checks passed")

    # --- Rule 15: Extreme-seeking warning (Qwen bias) ---
    if _has_extreme_seeking(q):
        result.warnings.append("Uses extreme-seeking words (highest/lowest/most) — Qwen checks these first; consider threshold filters instead")
    else:
        result.passed_checks.append("No extreme-seeking bias detected")

    # --- Rule 16: Threshold filter detection (good practice) ---
    if _has_threshold_filter(q):
        result.passed_checks.append("Uses threshold filters (creates genuine visual complexity)")

    # --- Rule 17: Multi-panel reference (good practice) ---
    if _references_multi_panel(q):
        result.passed_checks.append("References multiple panels (cross-panel reasoning)")

    # --- Rule 18: Caption-solvable check ---
    if caption and _is_caption_solvable(caption, q):
        result.warnings.append("Caption is very descriptive / question asks about caption — image may not be required")
    elif caption:
        result.passed_checks.append("Caption is not overly descriptive")

    # --- Rule 19: Answer is intermediate (not extreme) ---
    if _answer_is_extreme(a):
        result.warnings.append("Answer is an extreme value (highest/lowest) — intermediate values are harder for models")

    # ─── NEW Handbook rules (ArXiv QA Expert Handbook) ───

    # --- Rule 20: The two tests (handbook §3) ---
    if not _passes_visual_dependence_test(q):
        result.errors.append("Test 1 FAILED: A smart person could answer this without the image")
    else:
        result.passed_checks.append("Passes visual-dependence test (handbook §3)")
    if not _passes_one_answer_test(q, a):
        result.warnings.append("Test 2 WARNING: Answer may be subjective / two reasonable people could give different answers")
    else:
        result.passed_checks.append("Passes one-answer test (handbook §3)")

    # --- Rule 21: Math-heavy (handbook common error) ---
    if any(re.search(p, q, re.IGNORECASE) for p in MATH_HEAVY_PATTERNS):
        result.warnings.append("Question focuses on calculation rather than visuo-spatial reasoning (handbook common error)")
    else:
        result.passed_checks.append("Question is visuo-spatial, not pure calculation")

    # --- Rule 22: Text-heavy (handbook common error) ---
    if any(re.search(p, q, re.IGNORECASE) for p in TEXT_HEAVY_PATTERNS):
        result.errors.append("Question is text-only — does not require visual reasoning (handbook common error)")
    else:
        result.passed_checks.append("Question is not text-only")

    # --- Rule 23: Long-winded/awkward (handbook common error) ---
    if any(re.search(p, q, re.IGNORECASE) for p in LONG_WINDED_INDICATORS):
        result.warnings.append("Question is long-winded/awkward — rewrite for clarity (handbook common error)")

    # --- Rule 24: Noise conditions (handbook common error #7) ---
    if any(re.search(p, q, re.IGNORECASE) for p in NOISE_CONDITION_PATTERNS):
        result.warnings.append("Question has a condition that may not materially change the answer (handbook error #7)")

    # --- Rule 25: List answer with 3+ elements (handbook §5) ---
    if _answer_is_list_of_three_plus(a):
        result.warnings.append("Answer is a list with more than 3 short elements — handbook §5 bans this")

    # --- Rule 26: MCQ options check (handbook §5) ---
    if options:
        mcq_issues = _check_mcq_options(options)
        for issue in mcq_issues:
            result.warnings.append(issue)
        if not any("MCQ" in i for i in mcq_issues):
            result.passed_checks.append(f"MCQ has {len(options)} options meeting handbook §5")

    # --- Rule 27: Watermark / copyright hint (handbook HDM checklist) ---
    # Best-effort filename check; full detection would require image analysis
    if image_path:
        lower_path = image_path.lower()
        for hint in WATERMARK_HINTS:
            if re.search(hint, lower_path):
                result.warnings.append("Filename suggests potential watermark/copyright — verify CC0 license")

    # --- Rule 28: Image-type / task-type match ---
    # If figure was classified as general_image but task_type is chart, warn
    if figure_type and task_type:
        mismatch = (
            (figure_type == "general_image" and task_type in ("chart",))
            or (figure_type == "chart_graph_text" and task_type == "spatial")
        )
        if mismatch:
            result.warnings.append(
                f"figure_type='{figure_type}' may not match task_type='{task_type}'"
            )
        else:
            result.passed_checks.append("figure_type matches task_type")

    # --- Calculate quality score ---
    result.quality_score = _calculate_score(result)

    logger.info("validate_task result valid=%s score=%.1f errors=%d warnings=%d",
                result.is_valid, result.quality_score, len(result.errors), len(result.warnings))
    return result


def validate_mcq(
    question: str,
    answer: str,
    options: list[str],
    answer_format: str = "word",
    image_path: str = "",
    caption: str = "",
) -> ValidationResult:
    """Convenience wrapper for MCQ validation including options check.

    Per handbook §5, MCQ must have 8+ options with same format and no
    'none' / 'cannot be determined' distractors.
    """
    return validate_task(
        question=question,
        answer=answer,
        answer_format=answer_format,
        image_path=image_path,
        caption=caption,
        options=options,
    )


def _is_binary_question(q: str) -> bool:
    """Check if question is binary (yes/no, true/false)."""
    q_lower = q.lower().strip()
    for pattern in BINARY_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def _restricts_options(q: str) -> bool:
    """Check if question restricts options (e.g. 'Out of the 3...')."""
    for pattern in RESTRICT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


def _has_domain_jargon(q: str) -> bool:
    """Check for domain-specific terminology."""
    for pattern in DOMAIN_JARGON:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


def _references_visual_content(q: str) -> bool:
    """Heuristic: check if question likely references visual elements."""
    visual_refs = [
        "chart", "graph", "figure", "image", "diagram", "plot",
        "table", "panel", "bar", "line", "pie", "color", "shape",
        "object", "left", "right", "top", "bottom", "above", "below",
        "next to", "between", "behind", "front", "show", "display",
        "axis", "label", "legend", "title", "caption", "y-axis", "x-axis",
        "grid", "row", "column", "cell", "tile", "circle", "square",
        "highlighted", "marked", "circled", "indicated", "pointed",
    ]
    q_lower = q.lower()
    return any(ref in q_lower for ref in visual_refs)


def _is_number(s: str) -> bool:
    """Check if string looks like a number."""
    s = s.strip().replace(",", "").replace("%", "").replace("^", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_explanation_question(q: str) -> bool:
    """Check if question asks for explanation.

    Handbook 'Common Errors': "Asking questions that are 'how', or asking
    'what trend do you see' are all explanation-type questions that will
    result in phrases/long sentences."
    """
    q_lower = q.lower()
    for pattern in EXPLANATION_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _has_reasoning_depth(q: str) -> bool:
    """Check if question requires multi-step reasoning."""
    q_lower = q.lower()
    for pattern in REASONING_INDICATORS:
        if re.search(pattern, q_lower):
            return True
    return False


def _answer_seems_derivable(q: str, a: str) -> bool:
    """Check if answer seems derivable from question context."""
    guessable = {"yes", "no", "true", "false", "none", "all", "both", "neither"}
    if a in guessable:
        return False
    return True


def _check_grammar(q: str) -> list[str]:
    """Basic grammar checks."""
    issues = []
    if "  " in q:
        issues.append("Double space detected")
    if q and q[0].islower():
        issues.append("Question should start with capital letter")
    return issues


def _has_extreme_seeking(q: str) -> bool:
    """Check for extreme-seeking words (Qwen bias)."""
    q_lower = q.lower()
    for pattern in EXTREME_SEEKING:
        if re.search(pattern, q_lower):
            return True
    return False


def _matches_chart_anti_pattern(q: str) -> list[str]:
    """Detect chart-furniture counting questions (labels/ticks/colorbars).

    Returns the list of anti-pattern descriptions matched. Empty list = good.
    """
    q_lower = q.lower()
    hits = []
    for pattern, desc in CHART_FURNITURE_ANTI_PATTERNS:
        if re.search(pattern, q_lower):
            hits.append(desc)
    return hits


def _references_chart_data(q: str) -> bool:
    """Detect whether question references actual chart data (values, peaks, regions)
    rather than chart furniture (labels, ticks, colorbars)."""
    q_lower = q.lower()
    return any(re.search(p, q_lower) for p in CHART_DATA_REFS)


def _is_generic_count_question(q: str) -> bool:
    """Detect 'How many X are in the image?' questions without filter/comparison.

    These are uniformly too easy for Qwen. Always flagged as error.
    """
    q_stripped = q.strip().lower()
    for pattern in GENERIC_COUNT_PATTERNS:
        if re.match(pattern, q_stripped):
            return True
    return False


def _has_threshold_filter(q: str) -> bool:
    """Check for threshold filters (good for creating difficulty)."""
    q_lower = q.lower()
    for pattern in THRESHOLD_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _references_multi_panel(q: str) -> bool:
    """Check if question references multiple panels/figures."""
    q_lower = q.lower()
    for pattern in MULTI_PANEL_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _is_caption_solvable(caption: str, q: str = "") -> bool:
    """Check if caption is too descriptive or question asks about caption.

    Args:
        caption: Figure caption text.
        q: Question text (to detect 'caption' references).
    """
    if not caption:
        return False
    if len(caption) > 150:
        return True
    if re.search(r"\d+[.%]", caption):
        return True
    if q and re.search(r"\bcaption\b", q.lower()):
        return True
    return False


def _answer_is_extreme(a: str) -> bool:
    """Check if answer is an extreme value."""
    extreme_words = {"highest", "lowest", "largest", "smallest", "maximum", "minimum",
                     "most", "least", "best", "worst", "first", "last", "top", "bottom"}
    return a.lower() in extreme_words


def _calculate_score(result: ValidationResult) -> float:
    """Calculate quality score (0-100) based on validation results."""
    score = 100.0

    # Penalties for errors (major issues)
    score -= len(result.errors) * 20

    # Penalties for warnings (minor issues)
    score -= len(result.warnings) * 5

    # Bonuses for good patterns
    good_patterns = [
        "multi-step reasoning",
        "threshold filters",
        "multiple panels",
        "not extreme-seeking",
    ]
    for check in result.passed_checks:
        for pattern in good_patterns:
            if pattern.lower() in check.lower():
                score += 5

    return max(0, min(100, score))
