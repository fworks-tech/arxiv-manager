"""Validation helper patterns and functions."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# --- Binary / T-F detection ---

BINARY_PATTERNS = [
    r"^(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\s+.+\?$",
    r"^(true|false)\s",
    r"^(yes|no)\s",
    r"\b(true or false)\b",
    r"\b(yes or no)\b",
]

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

MATH_HEAVY_PATTERNS = [
    r"\b(?:multiply|divide|subtract|add)\b",
    r"\b(?:calculate|compute|solve)\b",
    r"\b(?:sum|product|quotient|difference)\s+of\s+",
    r"\b\d+\s*[\*\/×÷]\s*\d+\b",
]

TEXT_HEAVY_PATTERNS = [
    r"^what does (?:this|the|it) say",
    r"^what is written",
    r"^what (?:does|is) the (?:text|caption|quote|title|heading|label) say",
]

LONG_WINDED_INDICATORS = [
    r"\bthe (?:one|item) (?:that|who) (?:is|has|contains) .*? and (?:is|has|contains)",
    r"^(?:the|a|an)\s+\w+\s+(?:that|who|which)\s+\w+\s+\w+",
]

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
    r"\b(larger|smaller|greater|less|fewer)\b.{0,40}\bthan\b",
    r"\b(magnitude)\b",
]

NOISE_CONDITION_PATTERNS = [
    r"\bbetween\s+[\d.]+\s*(?:nm|mm|cm|m|kg|lb|s|sec|min|hr|h|°[CFK])\s+and\s+[\d.]+",
    r"\bover\s+all\s+wavelengths\b",
    r"\bacross\s+all\b",
]

EXTREME_SEEKING = [
    r"\b(highest|lowest|largest|smallest|most|least|fewest|greatest)\b",
    r"\b(maximum|minimum|max|min)\b",
    r"\b(leftmost|rightmost|topmost|bottommost)\b",
    r"\b(best|worst|strongest|weakest)\b",
]

THRESHOLD_PATTERNS = [
    r"\b(fewer|less|more|greater|above|below|over|under|exactly|closest to|nearest to)\s+(?:than\s+)?\d",
    r"\b(between|within|around)\s+\d",
    r"\b(?:less|more)\s+than\s+\d",
]

MULTI_PANEL_PATTERNS = [
    r"\bpanel\s*\([a-z]\)",
    r"\bfigure\s*\d",
    r"\b(a)\b.*\b(b)\b",
    r"\bleft\b.*\bright\b",
    r"\btop\b.*\bbottom\b",
]

CHART_FURNITURE_ANTI_PATTERNS = [
    (r"\b(?:tick|axis|colorbar|legend)\s+(?:labels?|marks?|ticks?|numbers?)\b", "axis/label/tick counting"),
    (r"\b(?:count|how\s+many)\s+(?:the\s+)?(?:tick|axis|colorbar|legend)\b", "axis counting"),
    (r"\b(?:labeled|numerical)\s+(?:tick|values?|labels?|numbers?)\b.*\b(?:axis|colorbar|legend|tick)\b", "labeled value counting"),
]

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

GENERIC_COUNT_PATTERNS = [
    r"^how\s+many\s+[\w\s]+\s+(?:are|appear|exist|visible)\s+(?:in\s+)?(?:the\s+)?(?:image|figure|chart|diagram|plot)\s*[\?\.]?$",
    r"^count\s+(?:the\s+)?(?:total\s+)?(?:number\s+of\s+)?[\w\s]+(?:in|across)\s+(?:the\s+)?(?:image|figure|chart|diagram)\s*[\?\.]?$",
]

ANSWER_IN_QUESTION_PATTERNS = [
    r"\b\w+\s+(?:covers|ranges|spans|goes)\s+(?:\w+\s+)?(?:from\s+)?-?\d+(?:\.\d+)?\s*(?:to|-)\s*-?\d+(?:\.\d+)?\b.*\b\w+\s+(?:covers|ranges|spans|goes)\s+(?:\w+\s+)?(?:from\s+)?-?\d+(?:\.\d+)?\s*(?:to|-)\s*-?\d+(?:\.\d+)?",
    r"\baxis\s+\w+\s+(?:from\s+)?-?\d+(?:\.\d+)?\s*(?:to|-)\s*-?\d+(?:\.\d+)?\s*(?:and|,).{0,40}(?:from\s+)?-?\d+(?:\.\d+)?\s*(?:to|-)\s*-?\d+(?:\.\d+)?",
    r"\b(?:values?|scores?|numbers?)\s+(?:are|of|=)\s+-?\d+(?:\.\d+)?\s*(?:,|and|;)\s*-?\d+(?:\.\d+)?\s*(?:,|and|;)\s*-?\d+(?:\.\d+)?",
    r"\b\w+\s+is\s+-?\d+(?:\.\d+)?\s*(?:,|;|and)\s*\w+\s+is\s+-?\d+(?:\.\d+)?\s*(?:,|;|and)\s*\w+\s+is\s+-?\d+(?:\.\d+)?",
]

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

WATERMARK_HINTS = [
    r"shutterstock", r"getty", r"©", r"all rights reserved",
    r"istock", r"alamy", r"watermark", r"stock photo",
]

visual_refs = [
    "chart", "graph", "figure", "image", "diagram", "plot",
    "table", "panel", "bar", "line", "pie", "color", "shape",
    "object", "left", "right", "top", "bottom", "above", "below",
    "next to", "between", "behind", "front", "show", "display",
    "axis", "label", "legend", "title", "y-axis", "x-axis",
    "grid", "row", "column", "cell", "tile", "circle", "square",
    "highlighted", "marked", "circled", "indicated", "pointed",
]

# ─── Helper functions ──────────────────────────────────────────────


def _is_binary_question(q: str) -> bool:
    q_lower = q.lower().strip()
    for pattern in BINARY_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def _restricts_options(q: str) -> bool:
    for pattern in RESTRICT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


def _has_domain_jargon(q: str) -> bool:
    for pattern in DOMAIN_JARGON:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


def _references_visual_content(q: str) -> bool:
    q_lower = q.lower()
    return any(ref in q_lower for ref in visual_refs)


def _is_number(s: str) -> bool:
    s = s.strip().replace(",", "").replace("%", "").replace("^", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_explanation_question(q: str) -> bool:
    q_lower = q.lower()
    for pattern in EXPLANATION_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _has_reasoning_depth(q: str) -> bool:
    q_lower = q.lower()
    for pattern in REASONING_INDICATORS:
        if re.search(pattern, q_lower):
            return True
    return False


def _answer_seems_derivable(q: str, a: str) -> bool:
    guessable = {"yes", "no", "true", "false", "none", "all", "both", "neither"}
    if a in guessable:
        return False
    return True


def _check_grammar(q: str) -> list[str]:
    issues = []
    if "  " in q:
        issues.append("Double space detected")
    if q and q[0].islower():
        issues.append("Question should start with capital letter")
    return issues


def _has_extreme_seeking(q: str) -> bool:
    q_lower = q.lower()
    for pattern in EXTREME_SEEKING:
        if re.search(pattern, q_lower):
            return True
    return False


def _matches_chart_anti_pattern(q: str) -> list[str]:
    q_lower = q.lower()
    hits = []
    for pattern, desc in CHART_FURNITURE_ANTI_PATTERNS:
        if re.search(pattern, q_lower):
            hits.append(desc)
    return hits


def _references_chart_data(q: str) -> bool:
    q_lower = q.lower()
    return any(re.search(p, q_lower) for p in CHART_DATA_REFS)


def _is_generic_count_question(q: str) -> bool:
    q_stripped = q.strip().lower()
    for pattern in GENERIC_COUNT_PATTERNS:
        if re.match(pattern, q_stripped):
            return True
    return False


def _has_answer_in_question(q: str, a: str) -> bool:
    q_lower = q.lower()
    if not any(re.search(p, q_lower) for p in ANSWER_IN_QUESTION_PATTERNS):
        return False
    numbers = re.findall(r"-?\d+(?:\.\d+)?", q)
    if len(numbers) >= 2:
        try:
            nums = [float(n) for n in numbers]
            ans = float(a.strip())
            for n1 in nums:
                for n2 in nums:
                    if n2 != 0 and abs(n1 / n2 - ans) < 0.01 * abs(ans) + 0.01:
                        return True
                    if abs(n1 - n2 - ans) < 0.01 * abs(ans) + 0.01:
                        return True
                    if abs(n1 + n2 - ans) < 0.01 * abs(ans) + 0.01:
                        return True
        except (ValueError, ZeroDivisionError):
            pass
    return True


def _is_chart_math_only(q: str) -> bool:
    q_lower = q.lower()
    math_op = (
        re.search(r"\b(ratio|factor)\b", q_lower)
        or re.search(r"\b(what\s+is\s+the\s+sum|sum\s+of)\b", q_lower)
        or re.search(r"\b(what\s+is\s+the\s+difference|difference\s+between)\b", q_lower)
        or re.search(r"\b(what\s+is\s+the\s+product|product\s+of)\b", q_lower)
    )
    if not math_op:
        return False
    has_inline_data = (
        re.search(r"\b\w+\s+(?:covers|ranges|spans|goes)\s+(?:\w+\s+)?(?:from\s+)?-?\d", q_lower)
        or re.search(r"\baxis\s+\w+\s+(?:from\s+)?-?\d", q_lower)
        or re.search(r"\b(?:values?|scores?)\s+(?:are|of)\s+-?\d", q_lower)
    )
    return bool(has_inline_data)


def _has_threshold_filter(q: str) -> bool:
    q_lower = q.lower()
    for pattern in THRESHOLD_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _references_multi_panel(q: str) -> bool:
    q_lower = q.lower()
    for pattern in MULTI_PANEL_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _answer_is_extreme(a: str) -> bool:
    extreme_words = {"highest", "lowest", "largest", "smallest", "maximum", "minimum",
                     "most", "least", "best", "worst", "first", "last", "top", "bottom"}
    return a.lower() in extreme_words


def _answer_is_list_of_three_plus(a: str) -> bool:
    parts = re.split(r",|\band\b", a)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts) > 3


def _passes_visual_dependence_test(q: str) -> bool:
    q_lower = q.lower()
    if TEXT_HEAVY_PATTERNS and any(re.search(p, q_lower) for p in TEXT_HEAVY_PATTERNS):
        return False
    return any(ref in q_lower for ref in visual_refs)


def _passes_one_answer_test(q: str, a: str) -> bool:
    a_lower = a.strip().lower()
    if a_lower in {"varies", "maybe", "it depends", "unclear"}:
        return False
    if len(a.split()) > 4:
        return False
    return True


def _is_caption_solvable(caption: str, q: str = "") -> bool:
    if not caption:
        return False
    if len(caption) > 150:
        return True
    if re.search(r"\d+[.%]", caption):
        return True
    if q and re.search(r"\bcaption\b", q.lower()):
        return True
    return False


def _check_mcq_options(options: list[str] | None) -> list[str]:
    issues = []
    if not options:
        return issues
    if len(options) < 8:
        issues.append(f"MCQ has {len(options)} options; handbook requires 8+")
    for opt in options:
        if opt.strip().lower() in TRICK_ANSWERS:
            issues.append(f"MCQ option '{opt}' is a trick answer (handbook ban)")
    return issues


# _calculate_score is kept in validator.py to avoid circular import with ValidationResult
