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
    r"\bhybridization\b",
    r"\bsigma bond\b",
    r"\bpi bond\b",
    r"\bLUMO\b",
    r"\bHOMO\b",
    r"\bsp[23]\b",
    r"\bEBITDA\b",
    r"\bWACC\b",
    r"\bDCF\b",
    r"\bp-value\b",
    r"\bchi.?square\b",
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
    (
        r"\b(?:labeled|numerical)\s+(?:tick|values?|labels?|numbers?)\b.*\b(?:axis|colorbar|legend|tick)\b",
        "labeled value counting",
    ),
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
    r"shutterstock",
    r"getty",
    r"©",
    r"all rights reserved",
    r"istock",
    r"alamy",
    r"watermark",
    r"stock photo",
]

visual_refs = [
    "chart",
    "graph",
    "figure",
    "image",
    "diagram",
    "plot",
    "table",
    "panel",
    "bar",
    "line",
    "pie",
    "color",
    "shape",
    "object",
    "left",
    "right",
    "top",
    "bottom",
    "above",
    "below",
    "next to",
    "between",
    "behind",
    "front",
    "show",
    "display",
    "axis",
    "label",
    "legend",
    "title",
    "y-axis",
    "x-axis",
    "grid",
    "row",
    "column",
    "cell",
    "tile",
    "circle",
    "square",
    "highlighted",
    "marked",
    "circled",
    "indicated",
    "pointed",
]

# ─── Helper functions ──────────────────────────────────────────────


def _is_binary_question(q: str) -> bool:
    q_lower = q.lower().strip()
    for pattern in BINARY_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
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


def _is_int(s: str) -> bool:
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        return val == int(val)
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


MANUFACTURED_DIFFICULTY_PATTERNS = [
    r"^count\s+.*?\s+and\s+.*?\s+then\s+(?:count|add|sum|give)",
    r"^how\s+many\s+.*?\s+and\s+how\s+many\s+",
    r"^count\s+.*?\s+then\s+(?:add|sum|multiply|give)",
    r"rank(?:ed)?\s+by\s+(?:the\s+)?number(?:\s+of)?",
]


def _is_manufactured_difficulty(q: str, difficulty: str) -> bool:
    """Detect counting-only questions that use quantity as main difficulty.

    Fires for challenging/hardest when the question relies on bare counting
    without meaningful visual reasoning (comparison, spatial, classification).
    Counting is acceptable as a *final step* after reasoning, not as primary
    difficulty source.
    """
    if difficulty not in ("challenging", "hardest"):
        return False
    q_stripped = q.strip().lower()
    return any(re.search(p, q_stripped) for p in MANUFACTURED_DIFFICULTY_PATTERNS)


BINARY_ANSWER_PATTERNS = [
    (r"\bhigher\s+or\s+lower\b", "higher/lower"),
    (r"\blower\s+or\s+higher\b", "lower/higher"),
    (r"\bgreater\s+or\s+(?:less|lesser)\b", "greater/less"),
    (r"\b(?:lesser|less)\s+or\s+greater\b", "less/greater"),
    (r"\bincreas(?:e|ing|es)\s+or\s+decreas(?:e|ing|es)\b", "increase/decrease"),
    (r"\bdecreas(?:e|ing|es)\s+or\s+increas(?:e|ing|es)\b", "decrease/increase"),
    (r"\bupward\s+or\s+downward\b", "upward/downward"),
    (r"\bdownward\s+or\s+upward\b", "downward/upward"),
    (r"\b(?:does|is)\s+.*?\s+(up|down)\s+or\s+(up|down)\b", "up/down"),
]

BINARY_ANSWER_VALUES = {
    "higher", "lower", "greater", "less", "increases", "increasing",
    "increase", "decreases", "decreasing", "decrease", "upward",
    "downward", "up", "down", "before", "after", "left", "right",
    "yes", "no", "true", "false",
}

INLINE_CHOICE_PATTERNS = [
    r"\b(?:is|are|does|do|has|have|was|were|did|can|could|will|would|should)\s+.+?\s+or\s+.+?\?",
    r"\b(?:choose|select|pick)\s+(?:between|from)\s+.+?\s+or\s+",
    r"\b(?:is\s+it|are\s+they)\s+.+?\s+or\s+.+?\?",
    r"\b(?:which|what)\s+.+?:\s*.+?\s+or\s+.+?",
]


def _is_binary_answer(q: str, a: str) -> bool:
    a_lower = a.strip().lower().rstrip(".")
    if a_lower not in BINARY_ANSWER_VALUES:
        return False
    q_lower = q.lower()
    for pattern, _pair in BINARY_ANSWER_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _has_inline_choices(q: str) -> bool:
    """Detect questions that provide predefined choices (binary guess)."""
    q_lower = q.lower()
    for pattern in INLINE_CHOICE_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


SINGLE_MATCHMAKING_PATTERNS = [
    r"which\s+(?:single\s+)?panel\s+(?:meets|has|shows|satisfies|contains)",
    r"which\s+(?:component|node|gate|element)\s+(?:has|meets|satisfies)",
    r"identify\s+the\s+(?:panel|component|node|gate|element)\s+that",
    r"which\s+(?:one|single)\s+(?:has|meets|shows|satisfies|contains)",
]


def _is_single_matchmaking(q: str, a: str) -> bool:
    q_lower = q.lower()
    a_lower = a.strip().lower().rstrip(".")
    if not any(re.search(p, q_lower) for p in SINGLE_MATCHMAKING_PATTERNS):
        return False
    if len(a_lower.split()) > 3:
        return False
    # Exclude legitimate comparisons: "which panel has the higher/larger/greater value"
    # is strategy #2 (which-is-higher), not matchmaking
    if re.search(r"\b(higher|lower|larger|greater|steeper|flatter|faster|slower|more|most|fewer|less)\b", q_lower):
        return False
    return True


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
    return False


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
    extreme_words = {
        "highest",
        "lowest",
        "largest",
        "smallest",
        "maximum",
        "minimum",
        "most",
        "least",
        "best",
        "worst",
        "first",
        "last",
        "top",
        "bottom",
    }
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


ARITHMETIC_KEYWORDS = re.compile(
    r"\b(?:product|sum|difference|quotient|multipl(?:y|ied)|add(?:ed)?|subtract(?:ed)?|divide?)\b",
    re.IGNORECASE,
)


def _requires_arithmetic(q: str) -> bool:
    """Check if the question asks for an arithmetic operation on multiple values."""
    return bool(ARITHMETIC_KEYWORDS.search(q))


CALCULATION_KEYWORDS = [
    "change", "increase", "decrease", "difference", "ratio", "percentage",
    "rank", "sum", "total", "average", "how many more", "how many less",
    "percentage point", "drop", "rise", "exceed", "cross",
]


def _requires_calculation(q: str) -> bool:
    """Check if question contains arithmetic/calculation language."""
    q_lower = q.lower()
    return any(w in q_lower for w in CALCULATION_KEYWORDS)


def _is_two_answer_question(q: str) -> bool:
    """Detect questions that ask for two separate answers.

    Examples of two-answer questions:
    - "Which index has the largest decline and what is the percentage?"
    - "What is the highest value and which country has it?"
    - "How many X are there and how many Y?"
    """
    q_lower = q.lower().rstrip(".")
    # Normalise conjunctive "when Xed" so it doesn't match as a question word
    q_clean = re.sub(r"\bwhen\s+(?:ranked|sorted|compared|assigned|given|viewed|ordered|measured|numbered|evaluated|listed|arranged|placed|grouped|labeled|set)\b", " while ", q_lower)
    # Pattern: two question words (which/what/how) connected by " and "
    question_words = r"(?:which|what|how|where|when)"
    pattern = rf"{question_words}\b.*?\band\b.*?{question_words}\b"
    if re.search(pattern, q_clean):
        return True
    # Pattern: "X and what Y" or "which X and how Y"
    pattern2 = r"\b(?:which|what|how|where|when)\b.*?\band\b.*?\b(?:which|what|how|where|when)\b"
    if re.search(pattern2, q_clean):
        return True
    return False


# _calculate_score is kept in validator.py to avoid circular import with ValidationResult
