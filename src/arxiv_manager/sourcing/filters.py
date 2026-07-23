"""Image quality, complexity, and type classification filters.

Provides:
    compute_complexity()   - improved with density score
    classify_figure_type() - chart_graph_text vs general_image
    is_likely_sparse()     - aspect/variance-based filter
    is_likely_logo_or_icon() - small/decorative image filter
    compute_file_hash()    - SHA256 for dedup
    compute_perceptual_hash() - near-duplicate detection
    audit_figure()         - all-in-one figure audit
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Complexity scoring ────────────────────────────────────────────


def compute_complexity(image_path: Path) -> float:
    """Compute a complexity score for an image (0.0 = trivial, 1.0 = very complex).

    Uses color variance, edge density, detail, and element density.
    Element density (connected-component count) is the strongest predictor
    of Challenging-task suitability — images with many discrete elements
    force Qwen to actually count rather than guess.
    """
    try:
        img = Image.open(image_path)
    except Exception:
        return 0.0

    if img.mode != "RGB":
        img = img.convert("RGB")

    # 1. Color variance: more colors = more complex
    small = img.resize((50, 50))
    pixels = list(small.getdata())
    unique_colors = len(set(pixels))
    color_score = min(unique_colors / 1000, 1.0)

    # 2. Edge density via grayscale variance
    gray = img.convert("L").resize((100, 100))
    pixels_gray = list(gray.getdata())
    mean = sum(pixels_gray) / len(pixels_gray)
    variance = sum((p - mean) ** 2 for p in pixels_gray) / len(pixels_gray)
    edge_score = min(variance / 5000, 1.0)

    # 3. Detail via high-frequency content
    detail_score = _compute_detail_score(img)

    # 4. NEW: Element density (count of distinct connected components)
    density_score = _compute_density_score(img)

    # Weighted combination (density is most predictive of Challenging success)
    score = (
        0.20 * color_score
        + 0.30 * edge_score
        + 0.20 * detail_score
        + 0.30 * density_score
    )
    return round(min(max(score, 0.0), 1.0), 3)


def _compute_detail_score(img: Image.Image) -> float:
    """Measure high-frequency detail."""
    gray = img.convert("L").resize((100, 100))
    pixels = list(gray.getdata())
    changes = 0
    for i in range(1, len(pixels)):
        if abs(pixels[i] - pixels[i - 1]) > 15:
            changes += 1
    return min(changes / (len(pixels) * 0.3), 1.0)


def _compute_density_score(img: Image.Image) -> float:
    """Measure element density via connected components on a small binary image.

    More distinct elements (boxes, cells, points) → higher score.
    """
    try:
        gray = img.convert("L").resize((80, 80))
        pixels = list(gray.getdata())
        threshold = 200
        binary = [1 if p < threshold else 0 for p in pixels]
        w = 80
        # Count connected components via flood fill
        visited = [False] * len(binary)
        components = 0
        for i in range(len(binary)):
            if binary[i] == 1 and not visited[i]:
                components += 1
                _flood_fill(binary, visited, i, w)
        # Normalize: 30+ components is dense
        return min(components / 30.0, 1.0)
    except Exception:
        return 0.0


def _flood_fill(binary: list[int], visited: list[bool], start: int, w: int) -> None:
    """Iterative flood fill of a 1-cell in a width-w grid."""
    stack = [start]
    h = len(binary) // w
    while stack:
        i = stack.pop()
        if i < 0 or i >= len(binary) or visited[i] or binary[i] == 0:
            continue
        visited[i] = True
        x, y = i % w, i // w
        if x + 1 < w:
            stack.append(i + 1)
        if x - 1 >= 0:
            stack.append(i - 1)
        if y + 1 < h:
            stack.append(i + w)
        if y - 1 >= 0:
            stack.append(i - w)


# ─── Figure type classification ─────────────────────────────────────


def classify_figure_type(image_path: Path) -> dict:
    """Classify a figure as 'chart_graph_text' or 'general_image'.

    Heuristic 2-way classifier based on:
    - Background whiteness (charts often have white/light bg)
    - Axis-like edges (long horizontal/vertical lines near edges)
    - Color variance and saturation
    - Presence of dense text regions

    Returns:
        {"figure_type": "chart_graph_text" | "general_image", "confidence": 0.0-1.0}
    """
    try:
        img = Image.open(image_path)
    except Exception:
        return {"figure_type": "general_image", "confidence": 0.0}

    if img.mode != "RGB":
        img = img.convert("RGB")

    gray = img.convert("L").resize((200, 200))
    g_pixels = list(gray.getdata())
    w, h = gray.size

    # 1. Background whiteness (% of pixels > 240)
    white_ratio = sum(1 for p in g_pixels if p > 240) / len(g_pixels)

    # 2. Color saturation (charts typically low-saturation, photos high)
    small_rgb = img.resize((100, 100)).convert("RGB")
    rgb_pixels = list(small_rgb.getdata())
    saturations = []
    for r, g, b in rgb_pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        if mx > 0:
            saturations.append((mx - mn) / mx)
    avg_sat = sum(saturations) / len(saturations) if saturations else 0

    # 3. Edge lines near borders (charts often have axes)
    # Count long horizontal runs in top/bottom 10% of image
    border_lines = 0
    for y in [10, 30, h - 30, h - 10]:
        if y < 0 or y >= h:
            continue
        row = g_pixels[y * w : (y + 1) * w]
        run = 0
        max_run = 0
        for p in row:
            if p < 100:  # dark pixel
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > 40:
            border_lines += 1

    # 4. Text density (very small dark blobs scattered = text/labels)
    dark_blobs = 0
    bin80 = [1 if p < 100 else 0 for p in gray.resize((80, 80)).getdata()]
    visited = [False] * len(bin80)
    for i in range(len(bin80)):
        if bin80[i] == 1 and not visited[i]:
            dark_blobs += 1
            # Bounded flood fill (limit depth to keep it fast)
            stack = [i]
            count = 0
            while stack and count < 200:
                j = stack.pop()
                if j < 0 or j >= len(bin80) or visited[j] or bin80[j] == 0:
                    continue
                visited[j] = True
                count += 1
                x = j % 80
                y = j // 80
                if x + 1 < 80:
                    stack.append(j + 1)
                if x - 1 >= 0:
                    stack.append(j - 1)
                if y + 1 < 80:
                    stack.append(j + 80)
                if y - 1 >= 0:
                    stack.append(j - 80)

    # Combine signals
    chart_score = 0.0
    chart_score += 0.35 if white_ratio > 0.5 else 0.0
    chart_score += 0.25 if avg_sat < 0.25 else 0.0
    chart_score += 0.25 if border_lines >= 1 else 0.0
    chart_score += 0.15 if dark_blobs > 15 else 0.0

    is_chart = chart_score >= 0.4
    confidence = chart_score if is_chart else (1.0 - chart_score)

    return {
        "figure_type": "chart_graph_text" if is_chart else "general_image",
        "confidence": round(confidence, 3),
    }


# ─── Sparseness filter ─────────────────────────────────────────────


def is_likely_sparse(image_path: Path) -> bool:
    """Detect sparse/decoration figures (banners, dividers, mostly empty).

    Returns True if:
    - Aspect ratio > 5:1 or < 1:5 (banner-like)
    - Color variance very low (< 30)
    - Complexity already known to be < 0.2
    """
    try:
        img = Image.open(image_path)
    except Exception:
        return True

    w, h = img.size
    if w == 0 or h == 0:
        return True
    aspect = max(w / h, h / w)
    if aspect > 5.0:
        return True

    if img.mode != "RGB":
        img = img.convert("RGB")
    gray = img.convert("L").resize((100, 100))
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    if variance < 30:
        return True

    return False


# ─── Logo/icon filter ──────────────────────────────────────────────


def is_likely_logo_or_icon(image_path: Path) -> bool:
    """Heuristic: small, square images with few colors are likely logos."""
    try:
        img = Image.open(image_path)
    except Exception:
        return True

    w, h = img.size
    if w == h and w < 200:
        return True

    small = img.resize((20, 20))
    colors = len(set(small.getdata()))
    if colors < 5:
        return True

    return False


# ─── Text-only detection ──────────────────────────────────────────


def is_text_only(image_path: Path) -> bool:
    """Check if image is likely a text-only region (not a real figure).

    Text-heavy captures (from mis-detected captions) have:
    - Many horizontal edges (text lines)
    - Low color variance (black text on white)
    - Uniform row structure

    Uses same heuristic as extractor._is_likely_figure() but on PIL.
    """
    try:
        img = Image.open(image_path)
    except Exception:
        return False
    if img.mode != "RGB":
        img = img.convert("RGB")
    gray = img.convert("L").resize((100, 100))
    w = h = 100
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)

    # Color variance: text has very low (< 500)
    if variance >= 2000:
        return False  # high variance = figure

    # Horizontal edge ratio: count light→dark transitions per row
    # Threshold: use 200 instead of 100 because PDF-rendered text often
    # appears as gray (~160-200) rather than pure black (0).
    total_transitions = 0
    for y in range(h):
        row = pixels[y * w : (y + 1) * w]
        transitions = sum(1 for i in range(1, w) if (row[i] < 200) != (row[i - 1] < 200))
        total_transitions += transitions
    avg_transitions = total_transitions / h
    # Text has many transitions per row (7+); figures have few (0-3)
    if avg_transitions > 5:
        return True
    # Low variance + low transitions likely blank
    if variance < 200:
        return True
    return False


# ─── Hashing utilities ─────────────────────────────────────────────


def compute_file_hash(image_path: Path) -> str:
    """Compute SHA256 hash of an image file for deduplication."""
    data = image_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def compute_perceptual_hash(image_path: Path) -> str:
    """Compute perceptual hash for near-duplicate detection."""
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception:
        return ""


# ─── All-in-one audit ──────────────────────────────────────────────


def audit_figure(image_path: Path) -> dict:
    """Run all filters and return a single audit record."""
    logger.info("audit_figure entry path=%s", image_path)
    result = {
        "filesize_bytes": 0,
        "width": 0,
        "height": 0,
        "width_height_ratio": 0.0,
        "complexity_score": 0.0,
        "figure_type": "",
        "figure_type_confidence": 0.0,
        "is_dense": False,
        "is_logo_or_icon": False,
        "is_likely_sparse": False,
        "is_text_only": False,
        "is_suitable": False,
    }
    try:
        result["filesize_bytes"] = image_path.stat().st_size
    except Exception:
        return result

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            result["width"] = w
            result["height"] = h
            if h > 0:
                result["width_height_ratio"] = round(w / h, 3)
    except Exception:
        return result

    result["is_logo_or_icon"] = is_likely_logo_or_icon(image_path)
    result["is_likely_sparse"] = is_likely_sparse(image_path)
    result["is_text_only"] = is_text_only(image_path)
    result["complexity_score"] = compute_complexity(image_path)
    cls = classify_figure_type(image_path)
    result["figure_type"] = cls["figure_type"]
    result["figure_type_confidence"] = cls["confidence"]

    # Density heuristic: high complexity + non-sparse
    result["is_dense"] = (
        result["complexity_score"] >= 0.5 and not result["is_likely_sparse"]
    )

    # Suitable = passes all gates
    result["is_suitable"] = (
        not result["is_logo_or_icon"]
        and not result["is_likely_sparse"]
        and not result["is_text_only"]
        and result["complexity_score"] >= 0.3
        and result["width"] >= 200
        and result["height"] >= 200
        and result["filesize_bytes"] >= 5000
    )

    logger.info("audit_figure result cs=%.3f type=%s suitable=%s dense=%s text=%s sparse=%s logo=%s dims=%dx%d",
                 result["complexity_score"], result["figure_type"], result["is_suitable"],
                 result["is_dense"], result["is_text_only"], result["is_likely_sparse"],
                 result["is_logo_or_icon"], result["width"], result["height"])
    return result
