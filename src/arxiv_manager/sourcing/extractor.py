"""Figure extractor from PDFs using PyMuPDF."""

from __future__ import annotations

import io
import re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from ..storage import FIGURES_DIR


def extract_figures(pdf_path: Path, min_size: int = 300) -> list[dict]:
    """Extract images from a PDF.

    Handles both embedded raster images (with orientation correction)
    and vector graphics (by rendering caption regions).

    Args:
        pdf_path: Path to the PDF file.
        min_size: Minimum width/height in pixels to keep an image.

    Returns:
        List of dicts with keys: image_path, page_num, figure_num, width, height.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    results: list[dict] = []
    paper_id = pdf_path.stem
    seen_hashes = set()

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 1. Extract embedded raster images with orientation correction
        images = page.get_images(full=True)
        for img_idx, img_info in enumerate(images):
            xref = img_info[0]

            # Get image placement rect and transform matrix
            try:
                rect_transforms = page.get_image_rects(xref, transform=True)
                if not rect_transforms:
                    continue
                img_rect, transform = rect_transforms[0]
            except Exception:
                continue

            # Skip tiny images (likely icons/logos)
            if img_rect.width < 80 or img_rect.height < 80:
                continue

            # Extract raw image and apply PDF transform flips
            try:
                pix = fitz.Pixmap(doc, xref)
                pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

                # Apply orientation correction based on transform matrix signs
                # Matrix: (a, b, c, d, e, f) where x' = ax + cy + e, y' = bx + dy + f
                if transform.d < 0:
                    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                if transform.a < 0:
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

                # Resize very large images to reasonable dimensions while keeping quality
                max_dimension = 1500
                if pil_img.width > max_dimension or pil_img.height > max_dimension:
                    pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

                # Convert to RGB if needed
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")

            except Exception:
                continue

            # Skip if still too small after resize
            if pil_img.width < min_size or pil_img.height < min_size:
                continue

            img_hash = f"{paper_id}_p{page_num}_i{img_idx}"
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            img_path = FIGURES_DIR / f"{img_hash}.png"
            pil_img.save(str(img_path), "PNG")
            caption = _extract_caption_for_image(page, img_rect)

            results.append({
                "image_path": f"figures/{img_hash}.png",
                "page_num": page_num + 1,
                "figure_num": _guess_figure_num(caption),
                "width": pil_img.width,
                "height": pil_img.height,
                "caption": caption,
            })

        # 2. Extract vector figures by finding "Figure X" captions
        #    and looking ABOVE them for actual image content.
        #    Strategy: scan for an embedded image near the caption first;
        #    if found, use its rect. Otherwise render the region between
        #    the previous element and the caption.
        figure_captions = _find_figure_captions(page)
        # Sort captions top-to-bottom so we can track previous element y
        figure_captions.sort(key=lambda x: x[2].y0)

        for idx, (fig_num, caption_text, caption_rect) in enumerate(figure_captions):
            img_hash = f"{paper_id}_p{page_num}_fig{fig_num}"
            if img_hash in seen_hashes:
                continue

            # Determine figure region: top_of_figure = end of previous element or page top
            if idx > 0:
                _, _, prev_caption = figure_captions[idx - 1]
                top_y = prev_caption.y1 + 5
            else:
                top_y = page.rect.y0 + 30

            # Strategy A: Look for an embedded image ABOVE the caption
            fig_rect = None
            for img_idx, img_info in enumerate(images):
                try:
                    rects = page.get_image_rects(img_info[0])
                    if not rects:
                        continue
                    img_rect = rects[0]
                    # Image should be above the caption (within 0-100px gap)
                    # and at least partially overlapping horizontally
                    gap = caption_rect.y0 - img_rect.y1
                    if 0 <= gap < 150 and _rects_overlap_horizontally(img_rect, caption_rect):
                        fig_rect = img_rect
                        break
                except Exception:
                    continue

            # Strategy B: No embedded image found → render from top_y to caption
            if fig_rect is None:
                # Use full page width, capture from top_y to caption top
                fig_rect = fitz.Rect(
                    page.rect.x0 + 20,
                    top_y,
                    page.rect.x1 - 20,
                    caption_rect.y0 - 5,
                )

            # Skip if too small
            if fig_rect.width < 200 or fig_rect.height < 200:
                continue

            # Render and validate
            try:
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=fig_rect)
            except Exception:
                continue

            if _has_content(pix) and _is_likely_figure(pix):
                seen_hashes.add(img_hash)
                img_path = FIGURES_DIR / f"{img_hash}.png"
                pix.save(str(img_path))

                results.append({
                    "image_path": f"figures/{img_hash}.png",
                    "page_num": page_num + 1,
                    "figure_num": fig_num,
                    "width": pix.width,
                    "height": pix.height,
                    "caption": caption_text,
                })

    doc.close()
    return results


def _find_figure_captions(page: fitz.Page) -> list[tuple]:
    """Find figure captions on a page.

    Merges adjacent text spans on the same line before matching,
    because PDF text is often split across individual word spans
    (e.g., "Figure" and "2:" in separate spans).

    Matches "Figure X:" / "Figure X." / "FIG X:" / "Fig. X:" patterns.
    Filters inline references ("shown in Figure 1") that are mid-paragraph
    and not followed by a colon (caption style).
    """
    results = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            # Merge all spans on this line into one text string
            spans = line.get("spans", [])
            merged_text = "".join(s.get("text", "") for s in spans).strip()

            # Search for "Figure X" or "Fig. X" pattern in merged text
            m = re.search(
                r"(?:Fig(?:ure)?\.?\s*)(\d+(?:\.\d+)?[a-z]?)\s*[:.]?\s*",
                merged_text, re.IGNORECASE,
            )
            if not m:
                continue

            # Filter inline references:
            # 1. If preceded by lowercase letter or period → mid-sentence ref
            # 2. If "Figure" is far from start (> 30 chars) and not colon → inline
            # 3. Contains "see" "shows" "shown" before "Figure" in last 3 words → inline
            before = merged_text[:m.start()].strip()
            after_match = merged_text[m.end():].strip()
            is_colon = after_match.startswith(':')

            # Check for sentence-context before "Figure"
            if before:
                last_char = before[-1]
                # Lowercase letter or punctuation before "Figure" = mid-sentence
                if last_char.islower() or last_char in ('.', ',', ';'):
                    if not is_colon:
                        continue

            # Long prefix without colon = inline
            if m.start() > 30 and not is_colon:
                continue

            fig_num = m.group(1)
            # Use bbox of the first span containing the match as anchor
            fig_anchor = None
            for s in spans:
                s_text = s.get("text", "")
                if "fig" in s_text.lower():
                    bbox = s.get("bbox")
                    if bbox:
                        fig_anchor = fitz.Rect(bbox)
                        break
            if fig_anchor:
                results.append((fig_num, merged_text, fig_anchor))

    return results


def _extract_caption_for_image(page: fitz.Page, img_rect: fitz.Rect) -> str:
    """Extract caption text near an embedded image."""
    search_rect = fitz.Rect(
        img_rect.x0, img_rect.y1,
        img_rect.x1, min(img_rect.y1 + 80, page.rect.height)
    )
    text_blocks = page.get_text("dict", clip=search_rect)

    caption_parts = []
    for block in text_blocks.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    caption_parts.append(t)

    return " ".join(caption_parts).strip()


def _guess_figure_num(caption: str) -> str:
    """Try to extract figure number from caption."""
    m = re.search(r"(?:Fig(?:ure)?\.?\s*)(\d+(?:\.\d+)?[a-z]?)", caption, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _rects_overlap_horizontally(rect_a: fitz.Rect, rect_b: fitz.Rect, overlap_ratio: float = 0.1) -> bool:
    """Check if two rectangles overlap horizontally (share x-span).

    At least overlap_ratio of the narrower rect's width must overlap.
    """
    left = max(rect_a.x0, rect_b.x0)
    right = min(rect_a.x1, rect_b.x1)
    if right <= left:
        return False
    overlap = right - left
    min_width = min(rect_a.width, rect_b.width)
    if min_width <= 0:
        return False
    return overlap / min_width >= overlap_ratio


def _has_content(pix: fitz.Pixmap, threshold: float = 0.03) -> bool:
    """Check if pixmap has meaningful content (not just white/blank).

    Returns True if more than threshold% of sampled pixels are non-white.
    """
    samples = pix.samples
    n = pix.n  # bytes per pixel (RGB=3, RGBA=4)

    if len(samples) < n * 100:
        return False

    non_white = 0
    sampled = 0

    # Sample pixels across the image
    step = n * 50  # Sample every 50th pixel
    for i in range(0, min(len(samples), n * 5000), step):
        sampled += 1
        r, g, b = samples[i], samples[i+1], samples[i+2]
        if r < 240 or g < 240 or b < 240:
            non_white += 1

    if sampled == 0:
        return False

    return (non_white / sampled) >= threshold


def _is_likely_figure(pix: fitz.Pixmap, min_nonwhite_ratio: float = 0.04) -> bool:
    """Check if rendered region is a real figure vs plain text page.

    Text-heavy captures (from misidentified captions) have:
    - Uniform horizontal lines (text rows)
    - Low color variance (black on white)
    - Very few non-white pixels that are not text

    Figures (charts, diagrams, photos) have:
    - Higher color variance
    - Mixed edge directions
    - Clusters of non-white pixels with varying colors

    Returns True if the region likely contains a real figure.
    """
    if not _has_content(pix, threshold=min_nonwhite_ratio):
        return False

    samples = pix.samples
    n = pix.n
    w, h = pix.width, pix.height
    if w < 200 or h < 200 or len(samples) < n * 100:
        return False

    # Downsample to 80x80 and measure color variance
    step_x = max(1, w // 80)
    step_y = max(1, h // 80)

    r_vals, g_vals, b_vals = [], [], []
    horizontal_edges = 0
    vertical_edges = 0
    total_sampled = 0
    prev_white = True

    for y in range(0, h, step_y):
        row_start = True
        for x in range(0, w, step_x):
            i = (y * w + x) * n
            if i + 2 >= len(samples):
                break
            r, g, b = samples[i], samples[i+1], samples[i+2]
            r_vals.append(r)
            g_vals.append(g)
            b_vals.append(b)
            is_white = r > 240 and g > 240 and b > 240
            if not row_start and is_white != prev_white:
                horizontal_edges += 1  # transition = text line boundary
            row_start = False
            prev_white = is_white
            total_sampled += 1

    if total_sampled == 0:
        return False

    # Color variance: figures have more spread
    avg_r, avg_g, avg_b = sum(r_vals) / len(r_vals), sum(g_vals) / len(g_vals), sum(b_vals) / len(b_vals)
    var_r = sum((v - avg_r) ** 2 for v in r_vals) / len(r_vals)
    color_variance = var_r

    # Text has many horizontal edges (text rows). Figures have fewer.
    # If >30% of samples are horizontal transitions, it's likely text.
    text_edge_ratio = horizontal_edges / total_sampled
    if text_edge_ratio > 0.3 and color_variance < 1500:
        return False  # looks like text wall, not a figure

    # If color variance is very low (< 200) and horizontal edges are mid, likely text
    if color_variance < 200 and horizontal_edges > 20:
        return False  # uniform dark-on-white = text

    return True
