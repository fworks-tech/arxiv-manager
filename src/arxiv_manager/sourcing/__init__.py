"""Sourcing pipeline: search, download, extract, filter."""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

from sqlmodel import select

from ..db import get_session
from ..models import Paper, Figure, ImageStatus
from .arxiv import search_papers, get_paper_url
from .downloader import download_pdf
from .extractor import extract_figures
from .filters import (
    compute_complexity,
    compute_file_hash,
    compute_perceptual_hash,
    is_likely_logo_or_icon,
    audit_figure,
)


def _process_one_paper(paper_data: dict, min_complexity: float) -> dict:
    """Download and extract one paper (runs in thread pool)."""
    paper_id = paper_data["id"]
    pdf_path = download_pdf(paper_id)
    extracted = extract_figures(pdf_path)
    return {"paper_id": paper_id, "paper_data": paper_data, "extracted": extracted}


def run_pipeline(
    terms: list[str] | None = None,
    domain: str | None = None,
    limit: int = 10,
    min_complexity: float = 0.3,
    max_figures_per_paper: int = 3,
) -> list[Figure]:
    """Run the full sourcing pipeline: search -> download -> extract -> filter.

    Downloads and extracts papers in parallel (max 4 workers).
    Filters figures by complexity, auto-classifies type/density, and
    rejects sparse/logos before DB insert.

    Returns list of Figure records that passed all filters.
    """
    session = get_session()

    # 1. Search
    papers = search_papers(terms=terms, domain=domain, limit=limit)
    print(f"Found {len(papers)} papers matching search criteria.")

    # 2. Upsert paper records (serial, fast)
    for paper_data in papers:
        paper_id = paper_data["id"]
        existing = session.get(Paper, paper_id)
        if not existing:
            paper = Paper(
                id=paper_id,
                title=paper_data.get("title", ""),
                license=paper_data.get("license", "CC0"),
                categories=paper_data.get("categories", ""),
                source=paper_data.get("source", "arXiv CC0"),
                pdf_url=get_paper_url(paper_id),
            )
            session.add(paper)
    session.commit()

    # 3. Download + extract in parallel
    print("Downloading and extracting papers in parallel...")
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_process_one_paper, p, min_complexity) for p in papers]
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  [red]Worker failed: {e}")

    # 4. Filter, classify, store
    new_figures: list[Figure] = []
    for res in results:
        paper_id = res["paper_id"]
        extracted = res["extracted"]
        if not extracted:
            continue

        for img_data in extracted:
            full_path = Path(__file__).parent.parent.parent.parent / "storage" / img_data["image_path"]
            if not full_path.exists():
                continue

            def _reject(reason: str):
                """Reject a figure file: log + delete from disk."""
                print(f"  [Reject {reason}] {img_data['image_path']}")
                try:
                    full_path.unlink(missing_ok=True)
                except Exception:
                    pass

            # --- Pre-DB filters ---
            if is_likely_logo_or_icon(full_path):
                _reject("logo/icon")
                continue

            # Run audit (complexity + type + density + sparseness)
            audit = audit_figure(full_path)
            if audit["is_likely_sparse"] or audit["is_logo_or_icon"] or audit.get("is_text_only"):
                reason = "text-only" if audit.get("is_text_only") else "sparse/logo"
                _reject(f"{reason} (complexity={audit['complexity_score']:.3f})")
                continue
            if audit["complexity_score"] < min_complexity:
                _reject(f"low complexity ({audit['complexity_score']:.3f} < {min_complexity})")
                continue
            if audit["filesize_bytes"] < 5000:
                _reject(f"tiny file ({audit['filesize_bytes']}B)")
                continue

            # Dedup via SHA256
            img_hash = compute_file_hash(full_path)
            existing_fig = session.exec(
                select(Figure).where(Figure.image_hash == img_hash)
            ).first()
            if existing_fig:
                _reject("SHA256 duplicate")
                continue

            # Near-duplicate dedup via perceptual hash (phash)
            new_phash = compute_perceptual_hash(full_path)
            is_near_dup = False
            if new_phash:
                from sqlmodel import col
                same_paper_figs = session.exec(
                    select(Figure).where(Figure.paper_id == paper_id).where(col(Figure.perceptual_hash) != "")
                ).all()
                import imagehash
                for existing in same_paper_figs:
                    try:
                        existing_hash = imagehash.hex_to_hash(existing.perceptual_hash)
                        new_hash_obj = imagehash.hex_to_hash(new_phash)
                        distance = existing_hash - new_hash_obj
                        if distance < 8:
                            _reject(f"phash near-dup ({img_data['image_path']} ~ {existing.image_path}, dist={distance})")
                            is_near_dup = True
                            break
                    except Exception:
                        pass
            if is_near_dup:
                continue

            # Build Figure with all audit fields
            figure = Figure(
                paper_id=paper_id,
                image_path=img_data["image_path"],
                image_hash=img_hash,
                perceptual_hash=new_phash,
                caption=img_data.get("caption", ""),
                page_num=img_data.get("page_num", 0),
                figure_num=img_data.get("figure_num", ""),
                width=audit["width"],
                height=audit["height"],
                width_height_ratio=audit["width_height_ratio"],
                filesize_bytes=audit["filesize_bytes"],
                complexity_score=audit["complexity_score"],
                figure_type=audit["figure_type"],
                is_dense=audit["is_dense"],
                is_suitable=audit["is_suitable"],
                status=ImageStatus.NEW.value,
            )
            session.add(figure)
            new_figures.append(figure)

        session.commit()

        # Per-paper cap: keep only top N by complexity
        paper_new = [f for f in new_figures if f.paper_id == paper_id]
        if len(paper_new) > max_figures_per_paper:
            paper_new.sort(key=lambda f: f.complexity_score, reverse=True)
            excess = paper_new[max_figures_per_paper:]
            for fig in excess:
                try:
                    Path(fig.full_path).unlink(missing_ok=True)
                except Exception:
                    pass
                session.delete(fig)
                new_figures.remove(fig)
            session.commit()
            print(f"  [{paper_id}] capped to {max_figures_per_paper} (kept top {max_figures_per_paper} by complexity)")
        print(f"  [{paper_id}] {len(extracted)} extracted, stored figures: {len([f for f in new_figures if f.paper_id == paper_id])}")

    print(f"\nTotal new figures added: {len(new_figures)}")
    for fig in new_figures[:10]:
        status = "✓ HIGH" if fig.complexity_score >= min_complexity else "✗ LOW"
        print(f"  [{status}] {fig.image_path} (type={fig.figure_type}, complexity={fig.complexity_score:.3f})")
    if len(new_figures) > 10:
        print(f"  ... and {len(new_figures) - 10} more")

    return new_figures
