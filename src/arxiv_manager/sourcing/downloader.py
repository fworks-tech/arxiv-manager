"""PDF downloader."""

from __future__ import annotations

from pathlib import Path

import httpx

from ..storage import PAPERS_DIR


def download_pdf(paper_id: str, url: str | None = None) -> Path:
    """Download a paper PDF and save it locally.

    Args:
        paper_id: arXiv paper ID (e.g. "2301.12345").
        url: PDF URL. If None, constructs from paper_id.

    Returns:
        Path to the downloaded PDF.
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPERS_DIR / f"{paper_id}.pdf"

    if dest.exists():
        return dest

    if url is None:
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    resp = httpx.get(url, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
