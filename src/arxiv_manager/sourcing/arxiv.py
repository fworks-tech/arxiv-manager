"""arXiv CC0 index fetcher and searcher with automatic query expansion."""

from __future__ import annotations

import re
from typing import Any

import httpx

INDEX_URL = "https://stella-sirius-arxiv-search.vercel.app"
MANIFEST_URL = f"{INDEX_URL}/index-manifest.json"

_index_cache: list[dict[str, Any]] | None = None

# --- Query expansion (Tier 3): maps a search term to synonyms
# When a user searches "detection", papers with "detecting", "detector",
# "segmentation", etc. are also found. Matches Challenging-suitable domains.
QUERY_EXPANSION: dict[str, list[str]] = {
    "detection": ["detection", "detecting", "detector", "yolo", "faster r-cnn"],
    "segmentation": ["segmentation", "segmenting", "mask", "u-net", "boundary"],
    "neural network": ["neural", "network", "deep learning", "cnn", "transformer"],
    "optical": ["optical", "photon", "photonic", "lens", "mirror", "microscopy"],
    "classification": ["classification", "classifier", "classify", "categorization"],
    "object detection": ["object detection", "detection", "localization", "yolo", "r-cnn"],
    "counting": ["counting", "count", "enumeration", "density estimation"],
    "lattice": ["lattice", "grid", "mesh", "array", "tiling"],
    "benchmark": ["benchmark", "leaderboard", "evaluation", "comparison"],
    "architecture": ["architecture", "backbone", "design", "diagram"],
}


def _load_index() -> list[dict[str, Any]]:
    """Load all index records from the Vercel-hosted static JSON files."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    manifest = httpx.get(MANIFEST_URL, timeout=30).json()
    all_records: list[dict[str, Any]] = []
    for entry in manifest:
        url = f"{INDEX_URL}{entry['file']}"
        data = httpx.get(url, timeout=60).json()
        all_records.extend(data)

    _index_cache = all_records
    return _index_cache


def _expand_terms(terms: list[str]) -> list[str]:
    """Expand each term with its synonyms; preserves order but dedups.

    "detection" → ["detection", "detecting", "detector", "yolo", "faster r-cnn"]
    Unknown terms pass through unchanged.
    """
    expanded: list[str] = []
    seen = set()
    for t in terms:
        expansions = QUERY_EXPANSION.get(t.lower(), [t])
        for e in expansions[:6]:  # cap at 6 per term
            if e.lower() not in seen:
                expanded.append(e)
                seen.add(e.lower())
    return expanded  # no global cap — up to 18 for 3 terms


def search_papers(
    terms: list[str] | None = None,
    domain: str | None = None,
    source: str = "arXiv CC0",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search the CC0 index by title terms and domain category.

    Terms are automatically expanded with synonyms (Tier 3 optimization).
    Uses OR-matching per term group: any expanded synonym must match.

    Args:
        terms: Up to 3 title search terms (AND logic between groups).
        domain: Domain string (e.g. "computer science", "math", "bio").
        source: Which index to search (default: "arXiv CC0").
        limit: Max results to return.
    """
    records = _load_index()

    # Filter by source
    if source:
        records = [r for r in records if r.get("source") == source]

    # Filter by title terms (AND between groups, OR within each group)
    if terms:
        for term in terms[:3]:
            # Expand the term into synonyms
            expanded = _expand_terms([term])
            # OR-match: any of the expanded synonyms must match the title
            pattern = re.compile(
                r"(^|[^a-z0-9])(" + "|".join(re.escape(e) for e in expanded) + r")($|[^a-z0-9])",
                re.IGNORECASE,
            )
            records = [r for r in records if pattern.search(r.get("title", ""))]

    # Filter by domain/category
    if domain:
        domain_keywords = _expand_domain(domain)
        if domain_keywords:
            records = [
                r for r in records
                if _matches_domain(r.get("categories", ""), domain_keywords)
            ]

    return records[:limit]


def _expand_domain(domain: str) -> list[str]:
    """Expand a domain string into search keywords."""
    mapping = {
        "computer science": ["cs", "computer science"],
        "cs": ["cs", "computer science"],
        "math": ["math", "mathematics"],
        "mathematics": ["math", "mathematics"],
        "bio": ["q-bio", "biology", "quantitative biology"],
        "biology": ["q-bio", "biology", "quantitative biology"],
        "chemistry": ["chem-ph", "chemistry"],
        "finance": ["q-fin", "finance", "quantitative finance"],
        "physics": ["physics"],
        "statistics": ["stat", "statistics"],
        "medicine": ["med", "medicine"],
        "neuroscience": ["neuro", "neuroscience"],
        "economics": ["econ", "economics"],
        "engineering": ["engine", "engineering"],
    }
    key = domain.lower().strip()
    return mapping.get(key, [key])


def _matches_domain(categories: str, keywords: list[str]) -> bool:
    """Check if paper categories match any of the domain keywords."""
    cats = categories.split()
    for cat in cats:
        for kw in keywords:
            if cat.lower() == kw.lower() or cat.lower().startswith(kw.lower()):
                return True
            if len(kw) >= 4 and kw.lower() in cat.lower():
                return True
    return False


def get_paper_url(paper_id: str) -> str:
    """Get the PDF URL for a paper."""
    return f"https://arxiv.org/pdf/{paper_id}.pdf"
