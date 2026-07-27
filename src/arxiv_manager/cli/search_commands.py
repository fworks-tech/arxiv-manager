"""Search and fetch commands for arXiv papers."""

import typer
from rich.table import Table
from sqlmodel import select

from ..db import get_session
from ..models import Figure, ImageStatus, Paper
from ..sourcing import run_pipeline
from ..sourcing.arxiv import get_paper_url
from ..sourcing.arxiv import search_papers as do_search
from ..sourcing.downloader import download_pdf
from ..sourcing.extractor import extract_figures
from ..sourcing.filters import compute_complexity, compute_file_hash, is_likely_logo_or_icon
from ..storage import STORAGE_DIR
from . import console, search_app


@search_app.command("papers")
def search_papers_cmd(
    domain: str = typer.Option("", "--domain", "-d", help="Domain (e.g. 'computer science', 'math')"),
    terms: str = typer.Option("", "--terms", "-t", help="Comma-separated title search terms"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
):
    """Search arXiv CC0 papers by domain and terms."""
    term_list = [t.strip() for t in terms.split(",") if t.strip()] if terms else None
    results = do_search(terms=term_list, domain=domain or None, limit=limit)

    if not results:
        console.print("[yellow]No papers found matching criteria.[/]")
        return

    table = Table(title=f"Found {len(results)} CC0 papers")
    table.add_column("ID", style="cyan")
    table.add_column("Title", max_width=60)
    table.add_column("Categories", style="dim")
    table.add_column("Source")

    for r in results:
        table.add_row(r["id"], r["title"][:60], r.get("categories", ""), r.get("source", ""))

    console.print(table)


@search_app.command("fetch")
def fetch_paper_cmd(
    paper_id: str = typer.Argument(..., help="arXiv paper ID (e.g. 2301.12345)"),
    min_complexity: float = typer.Option(0.3, "--min-complexity"),
):
    """Download a paper and extract figures."""
    session = get_session()
    console.print(f"[bold]Fetching paper {paper_id}...[/]")

    existing = session.get(Paper, paper_id)
    if not existing:
        paper = Paper(
            id=paper_id,
            title=paper_id,
            license="CC0",
            categories="",
            source="arXiv CC0",
            pdf_url=get_paper_url(paper_id),
        )
        session.add(paper)
        session.commit()

    pdf_path = download_pdf(paper_id)
    console.print(f"  Downloaded: {paper_id}.pdf")

    extracted = extract_figures(pdf_path)
    console.print(f"  Extracted {len(extracted)} images from {paper_id}")

    new_figures = []
    for img_data in extracted:
        full_path = STORAGE_DIR / img_data["image_path"]

        if is_likely_logo_or_icon(full_path):
            continue

        img_hash = compute_file_hash(full_path)
        existing_fig = session.exec(
            select(Figure).where(Figure.image_hash == img_hash)
        ).first()
        if existing_fig:
            continue

        complexity = compute_complexity(full_path)
        figure = Figure(
            paper_id=paper_id,
            image_path=img_data["image_path"],
            image_hash=img_hash,
            caption=img_data.get("caption", ""),
            page_num=img_data.get("page_num", 0),
            figure_num=img_data.get("figure_num", ""),
            width=img_data.get("width", 0),
            height=img_data.get("height", 0),
            complexity_score=complexity,
            status=ImageStatus.NEW.value,
        )
        session.add(figure)
        new_figures.append(figure)

    session.commit()

    if new_figures:
        console.print(f"\n[green]Added {len(new_figures)} new figures:[/]")
        for fig in new_figures:
            status = "✓ HIGH" if fig.complexity_score >= min_complexity else "✗ LOW"
            console.print(f"  [{status}] {fig.image_path} (complexity={fig.complexity_score:.3f})")
    else:
        console.print("[yellow]No new figures extracted.[/]")


@search_app.command("fetch-many")
def fetch_many_cmd(
    domain: str = typer.Option("", "--domain", "-d"),
    terms: str = typer.Option("", "--terms", "-t"),
    limit: int = typer.Option(5, "--limit", "-l"),
    min_complexity: float = typer.Option(0.3, "--min-complexity"),
    max_figures_per_paper: int = typer.Option(3, "--max-figures", help="Keep at most N figures per paper"),
):
    """Run the full pipeline: search, download, extract, filter."""
    term_list = [t.strip() for t in terms.split(",") if t.strip()] if terms else None
    console.print(f"[bold]Running pipeline: domain={domain}, terms={terms}, limit={limit}[/]")
    figures = run_pipeline(
        terms=term_list,
        domain=domain or None,
        limit=limit,
        min_complexity=min_complexity,
        max_figures_per_paper=max_figures_per_paper,
    )
    console.print(f"\n[green]Pipeline complete. {len(figures)} new figures added.[/]")
