"""CLI interface using Typer + Rich."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .db import init_db

app = typer.Typer(
    name="arxiv-manager",
    help="ArXiv Manager Task Authoring Assistant",
    add_completion=False,
)
console = Console()

# Sub-commands
search_app = typer.Typer(help="Search and source images from arXiv CC0")
task_app = typer.Typer(help="Create and manage tasks")
images_app = typer.Typer(help="Audit, clean, reclassify, and rescore image library")
app.add_typer(search_app, name="search")
app.add_typer(task_app, name="task")
app.add_typer(images_app, name="images")


@app.callback()
def main():
    """ArXiv Manager Assistant — automate your task authoring workflow."""
    init_db()


@app.command("check")
def check_api():
    """Verify API connectivity, model response, and DB health."""
    import os
    import time
    import httpx
    from pathlib import Path
    from sqlmodel import select
    from .db import get_session
    from .models import Figure, Paper, Task

    console.print("[bold]Pre-flight checks[/]\n")

    # Check 1: API key
    console.print("1. API key...", end=" ")
    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        console.print("[red]✗ Missing OPENCODE_API_KEY[/]")
    else:
        console.print(f"[green]✓ Found ({key[:8]}...{key[-4:]})[/]")

    # Check 2: Model responds
    console.print("2. Model connectivity (minimax-m3)...", end=" ")
    try:
        start = time.time()
        resp = httpx.post(
            "https://opencode.ai/zen/go/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "minimax-m3",
                "messages": [{"role": "user", "content": "Reply: ok"}],
                "max_tokens": 10,
            },
            timeout=30,
        )
        elapsed = time.time() - start
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content.strip():
            console.print(f"[green]✓ Responded in {elapsed:.1f}s (content={len(content)}c)[/]")
        else:
            console.print(f"[red]✗ Empty response[/]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/]")

    # Check 3: DB connectivity
    console.print("3. Database...", end=" ")
    try:
        session = get_session()
        fig_count = len(list(session.exec(select(Figure)).all()))
        paper_count = len(list(session.exec(select(Paper)).all()))
        task_count = len(list(session.exec(select(Task)).all()))
        console.print(f"[green]✓ Connected — {paper_count} papers, {fig_count} figures, {task_count} tasks[/]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/]")

    # Check 4: Model can draft (text-only, tests think tag + basic output)
    console.print("4. Draft API (text-only, no image)...", end=" ")
    try:
        resp = httpx.post(
            "https://opencode.ai/zen/go/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "minimax-m3",
                "messages": [{"role": "user", "content": "Reply with exactly the word: OK"}],
                "max_tokens": 20,
            },
            timeout=30,
        )
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Strip think blocks before checking
        import re
        stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "OK" in stripped:
            console.print(f"[green]✓ Model responds (content={len(content)}c)[/]")
        else:
            console.print(f"[yellow]⚠ Unexpected response[/]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/]")

    console.print(f"\n[bold green]All checks complete.[/]")


# ─── SEARCH COMMANDS ──────────────────────────────────────────────

@search_app.command("papers")
def search_papers(
    domain: str = typer.Option("", "--domain", "-d", help="Domain (e.g. 'computer science', 'math')"),
    terms: str = typer.Option("", "--terms", "-t", help="Comma-separated title search terms"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
):
    """Search arXiv CC0 papers by domain and terms."""
    from .sourcing.arxiv import search_papers as do_search

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
def fetch_paper(
    paper_id: str = typer.Argument(..., help="arXiv paper ID (e.g. 2301.12345)"),
    min_complexity: float = typer.Option(0.3, "--min-complexity"),
):
    """Download a paper and extract figures."""
    from .db import get_session
    from .models import Paper, Figure, ImageStatus
    from .sourcing.arxiv import get_paper_url
    from .sourcing.downloader import download_pdf
    from .sourcing.extractor import extract_figures
    from .sourcing.filters import compute_complexity, compute_file_hash, is_likely_logo_or_icon
    from sqlmodel import select

    session = get_session()
    console.print(f"[bold]Fetching paper {paper_id}...[/]")

    # Upsert paper record
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

    # Download PDF
    pdf_path = download_pdf(paper_id)
    console.print(f"  Downloaded: {paper_id}.pdf")

    # Extract figures
    extracted = extract_figures(pdf_path)
    console.print(f"  Extracted {len(extracted)} images from {paper_id}")

    # Filter and store
    from .storage import STORAGE_DIR
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
        console.print("[yellow]No new figures extracted (too small, low complexity, or already exist).[/]")


@search_app.command("fetch-many")
def fetch_many(
    domain: str = typer.Option("", "--domain", "-d"),
    terms: str = typer.Option("", "--terms", "-t"),
    limit: int = typer.Option(5, "--limit", "-l"),
    min_complexity: float = typer.Option(0.3, "--min-complexity"),
    max_figures_per_paper: int = typer.Option(3, "--max-figures", help="Keep at most N figures per paper"),
):
    """Run the full pipeline: search, download, extract, filter."""
    from .sourcing import run_pipeline

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


# ─── IMAGE COMMANDS ───────────────────────────────────────────────

@images_app.command("list")
def list_images(
    status: str = typer.Option("", "--status", "-s", help="Filter by status: new, reviewed, used, rejected"),
    min_complexity: float = typer.Option(0, "--min-complexity", help="Min complexity score"),
    figure_type: str = typer.Option("", "--type", help="Filter by figure type: chart_graph_text | general_image"),
    suitable_only: bool = typer.Option(False, "--suitable", help="Show only is_suitable figures"),
    limit: int = typer.Option(30, "--limit", "-l"),
):
    """List images in the library."""
    from sqlmodel import select
    from .db import get_session
    from .models import Figure

    session = get_session()
    query = select(Figure)
    if status:
        query = query.where(Figure.status == status)
    if min_complexity > 0:
        query = query.where(Figure.complexity_score >= min_complexity)
    if figure_type:
        query = query.where(Figure.figure_type == figure_type)
    if suitable_only:
        query = query.where(Figure.is_suitable == True)  # noqa: E712
    query = query.order_by(Figure.complexity_score.desc()).limit(limit)

    figures = list(session.exec(query).all())
    if not figures:
        console.print("[yellow]No images found.[/]")
        return

    table = Table(title=f"Image Library ({len(figures)} shown)")
    table.add_column("ID", style="cyan")
    table.add_column("Path")
    table.add_column("Type", style="magenta")
    table.add_column("Complex", justify="right")
    table.add_column("Dense", justify="center")
    table.add_column("Size")
    table.add_column("KB", justify="right")
    table.add_column("Status")
    table.add_column("Caption", max_width=40, style="dim")

    for f in figures:
        status_style = {"new": "green", "used": "blue", "rejected": "red"}.get(f.status, "white")
        type_short = {"chart_graph_text": "chart", "general_image": "img"}.get(f.figure_type, "-")
        kb = (f.filesize_bytes or 0) / 1024
        table.add_row(
            str(f.id),
            f.image_path,
            type_short,
            f"{f.complexity_score:.3f}",
            "✓" if f.is_dense else "",
            f"{f.width}x{f.height}",
            f"{kb:.0f}",
            f"[{status_style}]{f.status}[/]",
            (f.caption[:40] + "...") if len(f.caption) > 40 else f.caption,
        )

    console.print(table)


@images_app.command("audit")
def audit_images():
    """Print image-library health report (disk vs DB, trash, orphans, complexity distribution)."""
    import glob
    import os
    import sqlite3

    from sqlmodel import select
    from .db import get_session
    from .models import Figure

    session = get_session()
    conn = sqlite3.connect("storage/arxiv-manager.db")
    c = conn.cursor()

    # Disk
    disk_files = [os.path.basename(f) for f in glob.glob("storage/figures/*.png")]
    # DB
    db_files = []
    for f in session.exec(select(Figure)).all():
        db_files.append(os.path.basename(f.image_path))
    # Broken DB rows (file missing)
    c.execute("SELECT image_path FROM figures")
    db_paths = [r[0] for r in c.fetchall()]
    broken = [p for p in db_paths if not os.path.exists(f"storage/{p}")]
    # Orphans (file on disk, not in DB)
    orphans = sorted(set(disk_files) - set(db_files))
    # Trash (small files)
    trash = []
    sizes = []
    for f in disk_files:
        path = f"storage/figures/{f}"
        if os.path.exists(path):
            sz = os.path.getsize(path)
            sizes.append((f, sz))
            if sz < 5000:
                trash.append((f, sz))
    sizes.sort(key=lambda x: x[1])
    # Complexity distribution
    c.execute("SELECT complexity_score FROM figures")
    scores = [r[0] for r in c.fetchall() if r[0] is not None]
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    dist = {}
    for lo, hi in buckets:
        dist[f"[{lo:.1f},{hi:.2f})"] = sum(1 for s in scores if lo <= s < hi)
    # Status counts
    c.execute("SELECT status, COUNT(*) FROM figures GROUP BY status")
    by_status = dict(c.fetchall())
    # Type counts
    c.execute("SELECT figure_type, COUNT(*) FROM figures WHERE figure_type != '' GROUP BY figure_type")
    by_type = dict(c.fetchall())
    # Suitable count
    c.execute("SELECT COUNT(*) FROM figures WHERE is_suitable = 1")
    suitable = c.fetchone()[0]
    conn.close()

    table = Table(title="Image Library Health Report", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="center")

    def status_icon(ok: bool) -> str:
        return "[green]✓[/]" if ok else "[red]✗[/]"

    table.add_row("Files on disk", str(len(disk_files)), status_icon(len(disk_files) == len(db_files)))
    table.add_row("Rows in DB", str(len(db_files)), status_icon(len(disk_files) == len(db_files)))
    table.add_row("Orphans (disk, no DB)", str(len(orphans)), status_icon(len(orphans) == 0))
    table.add_row("Broken (DB, no file)", str(len(broken)), status_icon(len(broken) == 0))
    table.add_row("Trash files (<5KB)", str(len(trash)), status_icon(len(trash) == 0))
    table.add_row("Suitable figures", f"{suitable} ({100*suitable/max(len(db_files),1):.1f}%)", "")
    table.add_row("", "", "")
    table.add_row("[bold]Complexity distribution[/]", "", "")
    for k, v in dist.items():
        table.add_row(f"  {k}", str(v), "")
    table.add_row("", "", "")
    table.add_row("[bold]Status[/]", "", "")
    for s, n in by_status.items():
        table.add_row(f"  {s}", str(n), "")
    if by_type:
        table.add_row("", "", "")
        table.add_row("[bold]Figure type[/]", "", "")
        for t, n in by_type.items():
            table.add_row(f"  {t}", str(n), "")

    console.print(table)

    if orphans:
        console.print("\n[yellow]Orphans (first 10):[/]")
        for f in orphans[:10]:
            console.print(f"  {f}")
    if broken:
        console.print("\n[red]Broken DB rows:[/]")
        for p in broken[:10]:
            console.print(f"  {p}")
    if trash:
        console.print("\n[red]Trash files:[/]")
        for f, sz in trash:
            console.print(f"  {sz:>6d}  {f}")


@images_app.command("clean")
def clean_images(
    no_backup: bool = typer.Option(False, "--no-backup", help="Skip DB backup before destructive ops"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Remove trash files, broken DB rows, and move orphans to _trash/."""
    import glob
    import os
    import sqlite3
    import shutil
    from datetime import datetime
    from sqlmodel import select

    from .db import get_session
    from .models import Figure

    if not yes and not Confirm.ask("This will delete trash + broken rows and move orphans. Continue?"):
        raise typer.Exit(0)

    conn = sqlite3.connect("storage/arxiv-manager.db")
    c = conn.cursor()

    if not no_backup:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"storage/arxiv-manager.db.clean-{ts}"
        shutil.copy2("storage/arxiv-manager.db", backup)
        console.print(f"[green]Backed up DB to {backup}[/]")

    # 1. Trash files (<5KB) — remove from disk + DB
    disk_files = [(f, os.path.getsize(f"storage/figures/{f}")) for f in os.listdir("storage/figures") if f.endswith(".png")]
    trash = [(f, s) for f, s in disk_files if s < 5000]
    trash_removed = 0
    for f, sz in trash:
        path = f"storage/figures/{f}"
        try:
            os.remove(path)
            trash_removed += 1
        except OSError:
            pass
        c.execute("DELETE FROM figures WHERE image_path = ?", (f"figures/{f}",))
    console.print(f"Removed {trash_removed} trash files")

    # 2. Broken DB rows
    c.execute("SELECT image_path FROM figures")
    db_paths = [r[0] for r in c.fetchall()]
    broken_removed = 0
    for p in db_paths:
        if not os.path.exists(f"storage/{p}"):
            c.execute("DELETE FROM figures WHERE image_path = ?", (p,))
            broken_removed += 1
    console.print(f"Removed {broken_removed} broken DB rows")

    # 3. Orphans → _trash/orphans_<ts>/
    disk_now = set(f for f in os.listdir("storage/figures") if f.endswith(".png"))
    c.execute("SELECT image_path FROM figures")
    db_now = set(os.path.basename(r[0]) for r in c.fetchall())
    orphans = sorted(disk_now - db_now)
    if orphans:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        trash_dir = f"storage/_trash/orphans_{ts}"
        os.makedirs(trash_dir, exist_ok=True)
        for f in orphans:
            shutil.move(f"storage/figures/{f}", f"{trash_dir}/{f}")
        console.print(f"Moved {len(orphans)} orphans to {trash_dir}")

    # 4. Deduplicate DB rows (keep matching disk hash)
    import hashlib
    c.execute("""
        SELECT image_path, COUNT(*) as cnt FROM figures
        GROUP BY image_path HAVING cnt > 1
    """)
    dupes = c.fetchall()
    deduped = 0
    for path, _ in dupes:
        disk_path = f"storage/{path}"
        if not os.path.exists(disk_path):
            continue
        with open(disk_path, "rb") as fh:
            disk_hash = hashlib.sha256(fh.read()).hexdigest()
        c.execute("SELECT id, image_hash FROM figures WHERE image_path = ?", (path,))
        rows = c.fetchall()
        matching = [r for r in rows if r[1] == disk_hash]
        stale = [r for r in rows if r[1] != disk_hash]
        for row in stale:
            c.execute("SELECT COUNT(*) FROM tasks WHERE figure_id = ?", (row[0],))
            if c.fetchone()[0] == 0:
                c.execute("DELETE FROM figures WHERE id = ?", (row[0],))
                deduped += 1
    console.print(f"Deduped {deduped} duplicate DB rows")

    conn.commit()
    c.execute("SELECT COUNT(*) FROM figures")
    db_count = c.fetchone()[0]
    disk_count = len([f for f in os.listdir("storage/figures") if f.endswith(".png")])
    conn.close()

    console.print(f"\n[green]Done. Disk: {disk_count} files, DB: {db_count} rows.[/]")


@images_app.command("reclassify")
def reclassify_images(
    limit: int = typer.Option(0, "--limit", "-l", help="Max figures to process (0 = all)"),
):
    """Run classify_figure_type() and audit_figure() on all figures, updating DB fields."""
    from sqlmodel import select
    from .db import get_session
    from .models import Figure
    from .sourcing.filters import audit_figure

    session = get_session()
    query = select(Figure)
    if limit > 0:
        query = query.limit(limit)
    figures = list(session.exec(query).all())
    console.print(f"Reclassifying {len(figures)} figures...")

    type_counts: dict[str, int] = {}
    suitable_count = 0
    for i, fig in enumerate(figures, 1):
        full_path = fig.full_path
        if not full_path.exists():
            continue
        try:
            audit = audit_figure(full_path)
            fig.width = audit["width"]
            fig.height = audit["height"]
            fig.width_height_ratio = audit["width_height_ratio"]
            fig.filesize_bytes = audit["filesize_bytes"]
            fig.complexity_score = audit["complexity_score"]
            fig.figure_type = audit["figure_type"]
            fig.is_dense = audit["is_dense"]
            fig.is_suitable = audit["is_suitable"]
            session.add(fig)
            type_counts[audit["figure_type"]] = type_counts.get(audit["figure_type"], 0) + 1
            if audit["is_suitable"]:
                suitable_count += 1
        except Exception as e:
            console.print(f"  [red]Failed on {fig.image_path}: {e}[/]")

        if i % 50 == 0:
            session.commit()
            console.print(f"  ...{i}/{len(figures)}")

    session.commit()
    console.print(f"\n[green]Done. Type counts: {type_counts}. Suitable: {suitable_count}/{len(figures)}[/]")


@images_app.command("rescore")
def rescore_images(
    limit: int = typer.Option(0, "--limit", "-l", help="Max figures to process (0 = all)"),
):
    """Re-run improved compute_complexity() on all figures."""
    from sqlmodel import select
    from .db import get_session
    from .models import Figure
    from .sourcing.filters import compute_complexity

    session = get_session()
    query = select(Figure)
    if limit > 0:
        query = query.limit(limit)
    figures = list(session.exec(query).all())
    console.print(f"Rescoring {len(figures)} figures with new density-aware scorer...")

    for i, fig in enumerate(figures, 1):
        full_path = fig.full_path
        if not full_path.exists():
            continue
        try:
            fig.complexity_score = compute_complexity(full_path)
            session.add(fig)
        except Exception as e:
            console.print(f"  [red]Failed on {fig.image_path}: {e}[/]")

        if i % 50 == 0:
            session.commit()
            console.print(f"  ...{i}/{len(figures)}")

    session.commit()
    console.print(f"\n[green]Done. Rescored {len(figures)} figures.[/]")


@images_app.command("rank")
def rank_images(
    min_complexity: float = typer.Option(0.4, "--min-complexity", help="Min complexity threshold"),
    limit: int = typer.Option(20, "--limit", "-l", help="Top N to show"),
):
    """Show top candidates for Challenging tasks, ranked by density + complexity + type."""
    from sqlmodel import select
    from .db import get_session
    from .models import Figure

    session = get_session()
    query = (
        select(Figure)
        .where(Figure.complexity_score >= min_complexity)
        .where(Figure.is_suitable == True)  # noqa: E712
    )
    figures = list(session.exec(query).all())
    # Rank: complexity * (1 + is_dense) * (1 + 0.2 if chart_graph_text)
    def score(f: Figure) -> float:
        s = f.complexity_score
        if f.is_dense:
            s *= 1.3
        if f.figure_type == "chart_graph_text":
            s *= 1.1
        return s
    figures.sort(key=score, reverse=True)
    figures = figures[:limit]

    if not figures:
        console.print("[yellow]No candidates found at this threshold. Try lowering --min-complexity.[/]")
        return

    table = Table(title=f"Top {len(figures)} Challenging Candidates")
    table.add_column("ID", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Complex", justify="right")
    table.add_column("Dense", justify="center")
    table.add_column("Size")
    table.add_column("Path")

    for f in figures:
        table.add_row(
            str(f.id),
            f"{score(f):.3f}",
            {"chart_graph_text": "chart", "general_image": "img"}.get(f.figure_type, "-"),
            f"{f.complexity_score:.3f}",
            "✓" if f.is_dense else "",
            f"{f.width}x{f.height}",
            f.image_path,
        )

    console.print(table)


# ─── TASK COMMANDS ────────────────────────────────────────────────

@task_app.command("new")
def create_task(
    image_id: int = typer.Option(..., "--image-id", "-i", help="Figure ID from the library"),
    ai: bool = typer.Option(False, "--ai", help="Use AI to draft Q&A"),
    hardest: bool = typer.Option(False, "--hardest", help="Generate HARDEST-classified question (Qwen-exploiting)"),
    challenging: bool = typer.Option(False, "--challenging", help="Generate CHALLENGING-classified question (Qwen-fail, Gemini-pass)"),
    draft_attempts: int = typer.Option(1, "--draft-attempts", help="Number of independent draft attempts (1=fast, 3=best)"),
    model: str = typer.Option("minimax-m3", "--model", help="Model ID (minimax-m3, kimi-k2.7-code)"),
    title: str = typer.Option("", "--title", "-t", help="Task title for the UI"),
):
    """Create a new task for an image."""
    from sqlmodel import select
    from .db import get_session
    from .models import Figure
    from .authoring import create_task as do_create
    from .authoring.validator import validate_task

    if hardest and challenging:
        console.print("[red]Cannot specify both --hardest and --challenging.[/]")
        raise typer.Exit(1)

    session = get_session()
    figure = session.get(Figure, image_id)
    if not figure:
        console.print(f"[red]Image {image_id} not found.[/]")
        raise typer.Exit(1)

    diff_label = "HARDEST" if hardest else ("CHALLENGING" if challenging else "")
    console.print(Panel(
        f"Image: {figure.image_path}\nCaption: {figure.caption}\n"
        f"Complexity: {figure.complexity_score:.3f}\n"
        f"Type: {figure.figure_type or 'unknown'}\n"
        f"Dense: {figure.is_dense}\n"
        f"Target: {diff_label or 'manual'}",
        title="Task Source"
    ))

    if ai or hardest or challenging:
        console.print("[bold]Generating AI draft...[/]")
        from .authoring.ai_draft import draft_qa, draft_qa_consensus
        import os
        difficulty = "hardest" if hardest else ("challenging" if challenging else "")
        if draft_attempts > 1:
            draft = draft_qa_consensus(
                figure.full_path, n_attempts=draft_attempts, verify=True,
                caption=figure.caption, provider="opencode",
                api_key=os.environ.get("OPENCODE_API_KEY"), difficulty=difficulty,
                figure_type=figure.figure_type, complexity_score=figure.complexity_score,
                model=model,
            )
        else:
            draft = draft_qa(
                figure.full_path, caption=figure.caption, provider="opencode",
                api_key=os.environ.get("OPENCODE_API_KEY"), difficulty=difficulty,
                figure_type=figure.figure_type, complexity_score=figure.complexity_score,
                model=model,
            )
        if draft:
            console.print(f"[green]AI drafted:[/]\n  Q: {draft['question']}\n  A: {draft['answer']}")
            if not Confirm.ask("Use this draft?"):
                draft = None
        else:
            console.print("[yellow]AI draft failed (no API key?). Falling back to manual entry.[/]")

    domain = Prompt.ask("Domain", default="Computer Science")
    if not ai or not draft:
        if not title:
            title = Prompt.ask("Task title", default=figure.caption[:60].strip() if figure.caption else "")
        question = Prompt.ask("Question")
        answer = Prompt.ask("Answer")
        answer_format = Prompt.ask("Answer format", choices=["number", "word", "phrase", "year", "percent", "integer"], default="word")
        task_type = Prompt.ask("Task type", choices=["chart", "general_image", "spatial"], default="chart")
    else:
        question = draft["question"]
        answer = draft["answer"]
        answer_format = draft.get("answer_format", "word")
        task_type = draft.get("task_type", "chart")
        if not title:
            title = figure.caption[:60].strip() if figure.caption else question[:60].strip()

    # Validate before saving
    validation = validate_task(question, answer, answer_format)
    console.print(f"\n[bold]Validation:[/]\n{validation.summary()}")

    if not validation.is_valid:
        if not Confirm.ask("\nSave anyway despite errors?"):
            raise typer.Exit(0)

    task = do_create(
        figure_id=image_id,
        title=title,
        domain=domain,
        question=question,
        answer=answer,
        answer_format=answer_format,
        task_type=task_type,
        ai_generated=(ai or hardest or challenging) and draft is not None,
    )
    console.print(f"\n[green]Task #{task.id} created (status: draft).[/]")


@task_app.command("validate")
def validate_existing(
    task_id: int = typer.Argument(..., help="Task ID to validate"),
    no_regen: bool = typer.Option(False, "--no-regen", help="Don't auto-regenerate on errors"),
    auto: bool = typer.Option(False, "--auto", help="Auto-regenerate without confirmation"),
    hardest: bool = typer.Option(False, "--hardest", help="Use HARDEST prompt for regeneration (Qwen-exploiting)"),
    challenging: bool = typer.Option(False, "--challenging", help="Use CHALLENGING prompt for regeneration (Qwen-fail, Gemini-pass)"),
    model: str = typer.Option("minimax-m3", "--model", help="Model ID (minimax-m3, kimi-k2.7-code)"),
):
    """Validate an existing task against handbook rules.
    
    When validation finds errors, the tool can automatically regenerate the Q&A
    using AI with the validation errors as feedback. Use --no-regen to skip this.
    Use --auto to skip confirmation prompts.
    Use --hardest to generate Qwen-exploiting questions.
    """
    from .db import get_session
    from .models import Task, Figure
    from .authoring import update_task
    from .authoring.validator import validate_task
    from .authoring.ai_draft import draft_qa
    import os

    session = get_session()
    task = session.get(Task, task_id)
    if not task:
        console.print(f"[red]Task {task_id} not found.[/]")
        raise typer.Exit(1)

    figure = session.get(Figure, task.figure_id)
    validation = validate_task(task.question, task.answer, task.answer_format, figure.image_path if figure else "")

    console.print(f"[bold]Task #{task_id} Validation:[/]\n{validation.summary()}")

    # Auto-regenerate if there are errors and --no-regen is not set
    if not validation.is_valid and not no_regen:
        errors = "; ".join(validation.errors)
        warnings = "; ".join(validation.warnings)
        feedback = f"Errors: {errors}"
        if warnings:
            feedback += f"\nWarnings: {warnings}"

        console.print(f"\n[yellow]Validation errors found:[/]")
        for e in validation.errors:
            console.print(f"  ❌ {e}")

        if not auto:
            if not Confirm.ask("\nAuto-regenerate with AI to fix these errors?"):
                raise typer.Exit(0)

        if not figure:
            console.print("[red]No figure attached — cannot regenerate.[/]")
            raise typer.Exit(1)

        console.print("[bold]Regenerating Q&A with feedback...[/]")
        api_key = os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            console.print("[red]No OPENCODE_API_KEY set — cannot regenerate.[/]")
            raise typer.Exit(1)

        draft = draft_qa(
            figure.full_path,
            caption=figure.caption,
            provider="opencode",
            api_key=api_key,
            feedback=feedback,
            difficulty="hardest" if hardest else ("challenging" if challenging else ""),
            figure_type=figure.figure_type,
            complexity_score=figure.complexity_score,
            model=model,
        )

        if draft:
            console.print(f"\n[green]New draft:[/]\n  Q: {draft['question']}\n  A: {draft['answer']}")
            if auto or Confirm.ask("Apply this draft?"):
                task = update_task(
                    task_id,
                    question=draft["question"],
                    answer=draft["answer"],
                    answer_format=draft.get("answer_format", task.answer_format),
                    task_type=draft.get("task_type", task.task_type),
                )
                # Re-validate after update
                validation = validate_task(task.question, task.answer, task.answer_format, figure.image_path if figure else "")
                console.print(f"\n[bold]Re-validation after update:[/]\n{validation.summary()}")
            else:
                console.print("[dim]Keeping original draft.[/]")
        else:
            console.print("[red]AI regeneration failed — keeping original.[/]")


@task_app.command("list")
def list_tasks(
    status: str = typer.Option("", "--status", "-s"),
    limit: int = typer.Option(30, "--limit", "-l"),
):
    """List tasks."""
    from .authoring import list_tasks as do_list

    tasks = do_list(status=status or None, limit=limit)
    if not tasks:
        console.print("[yellow]No tasks found.[/]")
        return

    table = Table(title=f"Tasks ({len(tasks)} shown)")
    table.add_column("ID", style="cyan")
    table.add_column("Title", max_width=30)
    table.add_column("Question", max_width=40)
    table.add_column("Answer", style="green")
    table.add_column("Status")
    table.add_column("Difficulty")
    table.add_column("AI")

    for t in tasks:
        status_style = {"draft": "yellow", "submitted": "blue", "approved": "green", "rework": "red"}.get(t.status, "white")
        table.add_row(
            str(t.id),
            (t.title[:30] + "...") if len(t.title) > 30 else t.title,
            t.question[:40] + ("..." if len(t.question) > 40 else ""),
            t.answer[:20],
            f"[{status_style}]{t.status}[/]",
            t.difficulty or "-",
            "✓" if t.ai_generated else "",
        )

    console.print(table)


@task_app.command("difficulty")
def set_diff(
    task_id: int = typer.Argument(...),
    difficulty: str = typer.Option(..., help="easy, challenging, or hardest"),
    qwen: int = typer.Option(0, "--qwen", help="Qwen pass count (0-4)"),
    gemini: int = typer.Option(0, "--gemini", help="Gemini pass count (0-4)"),
):
    """Set difficulty and model pass counts for a task."""
    from .tracking import set_difficulty

    task = set_difficulty(task_id, difficulty, qwen, gemini)
    if task:
        console.print(f"[green]Task #{task_id} → {difficulty} (Qwen: {qwen}/4, Gemini: {gemini}/4)[/]")
    else:
        console.print(f"[red]Task {task_id} not found.[/]")


@task_app.command("export")
def export_task_cmd(
    task_id: int = typer.Argument(...),
):
    """Export a task for the platform (copy-paste ready)."""
    from .tracking import export_task

    data = export_task(task_id)
    if not data:
        console.print(f"[red]Task {task_id} not found.[/]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Title:[/] {data['title']}\n"
        f"[bold]Domain:[/] {data['domain']}\n\n"
        f"[bold]Question:[/]\n{data['question']}\n\n"
        f"[bold]Answer:[/] [green]{data['answer']}[/]\n\n"
        f"[bold]Format:[/] {data['answer_format']}\n"
        f"[bold]Type:[/] {data['task_type']}\n"
        f"[bold]Image:[/] {data['image_path']}\n"
        f"[bold]Difficulty:[/] {data['difficulty'] or 'not set'}",
        title=f"Task #{task_id} Export",
    ))


@task_app.command("stats")
def stats():
    """Show progress statistics."""
    from .tracking import get_stats

    s = get_stats()

    table = Table(title="Progress Dashboard")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Tasks", str(s["total_tasks"]))
    for status, count in s["by_status"].items():
        table.add_row(f"  {status}", str(count))
    table.add_row("", "")
    table.add_row("By Difficulty", "")
    for diff, count in s["by_difficulty"].items():
        table.add_row(f"  {diff}", str(count))
    table.add_row("", "")
    table.add_row("Total Figures", str(s["total_figures"]))
    table.add_row("Used Figures", str(s["used_figures"]))

    console.print(table)


@task_app.command("submit")
def submit_task(
    task_id: int = typer.Argument(...),
    platform_id: str = typer.Option("", "--platform-id", help="Platform task ID"),
):
    """Mark a task as submitted."""
    from .tracking import mark_submitted

    task = mark_submitted(task_id, platform_id)
    if task:
        console.print(f"[green]Task #{task_id} marked as submitted.[/]")
    else:
        console.print(f"[red]Task {task_id} not found.[/]")


@task_app.command("new-batch")
def create_task_batch(
    count: int = typer.Option(5, "--count", "-n", help="Number of tasks to create"),
    min_complexity: float = typer.Option(0.5, "--min-complexity", help="Minimum complexity threshold"),
    domain: str = typer.Option("Computer Science", "--domain", "-d", help="Domain for the tasks"),
    task_type: str = typer.Option("chart", "--task-type", help="chart, general_image, or spatial"),
    challenging: bool = typer.Option(False, "--challenging", help="Use CHALLENGING prompt"),
    hardest: bool = typer.Option(False, "--hardest", help="Use HARDEST prompt"),
    draft_attempts: int = typer.Option(1, "--draft-attempts", help="Number of independent draft attempts (1=fast, 3=best)"),
    model: str = typer.Option("minimax-m3", "--model", help="Model ID (minimax-m3, kimi-k2.7-code)"),
    auto: bool = typer.Option(False, "--auto", help="Auto-confirm all (non-interactive)"),
):
    """Draft and create tasks for the top N suitable figures.

    Selects the best candidate figures (by complexity * density), drafts
    Q&A via minimax-m3, validates, and creates tasks — all non-interactive
    when --auto is set.
    """
    from sqlmodel import select
    from .db import get_session
    from .models import Figure, TaskStatus
    from .authoring import create_task
    from .authoring.validator import validate_task
    from .authoring.ai_draft import draft_qa, draft_qa_consensus
    import os

    if hardest and challenging:
        console.print("[red]Cannot specify both --hardest and --challenging.[/]")
        raise typer.Exit(1)

    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        console.print("[red]No OPENCODE_API_KEY set.[/]")
        raise typer.Exit(1)

    session = get_session()

    # Find top candidates
    query = (
        select(Figure)
        .where(Figure.complexity_score >= min_complexity)
        .where(Figure.is_suitable == True)  # noqa: E712
    )
    candidates = list(session.exec(query).all())
    # Rank: complexity * (1.3 if dense) * (1.1 if chart_graph_text)
    def score(f):
        s = f.complexity_score
        if f.is_dense:
            s *= 1.3
        if f.figure_type == "chart_graph_text":
            s *= 1.1
        return s
    candidates.sort(key=score, reverse=True)
    candidates = candidates[:count]
    submitted = 0

    if not candidates:
        console.print(f"[yellow]No suitable figures found at complexity ≥ {min_complexity}. Try running 'arxiv-manager images reclassify' first.[/]")
        raise typer.Exit(0)

    difficulty = "hardest" if hardest else ("challenging" if challenging else "")
    console.print(f"Drafting {len(candidates)} tasks (difficulty: {difficulty or 'manual'})...\n")

    for fig in candidates:
        console.print(f"[bold]Figure #{fig.id}[/] ({fig.figure_type}, density={fig.is_dense}, complexity={fig.complexity_score:.2f})")
        console.print(f"  Path: {fig.image_path}")

        if draft_attempts > 1:
            draft = draft_qa_consensus(
                fig.full_path, n_attempts=draft_attempts, verify=True,
                caption=fig.caption, provider="opencode",
                api_key=api_key, difficulty=difficulty or "",
                figure_type=fig.figure_type, complexity_score=fig.complexity_score,
                model=model,
            )
        else:
            draft = draft_qa(
                fig.full_path,
                caption=fig.caption,
                provider="opencode",
                api_key=api_key,
                difficulty=difficulty or "",
                figure_type=fig.figure_type,
                complexity_score=fig.complexity_score,
                model=model,
            )

        if not draft:
            console.print(f"  [red]Draft failed; skipping.[/]")
            continue

        question = draft["question"]
        answer = draft["answer"]
        answer_format = draft.get("answer_format", "word")
        task_type_val = draft.get("task_type", task_type)
        title = fig.figure_num or fig.caption[:60].strip() or question[:60].strip()

        validation = validate_task(
            question, answer, answer_format,
            figure_type=fig.figure_type, task_type=task_type_val,
        )

        if not validation.is_valid:
            if auto:
                console.print(f"  [yellow]Validation errors; skipping: {validation.errors[:2]}[/]")
                continue
            console.print(f"  [yellow]Validation errors:[/]")
            for e in validation.errors:
                console.print(f"    ❌ {e}")
            for w in validation.warnings[:2]:
                console.print(f"    ⚠️  {w}")
            if not Confirm.ask("  Save anyway?"):
                continue

        task = create_task(
            figure_id=fig.id,
            title=title,
            domain=domain,
            question=question,
            answer=answer,
            answer_format=answer_format,
            task_type=task_type_val,
            ai_generated=True,
        )
        fig.status = "used"
        session.add(fig)
        submitted += 1
        console.print(f"  [green]✓ Task #{task.id} created[/]")
        console.print(f"    Q: {question[:80]}")
        console.print(f"    A: {answer} ({answer_format})")
        console.print()

    session.commit()
    console.print(f"[bold green]Done. {submitted}/{len(candidates)} tasks created.[/]")


# ─── WEB SERVER ───────────────────────────────────────────────────

@app.command("web")
def web_server(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(7860, "--port"),
):
    """Start the web dashboard."""
    import uvicorn
    from .web.app import create_app

    app_instance = create_app()
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":
    app()
