"""Image library management commands."""

import typer
from rich.table import Table
from sqlmodel import select

from ..db import get_session
from ..models import Figure
from ..sourcing.filters import audit_figure, compute_complexity
from . import console, images_app


@images_app.command("list")
def list_images(
    status: str = typer.Option("", "--status", "-s", help="Filter by status: new, reviewed, used, rejected"),
    min_complexity: float = typer.Option(0, "--min-complexity", help="Min complexity score"),
    figure_type: str = typer.Option("", "--type", help="Filter by figure type: chart_graph_text | general_image"),
    suitable_only: bool = typer.Option(False, "--suitable", help="Show only is_suitable figures"),
    limit: int = typer.Option(30, "--limit", "-l"),
):
    """List images in the library."""
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
            str(f.id), f.image_path, type_short,
            f"{f.complexity_score:.3f}", "✓" if f.is_dense else "",
            f"{f.width}x{f.height}", f"{kb:.0f}",
            f"[{status_style}]{f.status}[/]",
            (f.caption[:40] + "...") if len(f.caption) > 40 else f.caption,
        )
    console.print(table)


@images_app.command("audit")
def audit_images_cmd():
    """Print image-library health report."""
    import glob
    import os
    import sqlite3

    session = get_session()
    conn = sqlite3.connect("storage/arxiv-manager.db")
    c = conn.cursor()

    disk_files = [os.path.basename(f) for f in glob.glob("storage/figures/*.png")]
    db_files = [os.path.basename(f.image_path) for f in session.exec(select(Figure)).all()]
    c.execute("SELECT image_path FROM figures")
    db_paths = [r[0] for r in c.fetchall()]
    broken = [p for p in db_paths if not os.path.exists(f"storage/{p}")]
    orphans = sorted(set(disk_files) - set(db_files))
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
    c.execute("SELECT complexity_score FROM figures")
    scores = [r[0] for r in c.fetchall() if r[0] is not None]
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    dist = {f"[{lo:.1f},{hi:.2f})": sum(1 for s in scores if lo <= s < hi) for lo, hi in buckets}
    c.execute("SELECT status, COUNT(*) FROM figures GROUP BY status")
    by_status = dict(c.fetchall())
    c.execute("SELECT figure_type, COUNT(*) FROM figures WHERE figure_type != '' GROUP BY figure_type")
    by_type = dict(c.fetchall())
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
    import os
    import shutil
    import sqlite3
    from datetime import datetime

    from rich.prompt import Confirm

    if not yes and not Confirm.ask("This will delete trash + broken rows and move orphans. Continue?"):
        raise typer.Exit(0)

    conn = sqlite3.connect("storage/arxiv-manager.db")
    c = conn.cursor()

    if not no_backup:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"storage/arxiv-manager.db.clean-{ts}"
        shutil.copy2("storage/arxiv-manager.db", backup)
        console.print(f"[green]Backed up DB to {backup}[/]")

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

    c.execute("SELECT image_path FROM figures")
    db_paths = [r[0] for r in c.fetchall()]
    broken_removed = 0
    for p in db_paths:
        if not os.path.exists(f"storage/{p}"):
            c.execute("DELETE FROM figures WHERE image_path = ?", (p,))
            broken_removed += 1
    console.print(f"Removed {broken_removed} broken DB rows")

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

    import hashlib
    c.execute("SELECT image_path, COUNT(*) as cnt FROM figures GROUP BY image_path HAVING cnt > 1")
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
    """Run classify_figure_type and audit_figure on all figures."""
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
    """Re-run improved compute_complexity on all figures."""
    session = get_session()
    query = select(Figure)
    if limit > 0:
        query = query.limit(limit)
    figures = list(session.exec(query).all())
    console.print(f"Rescoring {len(figures)} figures...")

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
    """Show top candidates for Challenging tasks."""
    session = get_session()
    query = (
        select(Figure)
        .where(Figure.complexity_score >= min_complexity)
        .where(Figure.is_suitable == True)  # noqa: E712
    )
    figures = list(session.exec(query).all())
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
        console.print("[yellow]No candidates found at this threshold.[/]")
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
            str(f.id), f"{score(f):.3f}",
            {"chart_graph_text": "chart", "general_image": "img"}.get(f.figure_type, "-"),
            f"{f.complexity_score:.3f}", "✓" if f.is_dense else "",
            f"{f.width}x{f.height}", f.image_path,
        )
    console.print(table)
