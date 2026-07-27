"""Admin and analytics CLI commands."""

import typer
from rich.panel import Panel
from rich.table import Table
from sqlmodel import desc, func, select

from ..db import get_session
from ..models import GenerationAttempt
from ..tracking import export_task, get_stats, mark_submitted, set_difficulty
from . import app, console, task_app


@task_app.command("difficulty")
def set_diff(
    task_id: int = typer.Argument(...),
    difficulty: str = typer.Option(..., help="easy, challenging, or hardest"),
    qwen: int = typer.Option(0, "--qwen", help="Qwen pass count (0-4)"),
    gemini: int = typer.Option(0, "--gemini", help="Gemini pass count (0-4)"),
):
    """Set difficulty and model pass counts for a task."""
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
def stats_cmd():
    """Show progress statistics."""
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
def submit_task_cmd(
    task_id: int = typer.Argument(...),
    platform_id: str = typer.Option("", "--platform-id", help="Platform task ID"),
):
    """Mark a task as submitted."""
    task = mark_submitted(task_id, platform_id)
    if task:
        console.print(f"[green]Task #{task_id} marked as submitted.[/]")
    else:
        console.print(f"[red]Task {task_id} not found.[/]")


@task_app.command("analytics")
def analytics_cmd(
    limit: int = typer.Option(50, "--limit", "-l", help="Recent attempts to analyze"),
):
    """Analyze generation attempt history for quality insights."""
    import json

    session = get_session()

    total = session.exec(select(func.count(GenerationAttempt.id))).one()
    if total == 0:
        console.print("[yellow]No generation attempts recorded yet. Generate some Q&A first.[/]")
        return

    recent = list(
        session.exec(
            select(GenerationAttempt)
            .order_by(desc(GenerationAttempt.created_at))
            .limit(limit)
        ).all()
    )

    console.print(f"\n[bold]Generation Analytics (last {len(recent)} of {total} attempts)[/]\n")

    # By difficulty
    diff_table = Table(title="Success Rate by Difficulty")
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column("Attempts", justify="right")
    diff_table.add_column("Success", justify="right")
    diff_table.add_column("Avg Quality", justify="right")
    diff_by_diff: dict[str, list[float]] = {}
    for r in recent:
        d = r.difficulty or "unknown"
        diff_by_diff.setdefault(d, []).append(1.0 if r.success else 0.0)
    for d, scores in sorted(diff_by_diff.items()):
        ok = sum(scores)
        total_d = len(scores)
        qualities = [r.validation_quality for r in recent if r.difficulty == d and r.validation_quality > 0]
        avg_q = sum(qualities) / len(qualities) if qualities else 0
        diff_table.add_row(d, str(total_d), f"{ok:.0f}/{total_d}", f"{avg_q:.1f}")
    console.print(diff_table)

    # By model
    model_table = Table(title="Performance by Model")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("Attempts", justify="right")
    model_table.add_column("Avg Quality", justify="right")
    by_model: dict[str, list[float]] = {}
    for r in recent:
        m = r.model_name or "unknown"
        by_model.setdefault(m, [])
        if r.validation_quality > 0:
            by_model[m].append(r.validation_quality)
    for m, qs in sorted(by_model.items()):
        avg_q = sum(qs) / len(qs) if qs else 0
        model_table.add_row(m, str(len(qs)), f"{avg_q:.1f}")
    console.print(model_table)

    # By figure_type
    ft_table = Table(title="Success Rate by Figure Type")
    ft_table.add_column("Figure Type", style="cyan")
    ft_table.add_column("Attempts", justify="right")
    ft_table.add_column("Avg Quality", justify="right")
    by_ft: dict[str, list[float]] = {}
    for r in recent:
        ft = r.figure_type or "unknown"
        by_ft.setdefault(ft, [])
        if r.validation_quality > 0:
            by_ft[ft].append(r.validation_quality)
    for ft, qs in sorted(by_ft.items()):
        avg_q = sum(qs) / len(qs) if qs else 0
        ft_table.add_row(ft, str(len(qs)), f"{avg_q:.1f}")
    console.print(ft_table)

    # Most common validation errors
    error_counts: dict[str, int] = {}
    for r in recent:
        if r.validation_errors and r.validation_errors != "[]":
            try:
                errs = json.loads(r.validation_errors)
                for e in errs:
                    error_counts[e] = error_counts.get(e, 0) + 1
            except json.JSONDecodeError:
                pass
    if error_counts:
        err_table = Table(title="Most Common Validation Errors")
        err_table.add_column("Error", style="red")
        err_table.add_column("Count", justify="right")
        for e, cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
            err_table.add_row(e[:80], str(cnt))
        console.print(err_table)

    # Best config recommendation
    config_scores: dict[tuple[str, str, str], list[float]] = {}
    for r in recent:
        if r.validation_quality > 0:
            key = (r.figure_type or "any", r.difficulty or "any", r.model_name or "any")
            config_scores.setdefault(key, []).append(r.validation_quality)
    if config_scores:
        best = max(config_scores.items(), key=lambda x: sum(x[1]) / len(x[1]))
        (ft, diff, model), qs = best
        console.print(f"\n[bold green]Best config:[/] figure_type={ft}, difficulty={diff}, model={model} "
                      f"(avg quality={sum(qs)/len(qs):.1f}, n={len(qs)})")

    # Low-quality generation examples
    low_quality = [r for r in recent if 0 < r.validation_quality < 60]
    if low_quality:
        console.print(f"\n[yellow]Low-quality generations ({len(low_quality)}):[/]")
        for r in low_quality[:3]:
            console.print(f"  Attempt #{r.id}: quality={r.validation_quality:.0f}, "
                         f"type={r.generation_type}, difficulty={r.difficulty}")
            if r.generated_question:
                console.print(f"    Q: {r.generated_question[:80]}")
            if r.validation_errors and r.validation_errors != "[]":
                try:
                    errs = json.loads(r.validation_errors)
                    for e in errs[:2]:
                        console.print(f"    ❌ {e}")
                except json.JSONDecodeError:
                    pass

    session.close()


@app.command("web")
def web_server(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(7860, "--port"),
):
    """Start the web dashboard."""
    import uvicorn

    from ..web.app import create_app

    app_instance = create_app()
    uvicorn.run(app_instance, host=host, port=port)
