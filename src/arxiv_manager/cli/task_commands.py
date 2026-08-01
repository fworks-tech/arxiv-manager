"""Task creation and management commands."""

import typer
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sqlmodel import select

from ..authoring import create_task, list_tasks, update_task
from ..authoring.ai_draft import draft_qa, draft_qa_consensus
from ..authoring.validator import validate_task
from ..db import get_session
from ..models import Figure, Task
from . import console, task_app


@task_app.command("new")
def new_task(
    image_id: int = typer.Option(..., "--image-id", "-i", help="Figure ID from the library"),
    ai: bool = typer.Option(False, "--ai", help="Use AI to draft Q&A"),
    hardest: bool = typer.Option(False, "--hardest", help="Generate HARDEST-classified question"),
    challenging: bool = typer.Option(False, "--challenging", help="Generate CHALLENGING-classified question"),
    draft_attempts: int = typer.Option(1, "--draft-attempts", help="Number of draft attempts (1=fast, 3=best)"),
    model: str = typer.Option("minimax-m3", "--model", help="Model ID"),
    title: str = typer.Option("", "--title", "-t", help="Task title"),
):
    """Create a new task for an image."""
    import os

    if hardest and challenging:
        console.print("[red]Cannot specify both --hardest and --challenging.[/]")
        raise typer.Exit(1)

    session = get_session()
    figure = session.get(Figure, image_id)
    if not figure:
        console.print(f"[red]Image {image_id} not found.[/]")
        raise typer.Exit(1)

    diff_label = "HARDEST" if hardest else ("CHALLENGING" if challenging else "")
    console.print(
        Panel(
            f"Image: {figure.image_path}\nCaption: {figure.caption}\n"
            f"Complexity: {figure.complexity_score:.3f}\n"
            f"Type: {figure.figure_type or 'unknown'}\n"
            f"Dense: {figure.is_dense}\n"
            f"Target: {diff_label or 'manual'}",
            title="Task Source",
        )
    )

    draft = None
    if ai or hardest or challenging:
        console.print("[bold]Generating AI draft...[/]")
        difficulty = "hardest" if hardest else ("challenging" if challenging else "")
        api_key = os.environ.get("OPENCODE_API_KEY")
        if draft_attempts > 1:
            draft = draft_qa_consensus(
                figure.full_path,
                n_attempts=draft_attempts,
                verify=True,
                caption=figure.caption,
                api_key=api_key,
                difficulty=difficulty,
                figure_type=figure.figure_type,
                complexity_score=figure.complexity_score,
                model=model,
            )
        else:
            draft = draft_qa(
                figure.full_path,
                caption=figure.caption,
                api_key=api_key,
                difficulty=difficulty,
                figure_type=figure.figure_type,
                complexity_score=figure.complexity_score,
                model=model,
                figure_id=figure.id,
            )
        if draft:
            console.print(f"[green]AI drafted:[/]\n  Q: {draft['question']}\n  A: {draft['answer']}")
            if not Confirm.ask("Use this draft?"):
                draft = None
        else:
            console.print("[yellow]AI draft failed. Falling back to manual entry.[/]")

    domain = Prompt.ask("Domain", default="Computer Science")
    if not ai or not draft:
        if not title:
            title = Prompt.ask("Task title", default=figure.caption[:60].strip() if figure.caption else "")
        question = Prompt.ask("Question")
        answer = Prompt.ask("Answer")
        answer_format = Prompt.ask(
            "Answer format", choices=["number", "word", "phrase", "year", "percent", "integer"], default="word"
        )
        task_type = Prompt.ask("Task type", choices=["chart", "general_image", "spatial"], default="chart")
    else:
        question = draft["question"]
        answer = draft["answer"]
        answer_format = draft.get("answer_format", "word")
        task_type = draft.get("task_type", "chart")
        if not title:
            title = figure.caption[:60].strip() if figure.caption else question[:60].strip()

    validation = validate_task(question, answer, answer_format)
    console.print(f"\n[bold]Validation:[/]\n{validation.summary()}")

    if not validation.is_valid:
        if not Confirm.ask("\nSave anyway despite errors?"):
            raise typer.Exit(0)

    task = create_task(
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
def validate_existing_cmd(
    task_id: int = typer.Argument(..., help="Task ID to validate"),
    no_regen: bool = typer.Option(False, "--no-regen", help="Don't auto-regenerate on errors"),
    auto: bool = typer.Option(False, "--auto", help="Auto-regenerate without confirmation"),
    hardest: bool = typer.Option(False, "--hardest", help="Use HARDEST prompt"),
    challenging: bool = typer.Option(False, "--challenging", help="Use CHALLENGING prompt"),
    model: str = typer.Option("minimax-m3", "--model", help="Model ID"),
):
    """Validate an existing task against handbook rules."""
    import os

    session = get_session()
    task = session.get(Task, task_id)
    if not task:
        console.print(f"[red]Task {task_id} not found.[/]")
        raise typer.Exit(1)

    figure = session.get(Figure, task.figure_id)
    validation = validate_task(task.question, task.answer, task.answer_format, figure.image_path if figure else "")

    console.print(f"[bold]Task #{task_id} Validation:[/]\n{validation.summary()}")

    if not validation.is_valid and not no_regen:
        errors = "; ".join(validation.errors)
        warnings = "; ".join(validation.warnings)
        feedback = f"Errors: {errors}"
        if warnings:
            feedback += f"\nWarnings: {warnings}"

        console.print("\n[yellow]Validation errors found:[/]")
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
            api_key=api_key,
            feedback=feedback,
            difficulty="hardest" if hardest else ("challenging" if challenging else ""),
            figure_type=figure.figure_type,
            complexity_score=figure.complexity_score,
            model=model,
            figure_id=figure.id,
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
                validation = validate_task(
                    task.question, task.answer, task.answer_format, figure.image_path if figure else ""
                )
                console.print(f"\n[bold]Re-validation after update:[/]\n{validation.summary()}")
            else:
                console.print("[dim]Keeping original draft.[/]")
        else:
            console.print("[red]AI regeneration failed — keeping original.[/]")


@task_app.command("list")
def list_tasks_cmd(
    status: str = typer.Option("", "--status", "-s"),
    limit: int = typer.Option(30, "--limit", "-l"),
):
    """List tasks."""
    tasks = list_tasks(status=status or None, limit=limit)
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
        status_style = {"draft": "yellow", "submitted": "blue", "approved": "green", "rework": "red"}.get(
            t.status, "white"
        )
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


@task_app.command("new-batch")
def create_task_batch(
    count: int = typer.Option(5, "--count", "-n", help="Number of tasks to create"),
    min_complexity: float = typer.Option(0.5, "--min-complexity", help="Minimum complexity threshold"),
    domain: str = typer.Option("Computer Science", "--domain", "-d", help="Domain"),
    task_type: str = typer.Option("chart", "--task-type", help="chart, general_image, or spatial"),
    challenging: bool = typer.Option(False, "--challenging", help="Use CHALLENGING prompt"),
    hardest: bool = typer.Option(False, "--hardest", help="Use HARDEST prompt"),
    draft_attempts: int = typer.Option(1, "--draft-attempts", help="Number of independent draft attempts"),
    model: str = typer.Option("minimax-m3", "--model", help="Model ID"),
    auto: bool = typer.Option(False, "--auto", help="Auto-confirm all (non-interactive)"),
):
    """Draft and create tasks for the top N suitable figures."""
    import os

    if hardest and challenging:
        console.print("[red]Cannot specify both --hardest and --challenging.[/]")
        raise typer.Exit(1)

    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        console.print("[red]No OPENCODE_API_KEY set.[/]")
        raise typer.Exit(1)

    session = get_session()

    query = (
        select(Figure).where(Figure.complexity_score >= min_complexity).where(Figure.is_suitable == True)  # noqa: E712
    )
    candidates = list(session.exec(query).all())

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
        console.print(f"[yellow]No suitable figures found at complexity >= {min_complexity}.[/]")
        raise typer.Exit(0)

    difficulty = "hardest" if hardest else ("challenging" if challenging else "")
    console.print(f"Drafting {len(candidates)} tasks (difficulty: {difficulty or 'manual'})...\n")

    for fig in candidates:
        console.print(
            f"[bold]Figure #{fig.id}[/] ({fig.figure_type}, density={fig.is_dense}, complexity={fig.complexity_score:.2f})"
        )
        console.print(f"  Path: {fig.image_path}")

        if draft_attempts > 1:
            draft = draft_qa_consensus(
                fig.full_path,
                n_attempts=draft_attempts,
                verify=True,
                caption=fig.caption,
                api_key=api_key,
                difficulty=difficulty or "",
                figure_type=fig.figure_type,
                complexity_score=fig.complexity_score,
                model=model,
            )
        else:
            draft = draft_qa(
                fig.full_path,
                caption=fig.caption,
                api_key=api_key,
                difficulty=difficulty or "",
                figure_type=fig.figure_type,
                complexity_score=fig.complexity_score,
                model=model,
                figure_id=fig.id,
            )

        if not draft:
            console.print("  [red]Draft failed; skipping.[/]")
            continue

        question = draft["question"]
        answer = draft["answer"]
        answer_format = draft.get("answer_format", "word")
        task_type_val = draft.get("task_type", task_type)
        title = fig.figure_num or fig.caption[:60].strip() or question[:60].strip()

        validation = validate_task(
            question, answer, answer_format, figure_type=fig.figure_type, task_type=task_type_val
        )

        if not validation.is_valid:
            if auto:
                console.print(f"  [yellow]Validation errors; skipping: {validation.errors[:2]}[/]")
                continue
            console.print("  [yellow]Validation errors:[/]")
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


@task_app.command("determinism")
def determinism_check_cmd(
    task_id: int = typer.Argument(..., help="Task ID to check"),
    runs: int = typer.Option(3, "--runs", "-r", help="Number of sampled reads (max 5)"),
):
    """Run the question through the vision model N times and verify every read matches the golden answer.

    Proves the answer is objectively derivable from the image — the strongest
    available signal that independent readers give the same answer.
    """
    import os

    from ..authoring.ai_draft._determinism import check_determinism_for_qa
    from ..storage import STORAGE_DIR

    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        console.print("[red]No OPENCODE_API_KEY set.[/]")
        raise typer.Exit(1)

    session = get_session()
    try:
        task = session.get(Task, task_id)
        if not task:
            console.print(f"[red]Task {task_id} not found.[/]")
            raise typer.Exit(1)

        img_path = STORAGE_DIR / task.image_path
        if not img_path.exists():
            console.print("[red]Image not found.[/]")
            raise typer.Exit(1)

        result = check_determinism_for_qa(
            task.question,
            task.answer,
            task.answer_format,
            img_path,
            api_key,
            runs=max(1, min(int(runs), 5)),
            difficulty=task.difficulty or "challenging",
        )

        console.print(f"\n[bold]Task #{task_id} Determinism Check[/] (golden: [green]{task.answer}[/])")
        for i, r in enumerate(result["runs"], 1):
            mark = "[green]✓ match[/]" if r["match"] else "[red]✗ diverge[/]"
            console.print(f"  Read {i}: {r['answer'] or '(no answer)'} {mark}")
        if result["deterministic"]:
            console.print("\n[bold green]✅ Deterministic — all reads match the golden answer.[/]")
        else:
            console.print(f"\n[bold yellow]🎲 Not deterministic.[/] Diverging: {result['diverging'] or '(no answers)'}")
    finally:
        session.close()
