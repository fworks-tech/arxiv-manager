"""CLI interface using Typer + Rich."""

from __future__ import annotations

import typer
from rich.console import Console

from ..db import init_db

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


# Import sub-modules to register commands (eager — Typer requires this)
from . import (  # noqa: E402
    admin_commands,  # noqa: F401
    check,  # noqa: F811, F401
    image_commands,  # noqa: F401
    index_commands,  # noqa: F401
    search_commands,  # noqa: F401
    task_commands,  # noqa: F401
)
