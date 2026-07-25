"""CLI commands for indexing figures into the vector store."""

from __future__ import annotations

import logging

from ..db import get_session
from ..models import Figure, Paper
from . import images_app

logger = logging.getLogger(__name__)


@images_app.command("index")
def index_figures(
    limit: int = 0,
    figure_type: str = "",
):
    """Batch-embed figures into ChromaDB vector store for RAG.

    Processes figures from the database, indexes their captions,
    paper titles, and metadata into the vector store for semantic search.
    """
    from rich import print as rprint
    from ..components.hybrid_retriever import HybridRetriever

    session = get_session()
    try:
        query = session.query(Figure).filter(Figure.is_suitable == True)
        if figure_type:
            query = query.filter(Figure.figure_type == figure_type)
        if limit > 0:
            query = query.limit(limit)
        figures = query.all()

        if not figures:
            rprint("[yellow]No suitable figures found to index.[/]")
            return

        retriever = HybridRetriever()
        indexed = 0
        errors = 0

        for fig in figures:
            paper = session.get(Paper, fig.paper_id)
            paper_title = paper.title if paper else ""

            try:
                retriever.add_figure(
                    figure_id=fig.id,
                    caption=fig.caption,
                    figure_type=fig.figure_type,
                    paper_title=paper_title,
                    difficulty="",
                    metadata={
                        "paper_id": fig.paper_id,
                        "page_num": fig.page_num,
                        "complexity": fig.complexity_score,
                        "is_dense": str(fig.is_dense),
                    },
                )
                indexed += 1
            except Exception as e:
                logger.warning("index: failed for figure %d: %s", fig.id, e)
                errors += 1

        rprint(f"[green]Indexed {indexed} figures[/]" +
               (f" ([red]{errors} errors[/])" if errors else ""))

    finally:
        session.close()
