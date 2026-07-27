"""Pre-flight health check command."""
from . import app, console


@app.command("check")
def check_api():
    """Verify API connectivity, model response, and DB health."""
    import os
    import time

    import httpx
    from sqlmodel import select

    from ..db import get_session
    from ..models import Figure, Paper, Task

    console.print("[bold]Pre-flight checks[/]\n")

    console.print("1. API key...", end=" ")
    key = os.environ.get("OPENCODE_API_KEY")
    if not key:
        console.print("[red]✗ Missing OPENCODE_API_KEY[/]")
    else:
        console.print(f"[green]✓ Found ({key[:8]}...{key[-4:]})[/]")

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
            console.print("[red]✗ Empty response[/]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/]")

    console.print("3. Database...", end=" ")
    try:
        session = get_session()
        fig_count = len(list(session.exec(select(Figure)).all()))
        paper_count = len(list(session.exec(select(Paper)).all()))
        task_count = len(list(session.exec(select(Task)).all()))
        console.print(f"[green]✓ Connected — {paper_count} papers, {fig_count} figures, {task_count} tasks[/]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/]")

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
        import re
        stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "OK" in stripped:
            console.print(f"[green]✓ Model responds (content={len(content)}c)[/]")
        else:
            console.print("[yellow]⚠ Unexpected response[/]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/]")

    console.print("\n[bold green]All checks complete.[/]")
