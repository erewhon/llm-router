"""CLI for code ingestion and search."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from llm_router.rag.chunker import chunk_file, detect_lang
from llm_router.rag.embed import qwen3_embedder
from llm_router.rag.store import QdrantStore

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "target",
    ".idea",
    ".vscode",
}

console = Console()


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if detect_lang(path) is None:
            continue
        yield path


@click.group()
def cli() -> None:
    """RAG indexing and search."""


@cli.command("ingest-code")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--repo-name", default=None, help="Override repo name (defaults to dir name).")
@click.option("--batch-size", default=32, show_default=True, help="Embed batch size.")
def ingest_code(repo_path: Path, repo_name: str | None, batch_size: int) -> None:
    """Index a code repo into the Qdrant `code` collection."""
    root = repo_path.resolve()
    repo = repo_name or root.name
    embedder = qwen3_embedder()
    store = QdrantStore()

    files = list(_iter_files(root))
    console.print(f"[bold]Scanning[/bold] {len(files)} files in {root}")

    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(repo, root, f))
    console.print(f"[bold]Produced[/bold] {len(all_chunks)} chunks")

    if not all_chunks:
        embedder.close()
        return

    with Progress() as progress:
        task = progress.add_task("[cyan]Embed + upsert", total=len(all_chunks))
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            vectors = embedder.embed([c.text for c in batch])
            store.upsert_code(batch, vectors)
            progress.update(task, advance=len(batch))

    embedder.close()
    console.print(f"[green]Done. Indexed {len(all_chunks)} chunks for repo '{repo}'.[/green]")


@cli.command("search-code")
@click.argument("query", type=str)
@click.option("--top-k", default=10, show_default=True)
@click.option("--repo", default=None, help="Filter by repo name.")
def search_code(query: str, top_k: int, repo: str | None) -> None:
    """Search the code collection."""
    embedder = qwen3_embedder()
    store = QdrantStore()
    qv = embedder.embed([query], is_query=True)[0]
    results = store.search_code(qv, top_k=top_k, repo=repo)
    for r in results:
        header = (
            f"[bold]{r['repo']}/{r['path']}:{r['start_line']}-{r['end_line']} "
            f"({r['chunk_kind']} {r['name']}) score={r['score']:.3f}[/bold]"
        )
        console.rule(header)
        text = r.get("text", "")
        if len(text) > 800:
            text = text[:800] + "\n... (truncated)"
        console.print(text, markup=False, highlight=False)
    embedder.close()
