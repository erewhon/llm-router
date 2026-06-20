"""CLI for code ingestion and search."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress

from llm_router.rag.chunker import chunk_file, detect_lang
from llm_router.rag.docparse import chunk_document, is_supported
from llm_router.rag.embed import qwen3_embedder
from llm_router.rag.store import DOC_COLLECTION, QdrantStore

DOC_QUERY_TASK = "Given a query, retrieve relevant document passages that answer it"

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


def _iter_docs(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and is_supported(path):
            yield path


@cli.command("ingest-docs")
@click.argument("doc_path", type=click.Path(exists=True, path_type=Path))
@click.option("--topics", default="", help="Comma-separated topic tags (e.g. politics,astronomy).")
@click.option("--batch-size", default=16, show_default=True, help="Embed batch size.")
def ingest_docs(doc_path: Path, topics: str, batch_size: int) -> None:
    """Index documents (PDF/DOCX/PPTX/HTML/MD/TXT) into the Qdrant `documents` collection."""
    target = doc_path.resolve()
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    if target.is_file():
        root, files = target.parent, [target]
    else:
        root, files = target, list(_iter_docs(target))
    console.print(f"[bold]Scanning[/bold] {len(files)} document(s) in {target}")

    all_chunks = []
    for f in files:
        try:
            all_chunks.extend(chunk_document(root, f, topic_list))
        except Exception as exc:  # keep going past a bad file
            console.print(f"[yellow]skip {f.name}: {exc}[/yellow]")
    console.print(f"[bold]Produced[/bold] {len(all_chunks)} chunks")
    if not all_chunks:
        return

    embedder = qwen3_embedder(query_task=DOC_QUERY_TASK)
    store = QdrantStore(collection=DOC_COLLECTION)
    with Progress() as progress:
        task = progress.add_task("[cyan]Embed + upsert", total=len(all_chunks))
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            vectors = embedder.embed([c.text for c in batch])
            if i == 0:
                store.ensure_collection(len(vectors[0]))
            store.upsert_docs(batch, vectors)
            progress.update(task, advance=len(batch))
    embedder.close()
    tag_note = f" tagged {topic_list}" if topic_list else ""
    console.print(f"[green]Done. Indexed {len(all_chunks)} doc chunks{tag_note}.[/green]")


@cli.command("search-docs")
@click.argument("query", type=str)
@click.option("--top-k", default=10, show_default=True)
@click.option("--topic", "topics", multiple=True, help="Filter by topic tag (repeatable).")
def search_docs(query: str, top_k: int, topics: tuple[str, ...]) -> None:
    """Search the documents collection."""
    embedder = qwen3_embedder(query_task=DOC_QUERY_TASK)
    store = QdrantStore(collection=DOC_COLLECTION)
    qv = embedder.embed([query], is_query=True)[0]
    results = store.search_docs(qv, top_k=top_k, topics=list(topics) or None)
    for r in results:
        page = f" p{r['page']}" if r.get("page") is not None else ""
        tags = ",".join(r.get("topics", []))
        tag_str = f" [{tags}]" if tags else ""
        header = (
            f"[bold]{r['source']}#{r['chunk_index']}{page}{tag_str} "
            f"score={r['score']:.3f}[/bold]  {r.get('title', '')}"
        )
        console.rule(header)
        text = r.get("text", "")
        if len(text) > 800:
            text = text[:800] + "\n... (truncated)"
        console.print(text, markup=False, highlight=False)
    embedder.close()
