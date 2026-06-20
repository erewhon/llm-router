"""MCP server exposing RAG search over the code + documents collections.

Runs over stdio via the ``llm-router-rag-mcp`` entry point, so Claude Code /
opencode (any MCP client) can search the indexed corpora. Both tools reuse the
same embed -> Qdrant ANN -> (optional) rerank pipeline as the CLI.

OpenArc's reranker wedges under concurrent load, so a module lock serialises
searches — only one rerank request reaches OpenArc at a time.
"""

from __future__ import annotations

import threading

from mcp.server.fastmcp import FastMCP

from llm_router.rag.embed import qwen3_embedder
from llm_router.rag.rerank import rerank_hits
from llm_router.rag.store import DOC_COLLECTION, QdrantStore

DOC_QUERY_TASK = "Given a query, retrieve relevant document passages that answer it"
MIN_RERANK_POOL = 20

mcp = FastMCP("llm-router-rag")
_lock = threading.Lock()


def _embed_query(query: str, query_task: str | None) -> list[float]:
    embedder = qwen3_embedder(query_task=query_task) if query_task else qwen3_embedder()
    try:
        return embedder.embed([query], is_query=True)[0]
    finally:
        embedder.close()


def _format_code(hits: list[dict]) -> str:
    if not hits:
        return "No results."
    out: list[str] = []
    for i, h in enumerate(hits, 1):
        rr = f" rerank={h['rerank_score']:.3f}" if "rerank_score" in h else ""
        out.append(
            f"[{i}] {h['repo']}/{h['path']}:{h['start_line']}-{h['end_line']} "
            f"({h['chunk_kind']} {h['name']}) score={h['score']:.3f}{rr}"
        )
        text = h.get("text", "")
        out.append(text[:1200] + "\n...(truncated)" if len(text) > 1200 else text)
        out.append("")
    return "\n".join(out).rstrip()


def _format_docs(hits: list[dict]) -> str:
    if not hits:
        return "No results."
    out: list[str] = []
    for i, h in enumerate(hits, 1):
        page = f" p{h['page']}" if h.get("page") is not None else ""
        tags = ",".join(h.get("topics", []))
        tag_str = f" [{tags}]" if tags else ""
        rr = f" rerank={h['rerank_score']:.3f}" if "rerank_score" in h else ""
        out.append(
            f"[{i}] {h['source']}#{h['chunk_index']}{page}{tag_str} "
            f"score={h['score']:.3f}{rr}  {h.get('title', '')}"
        )
        text = h.get("text", "")
        out.append(text[:1200] + "\n...(truncated)" if len(text) > 1200 else text)
        out.append("")
    return "\n".join(out).rstrip()


@mcp.tool()
def search_code(
    query: str, top_k: int = 5, repo: str | None = None, rerank: bool = True
) -> str:
    """Semantic search over indexed source code (functions, classes, whole files).

    Use natural-language intent, not keywords ("where do we map a model to a
    backend node", not "model node map"). Returns ranked file:line locations
    with code snippets.

    Args:
        query: Natural-language description of what you're looking for.
        top_k: Number of results to return.
        repo: Optional repo-name filter.
        rerank: Rerank the candidate pool with the cross-encoder (more precise).
    """
    with _lock:
        qv = _embed_query(query, None)
        pool = max(top_k, MIN_RERANK_POOL) if rerank else top_k
        hits = QdrantStore().search_code(qv, top_k=pool, repo=repo)
        if rerank:
            hits = rerank_hits(query, hits, top_k)
        return _format_code(hits[:top_k])


@mcp.tool()
def search_docs(
    query: str, top_k: int = 5, topic: str | None = None, rerank: bool = True
) -> str:
    """Semantic search over indexed documents (PDFs, DOCX, web pages, notes).

    Returns ranked passages with source, page, and topic tags.

    Args:
        query: Natural-language question or description.
        top_k: Number of results to return.
        topic: Optional topic-tag filter (e.g. "politics", "astronomy", "technology").
        rerank: Rerank the candidate pool with the cross-encoder (more precise).
    """
    with _lock:
        qv = _embed_query(query, DOC_QUERY_TASK)
        pool = max(top_k, MIN_RERANK_POOL) if rerank else top_k
        store = QdrantStore(collection=DOC_COLLECTION)
        hits = store.search_docs(qv, top_k=pool, topics=[topic] if topic else None)
        if rerank:
            hits = rerank_hits(query, hits, top_k)
        return _format_docs(hits[:top_k])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
