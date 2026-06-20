"""Cross-encoder reranker (qwen3-reranker-4b on OpenArc).

OpenArc's rerank endpoint is direct-call only — LiteLLM's rerank path returns
"Unsupported provider: openai" for it. The response shape is OpenVINO-specific:

    {"data": [{"index": <orig_pos>, "ranked_documents": {"doc": ..., "score": ...}}, ...]}

so we map it to plain (original_index, score) pairs sorted high-to-low.
"""

from __future__ import annotations

from collections import defaultdict, deque

import httpx

from llm_router.rag.embed import OPENARC_URL


class OpenArcReranker:
    def __init__(
        self,
        base_url: str = OPENARC_URL,
        model: str = "qwen3-reranker-4b",
        rerank_path: str = "/v1/rerank",
        max_doc_chars: int = 4_000,
        batch_size: int = 8,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.rerank_path = rerank_path
        # Cross-encoder relevance only needs the leading text; full chunks (code
        # can be ~30k chars) are slow and get truncated by the model anyway.
        self.max_doc_chars = max_doc_chars
        # Send small sequential batches rather than one big request: OpenArc's
        # reranker worker wedges under a large or concurrent load, so we keep
        # each request small and never fire them in parallel.
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenArcReranker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """Score each document against the query.

        Returns ``[(original_index, score), ...]`` sorted by score descending,
        truncated to ``top_n`` if given.

        Note: OpenArc returns ``data`` already sorted by relevance, and its
        ``index`` field is the output *rank*, not the original position — so we
        recover the original index by matching the echoed ``doc`` text (verified
        to be returned verbatim, even for long chunks).
        """
        if not documents:
            return []
        # Truncate before sending; OpenArc echoes back exactly what we send, so
        # build the index map from the same truncated strings we post.
        sent = [d[: self.max_doc_chars] for d in documents]

        # text -> queue of original indices (a queue so duplicate texts map 1:1)
        positions: dict[str, deque[int]] = defaultdict(deque)
        for i, doc in enumerate(sent):
            positions[doc].append(i)

        scored: list[tuple[int, float]] = []
        for start in range(0, len(sent), self.batch_size):
            batch = sent[start : start + self.batch_size]
            resp = self._client.post(
                f"{self.base_url}{self.rerank_path}",
                json={"model": self.model, "query": query, "documents": batch},
            )
            resp.raise_for_status()
            for item in resp.json().get("data", []):
                ranked = item["ranked_documents"]
                queue = positions.get(ranked["doc"])
                if not queue:
                    continue  # unmatched echo (shouldn't happen) — drop it
                scored.append((queue.popleft(), float(ranked["score"])))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n] if top_n is not None else scored


def rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Rerank Qdrant payload dicts (each with a ``text`` field) by relevance.

    Returns the top-k hits reordered, each with a ``rerank_score`` added.
    """
    if not hits:
        return []
    with OpenArcReranker() as reranker:
        ranked = reranker.rerank(query, [h.get("text", "") for h in hits], top_n=top_k)
    out: list[dict] = []
    for idx, score in ranked:
        hit = dict(hits[idx])
        hit["rerank_score"] = score
        out.append(hit)
    return out
