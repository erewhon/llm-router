"""OpenAI-compatible embedding clients (OpenArc, OVMS)."""

from __future__ import annotations

import httpx

OPENARC_URL = "http://euclid.local:5404"
OVMS_URL = "http://euclid.local:5399"


class OpenAIEmbedder:
    """Generic OpenAI-spec embeddings client.

    Single-stream mode sends one input per HTTP call (needed for OpenArc until
    its batched-input bug is fixed upstream). Bisect-on-failure handles
    per-model token limits by halving the batch (and then the char cap).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        embeddings_path: str = "/v1/embeddings",
        max_input_chars: int = 30_000,
        single_stream: bool = False,
        query_prefix: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embeddings_path = embeddings_path
        self.max_input_chars = max_input_chars
        self.single_stream = single_stream
        self.query_prefix = query_prefix
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAIEmbedder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self.query_prefix + t for t in texts] if is_query and self.query_prefix else texts
        if self.single_stream:
            out: list[list[float]] = []
            for t in prepared:
                out.extend(self._embed_recursive([t], self.max_input_chars))
            return out
        return self._embed_recursive(prepared, self.max_input_chars)

    def _embed_recursive(self, texts: list[str], max_chars: int) -> list[list[float]]:
        truncated = [t[:max_chars] for t in texts]
        resp = self._client.post(
            f"{self.base_url}{self.embeddings_path}",
            json={"model": self.model, "input": truncated},
        )
        if resp.status_code == 200:
            data = resp.json()["data"]
            data.sort(key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        if resp.status_code == 400 and "Input length" in resp.text:
            if len(texts) == 1:
                if max_chars <= 200:
                    resp.raise_for_status()
                return self._embed_recursive(texts, max_chars // 2)
            mid = len(texts) // 2
            return self._embed_recursive(texts[:mid], max_chars) + self._embed_recursive(
                texts[mid:], max_chars
            )
        resp.raise_for_status()
        return []


def qwen3_embedder(
    query_task: str = "Given a query, retrieve relevant code passages",
    **overrides,
) -> OpenAIEmbedder:
    """qwen3-embedding-4b on OpenArc (2560-dim, single-stream)."""
    return OpenAIEmbedder(
        base_url=OPENARC_URL,
        model="qwen3-embedding-4b",
        embeddings_path="/v1/embeddings",
        max_input_chars=30_000,
        single_stream=True,
        query_prefix=f"Instruct: {query_task}\nQuery: ",
        **overrides,
    )


def codet5p_embedder(**overrides) -> OpenAIEmbedder:
    """codet5p-110m-embedding on OVMS (768-dim, batched)."""
    return OpenAIEmbedder(
        base_url=OVMS_URL,
        model="codet5p-110m-embedding",
        embeddings_path="/v3/embeddings",
        max_input_chars=2_400,
        single_stream=False,
        **overrides,
    )
