"""Qdrant store for code chunks."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from llm_router.rag.chunker import Chunk

DEFAULT_QDRANT_URL = "http://euclid.local:6333"
CODE_COLLECTION = "code"
NAMESPACE = uuid.UUID("a0b7f0bf-1a76-4d8f-9a6e-2b7a0d20c2c0")


def chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.repo}|{chunk.path}|{chunk.chunk_kind}|{chunk.name}|{chunk.start_line}"
    return str(uuid.uuid5(NAMESPACE, key))


class QdrantStore:
    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection: str = CODE_COLLECTION,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert_code(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        points = [
            PointStruct(id=chunk_id(c), vector=v, payload=c.model_dump())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def search_code(
        self,
        query_vec: list[float],
        top_k: int = 10,
        repo: str | None = None,
    ) -> list[dict]:
        flt: Filter | None = None
        if repo:
            flt = Filter(must=[FieldCondition(key="repo", match=MatchValue(value=repo))])
        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        ).points
        return [{"score": p.score, **(p.payload or {})} for p in res]
