"""Auto-routing via embedding similarity.

Classifies incoming prompts and routes to the best model alias
by comparing the prompt embedding against pre-computed category embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("auto-router")

# Category definitions: alias → description used for embedding
ROUTE_CATEGORIES: dict[str, str] = {
    "coder": (
        "Write code, debug, fix bugs, refactor, implement features, "
        "programming, software development, functions, classes, algorithms"
    ),
    "coder-veryfast": (
        "Quick simple question, one-liner, what does this do, "
        "brief lookup, short answer, trivial task, basic math, "
        "calculate, how many, definition, what is, simple fact"
    ),
    "thinker": (
        "Explain in depth, analyze, reason about, compare tradeoffs, "
        "plan architecture, think through, complex analysis, strategy"
    ),
    "research": (
        "Search the web, find current information, latest news, "
        "look up, what happened, recent events, fact check"
    ),
    "vision": (
        "Describe this image, what do you see, screenshot, photo, "
        "picture, visual, diagram, chart, OCR, read this image"
    ),
}

# Cached category embeddings (computed at startup)
_category_embeddings: dict[str, list[float]] | None = None
_embed_url: str = ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _get_embedding(text: str) -> list[float]:
    """Get embedding from the local embedding model."""
    async with httpx.AsyncClient(timeout=5) as client:
        resp = await client.post(
            f"{_embed_url}/v1/embeddings",
            json={
                "model": "text-embedding-nomic-embed-text-v1.5",
                "input": text,
            },
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def initialize(embed_url: str = "http://localhost:1234") -> None:
    """Pre-compute category embeddings at startup."""
    global _category_embeddings, _embed_url
    _embed_url = embed_url

    logger.info("Computing category embeddings for auto-router...")
    _category_embeddings = {}
    for alias, description in ROUTE_CATEGORIES.items():
        try:
            embedding = await _get_embedding(description)
            _category_embeddings[alias] = embedding
            logger.info(f"  {alias}: {len(embedding)} dims")
        except Exception as e:
            logger.warning(f"  {alias}: failed to embed — {e}")

    logger.info(f"Auto-router ready with {len(_category_embeddings)} categories")


async def classify(messages: list[dict[str, Any]]) -> str:
    """Classify a message list and return the best model alias."""
    if not _category_embeddings:
        logger.warning("Auto-router not initialized, defaulting to coder")
        return "coder"

    # Extract the last user message
    user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_msg = content
            elif isinstance(content, list):
                # Multimodal — has images → route to vision
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url" or part.get("type") == "image":
                            logger.info("Auto-route: detected image → vision")
                            return "vision"
                        if part.get("type") == "text":
                            user_msg = part.get("text", "")
            break

    if not user_msg:
        return "coder"

    # Truncate long prompts — we only need the intent, not the full context
    classify_text = user_msg[:500]

    try:
        prompt_embedding = await _get_embedding(classify_text)
    except Exception as e:
        logger.warning(f"Auto-route embedding failed: {e}, defaulting to coder")
        return "coder"

    # Find best match
    best_alias = "coder"
    best_score = -1.0
    scores = {}
    for alias, cat_embedding in _category_embeddings.items():
        score = _cosine_similarity(prompt_embedding, cat_embedding)
        scores[alias] = round(score, 3)
        if score > best_score:
            best_score = score
            best_alias = alias

    logger.info(f"Auto-route: '{classify_text[:60]}...' → {best_alias} ({best_score:.3f}) scores={scores}")
    return best_alias
