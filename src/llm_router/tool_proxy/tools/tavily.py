"""Tavily AI search tool."""

from __future__ import annotations

import logging

from llm_router.tool_proxy.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFINITION = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query",
        },
        "search_depth": {
            "type": "string",
            "enum": ["basic", "advanced"],
            "description": (
                "Search depth: 'basic' (faster, 1 credit) "
                "or 'advanced' (more thorough, 2 credits)"
            ),
        },
    },
    "required": ["query"],
}

DESCRIPTION = (
    "Search the web using Tavily API, which is designed for AI agents and returns "
    "clean, relevant excerpts with an optional AI-generated summary. "
    "Use this for high-quality search results. Costs API credits."
)


def create_tavily_search(api_key: str):
    """Create a tavily_search function bound to the given API key."""

    def tavily_search(query: str, search_depth: str = "basic") -> str:
        """Search the web using Tavily API."""
        import requests as req_lib

        try:
            resp = req_lib.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "include_answer": True,
                    "max_results": 5,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            lines = []
            answer = data.get("answer")
            if answer:
                lines.append(f"AI Summary: {answer}\n")

            for r in data.get("results", []):
                title = r.get("title", "")
                content = r.get("content", "")
                url = r.get("url", "")
                score = r.get("score", 0)
                lines.append(f"- {title} (relevance: {score:.2f})\n  {content}\n  {url}")

            return "\n\n".join(lines) if lines else "No results found."
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Tavily search failed: {e}"

    return tavily_search


def register(registry: ToolRegistry, api_key: str | None = None) -> None:
    """Register the tavily_search tool (only if API key is available)."""
    if not api_key:
        return
    registry.register("tavily_search", DESCRIPTION, DEFINITION, create_tavily_search(api_key))
