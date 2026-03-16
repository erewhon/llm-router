"""DuckDuckGo web search tool."""

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
        }
    },
    "required": ["query"],
}

DESCRIPTION = (
    "Search the web for current information using DuckDuckGo. "
    "Use this when you need up-to-date facts, news, or information you don't have."
)


def create_web_search(proxy: str | None = None):
    """Create a web_search function with optional proxy."""

    def web_search(query: str) -> str:
        """Search the web using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            results = list(DDGS(proxy=proxy).text(query, max_results=5))
            if not results:
                return "No results found."
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                lines.append(f"- {title}\n  {body}\n  {href}")
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {e}"

    return web_search


def register(registry: ToolRegistry, proxy: str | None = None) -> None:
    """Register the web_search tool."""
    registry.register("web_search", DESCRIPTION, DEFINITION, create_web_search(proxy))
