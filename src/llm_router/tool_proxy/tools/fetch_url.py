"""URL text extraction tool."""

from __future__ import annotations

import logging
import re

from llm_router.tool_proxy.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_FETCH_CHARS = 8000

DEFINITION = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The full URL to fetch (must start with http:// or https://)",
        }
    },
    "required": ["url"],
}

DESCRIPTION = (
    "Fetch and read the contents of a web page. Use this to read articles, "
    "documentation, or any URL. Returns the main text content of the page."
)


def create_fetch_url(proxy: str | None = None):
    """Create a fetch_url function with optional proxy."""
    proxies = {"http": proxy, "https": proxy} if proxy else None

    def fetch_url(url: str) -> str:
        """Fetch a URL and extract readable text content."""
        import requests as req_lib

        try:
            resp = req_lib.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"},
                proxies=proxies,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "json" not in content_type and "xml" not in content_type:
                return f"Non-text content type: {content_type}"

            html = resp.text

            # Use trafilatura for clean text extraction
            try:
                import trafilatura

                text = trafilatura.extract(
                    html,
                    include_links=True,
                    include_tables=True,
                    favor_recall=True,
                )
                if text:
                    if len(text) > MAX_FETCH_CHARS:
                        text = (
                            text[:MAX_FETCH_CHARS]
                            + f"\n\n[Truncated — showing first {MAX_FETCH_CHARS} "
                            f"of {len(text)} characters]"
                        )
                    return text
            except ImportError:
                pass

            # Fallback: strip HTML tags
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return f"Could not extract text content from: {url}"
            if len(text) > MAX_FETCH_CHARS:
                text = text[:MAX_FETCH_CHARS] + "\n\n[Truncated]"
            return text

        except Exception as e:
            logger.error(f"URL fetch failed: {e}")
            return f"Fetch failed: {e}"

    return fetch_url


def register(registry: ToolRegistry, proxy: str | None = None) -> None:
    """Register the fetch_url tool."""
    registry.register("fetch_url", DESCRIPTION, DEFINITION, create_fetch_url(proxy))
