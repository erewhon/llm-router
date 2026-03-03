"""SSE streaming helpers for OpenAI-compatible chat completions.

Extracted from agent-service.py (lines 366-399, 848-908).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi.responses import JSONResponse


def build_sse_chunk(
    chunk_id: str,
    model: str,
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> str:
    """Build a single SSE chunk in chat.completion.chunk format."""
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content

    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def build_response(
    model: str,
    content: str,
    usage: Any | None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> JSONResponse:
    """Build an OpenAI-compatible chat completion response."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls

    resp: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        resp["usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    return JSONResponse(content=resp)


def build_tool_calls_response(
    chunk_id: str,
    model: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    """Build a chat.completion response dict with tool_calls (for streaming breakout)."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls",
            }
        ],
    }
