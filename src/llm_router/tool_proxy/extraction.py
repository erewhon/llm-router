"""Tool call extraction from model output.

Handles both native OpenAI tool_calls and fallback XML <tool_call> tag extraction.
Extracted from agent-service.py (lines 264-362).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"(?:<think>.*?</think>|</think>)\s*", re.DOTALL)


@dataclass
class ExtractedToolCall:
    """A tool call parsed from model output text."""

    id: str
    name: str
    arguments: str


def extract_tool_calls_from_content(content: str) -> list[ExtractedToolCall]:
    """Extract tool calls from <tool_call> XML tags in content text.

    Fallback for when vLLM's structured parser fails to populate tool_calls.
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(content):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        args = obj.get("arguments", {})
        if not name:
            continue
        calls.append(
            ExtractedToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=json.dumps(args) if isinstance(args, dict) else str(args),
            )
        )
    return calls


def extract_tool_calls(msg: Any, content: str) -> list[dict[str, Any]]:
    """Check msg.tool_calls first, fall back to <tool_call> XML extraction.

    Returns a list of tool call dicts in OpenAI format.
    """
    if msg.tool_calls:
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    extracted = extract_tool_calls_from_content(content)
    if extracted:
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in extracted
        ]

    return []


def extract_thinking(content: str) -> tuple[str, str]:
    """Separate <think>...</think> blocks from content.

    Returns (reasoning_content, clean_content).
    """
    thinking_parts = []
    for m in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
        thinking_parts.append(m.group(1).strip())

    clean = _THINK_RE.sub("", content).strip()
    reasoning = "\n\n".join(thinking_parts)
    return reasoning, clean


def strip_tool_call_tags(content: str) -> str:
    """Remove <tool_call>...</tool_call> blocks from content."""
    return _TOOL_CALL_RE.sub("", content).strip()
