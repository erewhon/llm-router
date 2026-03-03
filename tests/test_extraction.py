"""Tests for tool call extraction and thinking extraction."""

from dataclasses import dataclass
from typing import Any

from llm_router.tool_proxy.extraction import (
    extract_thinking,
    extract_tool_calls,
    extract_tool_calls_from_content,
    strip_tool_call_tags,
)


def test_extract_tool_calls_from_xml():
    content = '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
    calls = extract_tool_calls_from_content(content)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert '"query"' in calls[0].arguments


def test_extract_multiple_tool_calls():
    content = (
        '<tool_call>{"name": "search", "arguments": {"q": "a"}}</tool_call>'
        '<tool_call>{"name": "calc", "arguments": {"expression": "1+1"}}</tool_call>'
    )
    calls = extract_tool_calls_from_content(content)
    assert len(calls) == 2
    assert calls[0].name == "search"
    assert calls[1].name == "calc"


def test_extract_invalid_json_skipped():
    content = "<tool_call>not json</tool_call>"
    calls = extract_tool_calls_from_content(content)
    assert len(calls) == 0


def test_extract_missing_name_skipped():
    content = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
    calls = extract_tool_calls_from_content(content)
    assert len(calls) == 0


def test_extract_thinking_basic():
    content = "<think>reasoning here</think>clean content"
    reasoning, clean = extract_thinking(content)
    assert reasoning == "reasoning here"
    assert clean == "clean content"


def test_extract_thinking_multiple():
    content = "<think>first</think>middle<think>second</think>end"
    reasoning, clean = extract_thinking(content)
    assert "first" in reasoning
    assert "second" in reasoning
    assert clean == "middleend"


def test_extract_thinking_orphaned_close():
    content = "</think>after orphan"
    reasoning, clean = extract_thinking(content)
    assert reasoning == ""
    assert clean == "after orphan"


def test_extract_thinking_no_thinking():
    content = "just normal content"
    reasoning, clean = extract_thinking(content)
    assert reasoning == ""
    assert clean == "just normal content"


def test_strip_tool_call_tags():
    content = 'before<tool_call>{"name":"test"}</tool_call>after'
    result = strip_tool_call_tags(content)
    assert result == "beforeafter"


def test_strip_tool_call_tags_no_tags():
    content = "no tags here"
    result = strip_tool_call_tags(content)
    assert result == "no tags here"


# --- Tests for extract_tool_calls with msg object ---


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction


@dataclass
class MockMessage:
    content: str
    tool_calls: list[Any] | None = None


def test_extract_from_msg_tool_calls():
    """When msg.tool_calls is populated, use it directly."""
    msg = MockMessage(
        content="",
        tool_calls=[
            MockToolCall(
                id="call_123",
                function=MockFunction(name="calculator", arguments='{"expression": "2+2"}'),
            )
        ],
    )
    result = extract_tool_calls(msg, "")
    assert len(result) == 1
    assert result[0]["function"]["name"] == "calculator"
    assert result[0]["id"] == "call_123"


def test_extract_fallback_to_xml():
    """When msg.tool_calls is empty, fall back to XML extraction."""
    msg = MockMessage(
        content='<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>',
        tool_calls=None,
    )
    result = extract_tool_calls(msg, msg.content)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "web_search"


def test_extract_no_tool_calls():
    """When neither msg.tool_calls nor XML tags present, return empty."""
    msg = MockMessage(content="just a response", tool_calls=None)
    result = extract_tool_calls(msg, msg.content)
    assert len(result) == 0
