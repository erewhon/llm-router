"""Tests for ThinkingStreamParser."""

from llm_router.tool_proxy.thinking import ThinkingStreamParser


def test_plain_text():
    parser = ThinkingStreamParser()
    r, c = parser.feed("Hello world")
    assert r == ""
    assert c == "Hello world"


def test_think_block():
    parser = ThinkingStreamParser()
    r, c = parser.feed("<think>reasoning here</think>content here")
    assert r == "reasoning here"
    assert c == "content here"


def test_think_block_split_across_chunks():
    parser = ThinkingStreamParser()
    r1, c1 = parser.feed("<think>part1")
    # Parser streams reasoning as it arrives (no need to wait for close tag)
    assert r1 == "part1"
    assert c1 == ""

    r2, c2 = parser.feed(" part2</think>final content")
    assert " part2" in r2
    assert c2 == "final content"


def test_tool_call_stripped():
    parser = ThinkingStreamParser()
    r, c = parser.feed('before<tool_call>{"name":"test"}</tool_call>after')
    assert r == ""
    assert c == "beforeafter"
    assert len(parser._tool_call_texts) == 1
    assert '"name":"test"' in parser._tool_call_texts[0]


def test_tool_call_split_across_chunks():
    parser = ThinkingStreamParser()
    _r1, c1 = parser.feed("before<tool_call>{")
    assert c1 == "before"

    _r2, c2 = parser.feed('"name":"test"}</tool_call>after')
    assert c2 == "after"
    assert len(parser._tool_call_texts) == 1


def test_think_and_tool_call():
    parser = ThinkingStreamParser()
    text = '<think>reasoning</think>content<tool_call>{"name":"calc"}</tool_call>more'
    r, c = parser.feed(text)
    assert r == "reasoning"
    assert c == "contentmore"
    assert len(parser._tool_call_texts) == 1


def test_partial_opening_tag():
    """Partial '<thi' at end of chunk should be buffered."""
    parser = ThinkingStreamParser()
    _r1, c1 = parser.feed("hello<thi")
    # The entire text is shorter than max tag len, so parser buffers it all
    # OR it emits "hello" and buffers "<thi". Either is valid.
    # Check combined output across chunks.
    all_content = c1

    # Complete the tag
    r2, c2 = parser.feed("nk>reasoning</think>done")
    all_content += c2
    assert "reasoning" in r2
    assert "done" in all_content
    assert "<thi" not in all_content


def test_orphaned_close_tag():
    """Orphaned </think> should be stripped."""
    parser = ThinkingStreamParser()
    r, c = parser.feed("content</think>more")
    assert r == ""
    assert "content" in c
    assert "more" in c
    assert "</think>" not in c


def test_multiple_think_blocks():
    parser = ThinkingStreamParser()
    r1, c1 = parser.feed("<think>first</think>between")
    assert r1 == "first"
    assert c1 == "between"

    r2, c2 = parser.feed("<think>second</think>end")
    assert r2 == "second"
    assert c2 == "end"


def test_in_think_property():
    parser = ThinkingStreamParser()
    assert not parser.in_think

    parser.feed("<think>inside")
    assert parser.in_think

    parser.feed("</think>outside")
    assert not parser.in_think


def test_empty_input():
    parser = ThinkingStreamParser()
    r, c = parser.feed("")
    assert r == ""
    assert c == ""
