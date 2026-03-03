"""Stateful streaming parser for <think> and <tool_call> tags.

Extracted from agent-service.py (lines 403-523). Handles partial tags
at chunk boundaries by buffering incomplete sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThinkingStreamParser:
    """Stateful parser that separates <think> and <tool_call> tags from content.

    Handles partial tags at chunk boundaries by buffering incomplete sequences.
    - <think>...</think> -> routed to reasoning output
    - <tool_call>...</tool_call> -> stripped from output (collected separately)
    """

    # State: None = normal content, "think" = inside <think>, "tool_call" = inside <tool_call>
    _state: str | None = None
    _buffer: str = ""
    _seen_think: bool = False
    _tool_call_texts: list[str] = field(default_factory=list)
    _current_tool_call: list[str] = field(default_factory=list)

    # All tags we handle: (open_tag, close_tag, state_name)
    _TAGS: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("<think>", "</think>", "think"),
            ("<tool_call>", "</tool_call>", "tool_call"),
        ]
    )

    def feed(self, text: str) -> tuple[str, str]:
        """Process a chunk of text.

        Returns (reasoning_text, content_text) — one or both may be empty.
        """
        self._buffer += text
        reasoning: list[str] = []
        content: list[str] = []

        while self._buffer:
            if self._state == "think":
                close_tag = "</think>"
                close_idx = self._buffer.find(close_tag)
                if close_idx != -1:
                    reasoning.append(self._buffer[:close_idx])
                    self._buffer = self._buffer[close_idx + len(close_tag) :]
                    self._state = None
                elif self._might_be_partial_tag(self._buffer, close_tag):
                    break
                else:
                    reasoning.append(self._buffer)
                    self._buffer = ""

            elif self._state == "tool_call":
                close_tag = "</tool_call>"
                close_idx = self._buffer.find(close_tag)
                if close_idx != -1:
                    self._current_tool_call.append(self._buffer[:close_idx])
                    self._tool_call_texts.append("".join(self._current_tool_call))
                    self._current_tool_call = []
                    self._buffer = self._buffer[close_idx + len(close_tag) :]
                    self._state = None
                elif self._might_be_partial_tag(self._buffer, close_tag):
                    break
                else:
                    # Still inside tool_call — buffer but don't emit
                    self._current_tool_call.append(self._buffer)
                    self._buffer = ""

            else:
                # Normal state — look for any opening tag
                earliest_idx = len(self._buffer)
                earliest_tag: tuple[str, str, str] | None = None

                for open_tag, _close_tag, state_name in self._TAGS:
                    idx = self._buffer.find(open_tag)
                    if idx != -1 and idx < earliest_idx:
                        earliest_idx = idx
                        earliest_tag = (open_tag, _close_tag, state_name)

                if earliest_tag is not None:
                    open_tag, _close_tag, state_name = earliest_tag
                    content.append(self._buffer[:earliest_idx])
                    self._buffer = self._buffer[earliest_idx + len(open_tag) :]
                    self._state = state_name
                    if state_name == "think":
                        self._seen_think = True
                else:
                    # Check for partial opening tags at the end
                    might_be_partial = False
                    for open_tag, _, _ in self._TAGS:
                        if self._might_be_partial_tag(self._buffer, open_tag):
                            might_be_partial = True
                            break

                    if might_be_partial:
                        # Hold back the ambiguous suffix
                        max_tag_len = max(len(t[0]) for t in self._TAGS)
                        safe_end = len(self._buffer) - max_tag_len + 1
                        if safe_end > 0:
                            content.append(self._buffer[:safe_end])
                            self._buffer = self._buffer[safe_end:]
                        break
                    else:
                        # Also handle orphaned closing tags
                        orphan_idx = len(self._buffer)
                        orphan_tag_len = 0
                        for _, close_tag, _ in self._TAGS:
                            idx = self._buffer.find(close_tag)
                            if idx != -1 and idx < orphan_idx:
                                orphan_idx = idx
                                orphan_tag_len = len(close_tag)

                        if orphan_tag_len > 0:
                            content.append(self._buffer[:orphan_idx])
                            self._buffer = self._buffer[orphan_idx + orphan_tag_len :]
                        else:
                            content.append(self._buffer)
                            self._buffer = ""

        return "".join(reasoning), "".join(content)

    @staticmethod
    def _might_be_partial_tag(text: str, tag: str) -> bool:
        """Check if the end of text could be the start of a partial tag."""
        return any(text.endswith(tag[:i]) for i in range(1, len(tag)))

    @property
    def in_think(self) -> bool:
        """Whether the parser is currently inside a <think> block."""
        return self._state == "think"
