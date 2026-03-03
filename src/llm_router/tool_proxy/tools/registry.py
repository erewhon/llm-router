"""Tool registry for the tool-calling proxy."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of available tools with their definitions and implementations."""

    def __init__(self) -> None:
        self._definitions: list[dict[str, Any]] = []
        self._functions: dict[str, Callable[..., str]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., str],
    ) -> None:
        """Register a tool definition and its implementation."""
        self._definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
        self._functions[name] = func

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """OpenAI-format tool definitions for passing to the model."""
        return list(self._definitions)

    @property
    def names(self) -> set[str]:
        """Set of registered tool names."""
        return set(self._functions.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._functions

    def execute(self, name: str, arguments: str) -> str:
        """Execute a tool call and return the result as a string."""
        func = self._functions.get(name)
        if not func:
            return f"Unknown tool: {name}"
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return f"Invalid arguments: {arguments}"
        logger.info(f"Executing tool: {name}({args})")
        return func(**args)
