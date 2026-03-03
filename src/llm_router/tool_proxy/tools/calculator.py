"""Safe math calculator tool."""

from __future__ import annotations

import math

from llm_router.tool_proxy.tools.registry import ToolRegistry

DEFINITION = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "The mathematical expression to evaluate (e.g. 'sqrt(144) + 2**10')",
        }
    },
    "required": ["expression"],
}

DESCRIPTION = (
    "Evaluate a mathematical expression. Supports arithmetic (+, -, *, /, **, %), "
    "functions (sqrt, sin, cos, tan, log, log10, log2, exp, abs, round, ceil, floor, "
    "factorial), and constants (pi, e, tau, inf). Use this for any calculation."
)

ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
}

ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "tau": math.tau,
}


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    from simpleeval import InvalidExpression, simple_eval

    try:
        result = simple_eval(
            expression,
            functions=ALLOWED_FUNCS,
            names=ALLOWED_NAMES,
        )
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            return str(int(result))
        return str(result)
    except InvalidExpression as e:
        return f"Invalid expression: {e}"
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return f"Math error: {e}"
    except Exception as e:
        return f"Calculator error: {e}"


def register(registry: ToolRegistry) -> None:
    """Register the calculator tool."""
    registry.register("calculator", DESCRIPTION, DEFINITION, calculator)
