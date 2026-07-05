"""Arithmetic tool. Evaluates a restricted subset of expression syntax via a
whitelisted AST walk -- never calls eval()/exec() on the model-supplied string.
"""

from __future__ import annotations

import ast

from .base import CallContext, ToolResult

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = _eval_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    # Anything else -- Call, Name, Attribute, Subscript, etc. -- is a default-deny:
    # only the node types explicitly handled above are ever evaluated.
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def evaluate_expression(expression: str) -> float:
    """Parse and evaluate a restricted arithmetic expression. Raises
    SyntaxError/ValueError/ZeroDivisionError on anything outside the whitelist;
    CalculatorTool.__call__ is responsible for catching these."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluate a single arithmetic expression using only numbers and the "
        "operators + - * / // % **. Does not support variables, function calls, "
        "or any other Python syntax."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression, e.g. '(42 + 8) / 5'",
            }
        },
        "required": ["expression"],
    }

    def __call__(self, call_context: CallContext, expression: str) -> ToolResult:
        try:
            result = evaluate_expression(expression)
        except ZeroDivisionError:
            return ToolResult(content="Error: division by zero", is_error=True)
        except (SyntaxError, ValueError, TypeError) as exc:
            return ToolResult(content=f"Error: invalid expression ({exc})", is_error=True)
        return ToolResult(content=str(result))
