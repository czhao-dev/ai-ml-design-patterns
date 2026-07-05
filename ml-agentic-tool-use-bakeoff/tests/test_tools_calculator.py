"""Tests for src/tools/calculator.py."""

from src.tools.base import CallContext
from src.tools.calculator import CalculatorTool, evaluate_expression


def test_add_and_divide():
    assert evaluate_expression("2 + 3") == 5
    assert evaluate_expression("10 / 4") == 2.5


def test_operator_precedence_and_parens():
    assert evaluate_expression("(42 + 8) / 5") == 10.0
    assert evaluate_expression("2 + 3 * 4") == 14


def test_divide_by_zero_returns_error_result():
    tool = CalculatorTool()
    result = tool(CallContext.for_task(), expression="1 / 0")
    assert result.is_error
    assert "division by zero" in result.content.lower()


def test_disallowed_call_node_rejected():
    tool = CalculatorTool()
    result = tool(CallContext.for_task(), expression="__import__('os').system('echo hi')")
    assert result.is_error


def test_disallowed_name_reference_rejected():
    tool = CalculatorTool()
    result = tool(CallContext.for_task(), expression="x + 1")
    assert result.is_error


def test_malformed_expression_returns_error_result():
    tool = CalculatorTool()
    result = tool(CallContext.for_task(), expression="2 +")
    assert result.is_error
