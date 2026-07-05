"""Tests for src/tools/error_injection.py."""

from src.tools.base import CallContext, ToolRegistry, ToolResult
from src.tools.error_injection import FlakyToolWrapper


class _DummyTool:
    name = "dummy_tool"
    description = "A dummy tool for testing."
    input_schema = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, call_context: CallContext, **kwargs) -> ToolResult:
        self.call_count += 1
        return ToolResult(content=f"real result #{self.call_count}")


def test_wrapper_returns_fake_error_on_configured_call_number():
    dummy = _DummyTool()
    wrapper = FlakyToolWrapper(
        dummy, target_tool_name="dummy_tool", fail_on_nth_call=2, error_message="injected failure"
    )
    registry = ToolRegistry()
    registry.register(wrapper)
    call_context = CallContext.for_task()

    first = registry.dispatch("dummy_tool", {}, call_context)
    second = registry.dispatch("dummy_tool", {}, call_context)

    assert not first.is_error
    assert second.is_error
    assert second.content == "injected failure"


def test_wrapper_passes_through_before_and_after_target_call():
    dummy = _DummyTool()
    wrapper = FlakyToolWrapper(
        dummy, target_tool_name="dummy_tool", fail_on_nth_call=2, error_message="injected failure"
    )
    registry = ToolRegistry()
    registry.register(wrapper)
    call_context = CallContext.for_task()

    first = registry.dispatch("dummy_tool", {}, call_context)
    second = registry.dispatch("dummy_tool", {}, call_context)
    third = registry.dispatch("dummy_tool", {}, call_context)

    assert not first.is_error
    assert second.is_error
    assert not third.is_error
    # The injected call never reaches the real tool -- only 2 real invocations
    # happened even though dispatch() was called 3 times.
    assert dummy.call_count == 2


def test_wrapper_only_fails_for_the_targeted_tool_name():
    dummy = _DummyTool()
    wrapper = FlakyToolWrapper(
        dummy, target_tool_name="some_other_tool", fail_on_nth_call=1, error_message="injected failure"
    )
    registry = ToolRegistry()
    registry.register(wrapper)
    call_context = CallContext.for_task()

    result = registry.dispatch("dummy_tool", {}, call_context)

    assert not result.is_error
