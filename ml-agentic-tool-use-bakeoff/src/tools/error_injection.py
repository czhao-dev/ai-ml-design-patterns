"""Error-injection wrapper used by error_recovery-category tasks to test whether
an agent architecture can recover from a mid-run tool failure."""

from __future__ import annotations

from .base import CallContext, Tool, ToolResult


class FlakyToolWrapper:
    """Wraps a real tool. On the configured Nth call (1-indexed) to the target
    tool name, returns a synthetic error instead of dispatching to the real tool.
    All other calls pass through unchanged.

    Relies on ToolRegistry.dispatch() having already incremented
    call_context.call_counts[self.name] for *this* call before invoking __call__
    -- it reads that count rather than incrementing it itself, so calls aren't
    double-counted. Construct a fresh instance per task run (never reuse across
    tasks) so injection state can't leak between tasks.
    """

    def __init__(
        self,
        wrapped: Tool,
        target_tool_name: str,
        fail_on_nth_call: int,
        error_message: str,
    ) -> None:
        self.wrapped = wrapped
        self.name = wrapped.name
        self.description = wrapped.description
        self.input_schema = wrapped.input_schema
        self.target_tool_name = target_tool_name
        self.fail_on_nth_call = fail_on_nth_call
        self.error_message = error_message

    def __call__(self, call_context: CallContext, **kwargs) -> ToolResult:
        current_call_number = call_context.call_counts.get(self.name, 0)
        if self.name == self.target_tool_name and current_call_number == self.fail_on_nth_call:
            return ToolResult(content=self.error_message, is_error=True)
        return self.wrapped(call_context, **kwargs)
