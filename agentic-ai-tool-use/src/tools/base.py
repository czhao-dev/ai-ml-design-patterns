"""Shared tool contracts: ToolResult, Tool protocol, CallContext, ToolRegistry.

Every tool's __call__ takes the per-task-run CallContext as its first argument
(even tools that ignore it) so ToolRegistry.dispatch has one uniform calling
convention, and so error-injection wrapping (see error_injection.py) can read
the call count that dispatch() has already recorded for this tool this run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class CallContext:
    """Mutable per-task-run state threaded through every tool dispatch.

    Constructed fresh at the start of each agent.run(task) call -- never
    reused across tasks, so error-injection call counts from one task can't
    leak into another.
    """

    call_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_task(cls) -> "CallContext":
        return cls(call_counts={})


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict

    def __call__(self, call_context: CallContext, **kwargs) -> ToolResult: ...


class ToolRegistry:
    """Holds the active set of tools for one agent run and dispatches calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def replace(self, name: str, tool: Tool) -> None:
        """Swap the tool registered under `name` (used to install a FlakyToolWrapper
        for a single task run, then restore the original afterward)."""
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_tools(self) -> list[dict]:
        """Convert every registered tool to the OpenAI Chat Completions function-calling
        shape: {"type": "function", "function": {"name", "description", "parameters"}}."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def dispatch(self, name: str, tool_input: dict, call_context: CallContext) -> ToolResult:
        """Look up the tool, record the call, and invoke it. Never raises -- any
        exception from the tool itself is caught and converted to an error
        ToolResult, since a crashed tool must surface to the agent as a normal
        tool-result error, not kill the whole benchmark run."""
        call_context.call_counts[name] = call_context.call_counts.get(name, 0) + 1
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(content=f"Unknown tool: {name}", is_error=True)
        try:
            return tool(call_context, **tool_input)
        except Exception as exc:  # noqa: BLE001 - convert to tool_result, don't crash the harness
            return ToolResult(content=f"Tool '{name}' raised an exception: {exc}", is_error=True)
