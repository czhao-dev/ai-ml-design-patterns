"""Tests for src/agents/react.py -- control flow only, no real API calls."""

from conftest import FakeOpenAIClient, make_stop_response, make_tool_call_response

from src.agents.react import ReActAgent
from src.tasks import Task
from src.tools.base import ToolRegistry, ToolResult


def _make_task(max_steps_override=None) -> Task:
    return Task(
        task_id="t1",
        category="arithmetic",
        prompt="What is 2 + 2?",
        expected_answer="4",
        answer_normalization="numeric",
        expected_tools=["calculator"],
        max_steps_override=max_steps_override,
    )


class _EchoTool:
    name = "calculator"
    description = "test tool"
    input_schema = {"type": "object", "properties": {}}

    def __call__(self, call_context, **kwargs) -> ToolResult:
        return ToolResult(content="4")


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


def test_stops_immediately_on_end_turn():
    client = FakeOpenAIClient([make_stop_response("4")])
    agent = ReActAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps=8)

    result = agent.run(_make_task())

    assert result.final_answer == "4"
    assert len(client.calls) == 1
    assert result.num_tool_calls == 0


def test_dispatches_tool_and_continues_to_end_turn():
    client = FakeOpenAIClient(
        [
            make_tool_call_response([("call_1", "calculator", {"expression": "2 + 2"})]),
            make_stop_response("4"),
        ]
    )
    agent = ReActAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps=8)

    result = agent.run(_make_task())

    assert result.final_answer == "4"
    assert result.num_tool_calls == 1
    assert result.tool_calls_made[0]["tool"] == "calculator"
    second_call_messages = client.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_call_messages)


def test_hits_max_steps_without_raising():
    max_steps = 3
    scripted = [
        make_tool_call_response([("call_x", "calculator", {"expression": "1+1"})]) for _ in range(max_steps)
    ]
    client = FakeOpenAIClient(scripted)
    agent = ReActAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps=max_steps)

    result = agent.run(_make_task())

    assert result.final_answer is None
    assert result.num_llm_calls == max_steps
    assert len(client.calls) == max_steps


def test_max_steps_override_takes_precedence():
    scripted = [make_tool_call_response([("call_x", "calculator", {"expression": "1+1"})]) for _ in range(2)]
    client = FakeOpenAIClient(scripted)
    agent = ReActAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps=8)

    result = agent.run(_make_task(max_steps_override=2))

    assert result.num_llm_calls == 2
