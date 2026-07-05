"""Tests for src/agents/plan_execute.py -- control flow only, no real API calls."""

from conftest import FakeOpenAIClient, make_stop_response

from src.agents.plan_execute import PlanExecuteAgent
from src.tasks import Task
from src.tools.base import ToolRegistry, ToolResult


def _make_task() -> Task:
    return Task(
        task_id="t1",
        category="multihop_qa",
        prompt="Who founded the company that makes Widget X1?",
        expected_answer="Jane Okoye",
        answer_normalization="string_ci",
        expected_tools=["search_knowledge_base"],
    )


class _EchoTool:
    name = "search_knowledge_base"
    description = "test tool"
    input_schema = {"type": "object", "properties": {}}

    def __call__(self, call_context, **kwargs) -> ToolResult:
        return ToolResult(content="AstraCorp was founded by Jane Okoye.")


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


def test_planner_call_has_no_tools_bound():
    client = FakeOpenAIClient(
        [
            make_stop_response('["Find who makes Widget X1", "Find who founded that company"]'),
            make_stop_response("Widget X1's maker is AstraCorp"),
            make_stop_response("AstraCorp was founded by Jane Okoye"),
            make_stop_response("Jane Okoye"),
        ]
    )
    agent = PlanExecuteAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps_per_subtask=4)

    agent.run(_make_task())

    assert "tools" not in client.calls[0]


def test_plan_json_parsed_correctly():
    client = FakeOpenAIClient(
        [
            make_stop_response('["step 1", "step 2"]'),
            make_stop_response("result 1"),
            make_stop_response("result 2"),
            make_stop_response("final answer"),
        ]
    )
    agent = PlanExecuteAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps_per_subtask=4)

    result = agent.run(_make_task())

    assert result.plan == ["step 1", "step 2"]


def test_malformed_plan_falls_back_to_line_split():
    client = FakeOpenAIClient(
        [
            make_stop_response("1. do X\n2. do Y"),
            make_stop_response("result 1"),
            make_stop_response("result 2"),
            make_stop_response("final answer"),
        ]
    )
    agent = PlanExecuteAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps_per_subtask=4)

    result = agent.run(_make_task())

    assert result.plan == ["do X", "do Y"]


def test_one_executor_subloop_per_subtask_and_synthesis_is_last_call():
    client = FakeOpenAIClient(
        [
            make_stop_response('["step 1", "step 2"]'),  # planner
            make_stop_response("result 1"),  # subtask 1 (no tool call made)
            make_stop_response("result 2"),  # subtask 2 (no tool call made)
            make_stop_response("final synthesis answer"),  # synthesis
        ]
    )
    agent = PlanExecuteAgent(client, "gpt-4.1-mini", _make_registry(), max_tokens=100, max_steps_per_subtask=4)

    result = agent.run(_make_task())

    assert result.final_answer == "final synthesis answer"
    assert len(client.calls) == 4
    assert "tools" not in client.calls[0]  # planner
    assert "tools" in client.calls[1]  # subtask 1's executor loop
    assert "tools" in client.calls[2]  # subtask 2's executor loop
    assert "tools" not in client.calls[-1]  # synthesis
