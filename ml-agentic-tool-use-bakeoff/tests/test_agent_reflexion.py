"""Tests for src/agents/reflexion.py -- control flow only, no real API calls."""

from conftest import FakeOpenAIClient, make_stop_response, make_tool_call_response

from src.agents.reflexion import ReflexionAgent
from src.tasks import Task
from src.tools.base import ToolRegistry, ToolResult


def _make_task() -> Task:
    return Task(
        task_id="t1",
        category="error_recovery",
        prompt="What is 2 + 2?",
        expected_answer="4",
        answer_normalization="numeric",
        expected_tools=["calculator"],
    )


class _FlakyOnceTool:
    """Fails on its first call, succeeds afterward -- simulates an injected error."""

    name = "calculator"
    description = "test tool"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, call_context, **kwargs) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return ToolResult(content="Error: temporarily unavailable", is_error=True)
        return ToolResult(content="4")


class _AlwaysFailTool:
    name = "calculator"
    description = "test tool"
    input_schema = {"type": "object", "properties": {}}

    def __call__(self, call_context, **kwargs) -> ToolResult:
        return ToolResult(content="Error: permanently broken", is_error=True)


def _make_registry(tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def test_no_reflection_when_first_attempt_clean():
    client = FakeOpenAIClient([make_stop_response("4")])
    agent = ReflexionAgent(
        client, "gpt-4.1-mini", _make_registry(_FlakyOnceTool()), max_tokens=100, max_steps=8, max_attempts=3
    )

    result = agent.run(_make_task())

    assert result.final_answer == "4"
    assert len(result.reflections) == 0
    assert len(client.calls) == 1


def test_retries_after_unresolved_tool_error_and_second_attempt_succeeds():
    client = FakeOpenAIClient(
        [
            # Attempt 1: tool call fails, model gives up with a self-admitted failure.
            make_tool_call_response([("call_1", "calculator", {"expression": "2 + 2"})]),
            make_stop_response("I was unable to compute the answer."),
            # Reflection call.
            make_stop_response("The calculator errored; retry the same calculation."),
            # Attempt 2: succeeds.
            make_tool_call_response([("call_2", "calculator", {"expression": "2 + 2"})]),
            make_stop_response("4"),
        ]
    )
    agent = ReflexionAgent(
        client, "gpt-4.1-mini", _make_registry(_FlakyOnceTool()), max_tokens=100, max_steps=8, max_attempts=3
    )

    result = agent.run(_make_task())

    assert result.final_answer == "4"
    assert len(result.reflections) == 1


def test_gives_up_after_max_attempts_without_infinite_loop():
    max_attempts = 3
    scripted = []
    for _ in range(max_attempts):
        scripted.append(make_tool_call_response([("call_x", "calculator", {"expression": "2 + 2"})]))
        scripted.append(make_stop_response("I was unable to compute the answer."))
        scripted.append(make_stop_response("reflection text"))
    scripted = scripted[:-1]  # no reflection call is made after the final attempt

    client = FakeOpenAIClient(scripted)
    agent = ReflexionAgent(
        client, "gpt-4.1-mini", _make_registry(_AlwaysFailTool()), max_tokens=100, max_steps=8, max_attempts=max_attempts
    )

    result = agent.run(_make_task())

    assert result.final_answer == "I was unable to compute the answer."
    assert len(result.reflections) == max_attempts - 1


def test_reflection_prompt_never_contains_expected_answer():
    # Note: the FINAL answer ("4") coincidentally equals task.expected_answer once
    # the tool succeeds on attempt 2 -- that's expected and not a leak. The actual
    # invariant under test is that the self-critique (reflection) call specifically
    # -- built only from the failed attempt's own transcript -- never embeds
    # task.expected_answer, so we check that call in isolation rather than every
    # call in the run.
    client = FakeOpenAIClient(
        [
            make_tool_call_response([("call_1", "calculator", {"expression": "2 + 2"})]),
            make_stop_response("I was unable to compute the answer."),
            make_stop_response("reflection text"),
            make_tool_call_response([("call_2", "calculator", {"expression": "2 + 2"})]),
            make_stop_response("4"),
        ]
    )
    agent = ReflexionAgent(
        client, "gpt-4.1-mini", _make_registry(_FlakyOnceTool()), max_tokens=100, max_steps=8, max_attempts=3
    )
    task = _make_task()

    agent.run(task)

    # The reflection call is the 3rd LLM call: attempt-1 tool call, attempt-1
    # stop, then the self-critique call.
    reflection_call_messages = client.calls[2]["messages"]
    for message in reflection_call_messages:
        content = message.get("content") or ""
        assert task.expected_answer not in content
