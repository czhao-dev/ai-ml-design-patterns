"""Shared test fixtures: a hand-written fake OpenAI client (no unittest.mock,
matching the repo's plain-pytest testing style) plus small response builders."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make `src` importable when pytest is run from the project root, without
# requiring a package install step (matches the repo's independent-venv,
# no-setup.py convention for API-based projects).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 10


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str  # JSON string, matching the real OpenAI SDK shape


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunctionCall


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)


def make_tool_call_response(
    tool_calls: list[tuple[str, str, dict]], usage: FakeUsage | None = None
) -> FakeResponse:
    """tool_calls: [(call_id, tool_name, tool_input_dict), ...]"""
    calls = [
        FakeToolCall(id=call_id, function=FakeFunctionCall(name=name, arguments=json.dumps(tool_input)))
        for call_id, name, tool_input in tool_calls
    ]
    message = FakeMessage(content=None, tool_calls=calls)
    return FakeResponse(choices=[FakeChoice(message=message, finish_reason="tool_calls")], usage=usage or FakeUsage())


def make_stop_response(text: str, usage: FakeUsage | None = None) -> FakeResponse:
    message = FakeMessage(content=text, tool_calls=None)
    return FakeResponse(choices=[FakeChoice(message=message, finish_reason="stop")], usage=usage or FakeUsage())


class FakeChatCompletions:
    def __init__(self, scripted_responses: list[FakeResponse]) -> None:
        self._responses = list(scripted_responses)
        self._index = 0
        self.calls: list[dict] = []

    def create(self, **kwargs) -> FakeResponse:
        self.calls.append(kwargs)
        if self._index >= len(self._responses):
            raise AssertionError(
                f"FakeChatCompletions ran out of scripted responses after {self._index} calls "
                "-- the agent under test made more LLM calls than the test scripted."
            )
        response = self._responses[self._index]
        self._index += 1
        return response


class FakeChat:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """Mirrors the `client.chat.completions.create(...)` shape of the real
    openai.OpenAI() client, scripted with a fixed sequence of responses."""

    def __init__(self, scripted_responses: list[FakeResponse]) -> None:
        self.chat = FakeChat(FakeChatCompletions(scripted_responses))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls
