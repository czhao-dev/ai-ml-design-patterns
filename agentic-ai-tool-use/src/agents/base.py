"""Shared agent contracts and the low-level OpenAI Chat Completions tool-use
loop used by all three architectures (ReAct, Plan-and-Execute, Reflexion)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..tasks import Task
from ..tools.base import CallContext, ToolRegistry


@dataclass
class AgentResult:
    task_id: str
    architecture: str
    final_answer: str | None
    transcript: list[dict] = field(default_factory=list)
    tool_calls_made: list[dict] = field(default_factory=list)
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    plan: list[str] | None = None
    reflections: list[str] = field(default_factory=list)


def _message_to_dict(message) -> dict:
    """Converts an OpenAI ChatCompletionMessage into the plain-dict shape the
    Chat Completions API expects when it's echoed back into `messages` on a
    subsequent call."""
    result: dict = {"role": "assistant", "content": message.content}
    if getattr(message, "tool_calls", None):
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return result


class BaseAgent(ABC):
    architecture: str

    def __init__(
        self,
        client,
        model: str,
        registry: ToolRegistry,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def run(self, task: Task) -> AgentResult: ...

    def _call_llm(self, messages: list[dict], bind_tools: bool):
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if bind_tools:
            kwargs["tools"] = self.registry.to_openai_tools()
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _run_tool_use_loop(
        self,
        messages: list[dict],
        system_prompt: str,
        call_context: CallContext,
        max_steps: int,
        tool_calls_log: list[dict],
    ) -> tuple[list[dict], str | None, int, int, int]:
        """Repeatedly calls the OpenAI API with tools bound, dispatching tool
        calls, until the model stops (finish_reason == "stop") or max_steps LLM
        calls have been made in this loop -- guaranteeing termination even if
        the model never stops calling tools.

        `tool_calls_log` is mutated in place (appended to) as a side effect, so
        a caller running several sub-loops within one agent.run() (e.g.
        Plan-and-Execute's per-subtask loops) can pass in one shared log across
        all of them and get an accurate total tool-call count for the task.

        Returns (full_messages_including_system_prompt, final_text_or_None,
        llm_calls_made, input_tokens, output_tokens).
        """
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        llm_calls = 0
        input_tokens = 0
        output_tokens = 0
        final_text: str | None = None

        for _ in range(max_steps):
            response = self._call_llm(full_messages, bind_tools=True)
            llm_calls += 1
            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens += usage.prompt_tokens
                output_tokens += usage.completion_tokens

            choice = response.choices[0]
            message = choice.message
            full_messages.append(_message_to_dict(message))

            if choice.finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_input = json.loads(tool_call.function.arguments or "{}")
                    result = self.registry.dispatch(tool_call.function.name, tool_input, call_context)
                    tool_calls_log.append(
                        {"tool": tool_call.function.name, "input": tool_input, "is_error": result.is_error}
                    )
                    full_messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result.content}
                    )
                continue

            if choice.finish_reason == "stop":
                final_text = message.content
                break

            # Anything else (e.g. "length") -- stop without a final answer.
            break

        return full_messages, final_text, llm_calls, input_tokens, output_tokens

    def _run_notool_call(self, system_prompt: str, user_content: str) -> tuple[str | None, int, int]:
        """A single LLM call with no tools bound -- used for Plan-and-Execute's
        planning/synthesis calls and Reflexion's self-critique call, so the
        model cannot execute anything during these steps. Returns
        (text_or_None, input_tokens, output_tokens)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self._call_llm(messages, bind_tools=False)
        usage = getattr(response, "usage", None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        choice = response.choices[0]
        return choice.message.content, input_tokens, output_tokens
