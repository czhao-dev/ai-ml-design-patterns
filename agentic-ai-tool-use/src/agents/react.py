"""ReAct agent: a single flat tool-use loop, tools bound from turn 1, no
planning phase and no explicit retry-on-failure logic. A tool error surfaces
as a tool result the model may or may not recover from within the same loop --
there is no harness-level intervention (contrast with ReflexionAgent)."""

from __future__ import annotations

import time

from ..tasks import Task
from ..tools.base import CallContext
from .base import AgentResult, BaseAgent

REACT_SYSTEM_PROMPT = (
    "You are a careful problem-solving assistant with access to tools: a "
    "calculator, a knowledge-base search tool, and a Python code executor. "
    "Reason about what you need before calling a tool. Call tools as needed to "
    "gather facts or compute results, then give a single, direct final answer "
    "with no extra commentary once you have enough information. If a tool "
    "returns an error, try a different approach rather than repeating the same "
    "failing call verbatim."
)


class ReActAgent(BaseAgent):
    architecture = "react"

    def __init__(
        self,
        client,
        model: str,
        registry,
        max_tokens: int,
        temperature: float = 0.0,
        max_steps: int = 8,
    ) -> None:
        super().__init__(client, model, registry, max_tokens, temperature)
        self.max_steps = max_steps

    def run(self, task: Task) -> AgentResult:
        call_context = CallContext.for_task()
        tool_calls_log: list[dict] = []
        start = time.monotonic()

        messages = [{"role": "user", "content": task.prompt}]
        max_steps = task.max_steps_override or self.max_steps
        transcript, final_answer, llm_calls, input_tokens, output_tokens = self._run_tool_use_loop(
            messages, REACT_SYSTEM_PROMPT, call_context, max_steps, tool_calls_log
        )

        return AgentResult(
            task_id=task.task_id,
            architecture=self.architecture,
            final_answer=final_answer,
            transcript=transcript,
            tool_calls_made=tool_calls_log,
            num_llm_calls=llm_calls,
            num_tool_calls=len(tool_calls_log),
            latency_sec=time.monotonic() - start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
