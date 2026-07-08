"""Reflexion agent: a bounded retry loop keyed off explicit failure signals --
an unresolved tool error, exhausting the step limit, or the model admitting
failure in its own words -- never the ground-truth answer. Between attempts, a
dedicated self-critique call (shown only the failed attempt's own transcript)
produces "lessons learned" that are prepended to the next attempt's prompt."""

from __future__ import annotations

import time

from ..tasks import Task
from ..tools.base import CallContext
from .base import AgentResult, BaseAgent

REFLEXION_SYSTEM_PROMPT = (
    "You are a careful problem-solving assistant with access to tools: a "
    "calculator, a knowledge-base search tool, and a Python code executor. "
    "Reason about what you need before calling a tool. Call tools as needed, "
    "then give a single, direct final answer with no extra commentary. If you "
    "are given lessons learned from a previous attempt, apply them. Give your "
    "final answer as the single bare fact requested -- a number, name, or "
    "short phrase -- with no surrounding sentence (e.g. \"445\" not \"The "
    "answer is 445\", and \"Jane Okoye\" not \"Jane Okoye founded the "
    "company\")."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are a self-critique assistant reviewing a failed attempt at a task. "
    "You are shown only the attempt's own transcript, never the correct answer. "
    "In 1-3 sentences, explain what likely went wrong and what should be done "
    "differently on the next attempt."
)

_FAILURE_PHRASES = (
    "cannot determine",
    "unable to",
    "failed to",
    "error occurred",
    "i don't know",
    "i do not know",
)


class ReflexionAgent(BaseAgent):
    architecture = "reflexion"

    def __init__(
        self,
        client,
        model: str,
        registry,
        max_tokens: int,
        temperature: float = 0.0,
        max_steps: int = 8,
        max_attempts: int = 3,
    ) -> None:
        super().__init__(client, model, registry, max_tokens, temperature)
        self.max_steps = max_steps
        self.max_attempts = max_attempts

    def run(self, task: Task) -> AgentResult:
        call_context = CallContext.for_task()
        tool_calls_log: list[dict] = []
        start = time.monotonic()
        reflections: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        llm_calls_total = 0
        max_steps = task.max_steps_override or self.max_steps

        transcript: list[dict] = []
        final_answer: str | None = None

        for attempt_num in range(1, self.max_attempts + 1):
            lessons_block = (
                "\n\nLessons learned from previous attempt(s):\n" + "\n".join(reflections)
                if reflections
                else ""
            )
            messages = [{"role": "user", "content": task.prompt + lessons_block}]
            attempt_tool_calls: list[dict] = []
            transcript, answer, llm_calls, in_tok, out_tok = self._run_tool_use_loop(
                messages, REFLEXION_SYSTEM_PROMPT, call_context, max_steps, attempt_tool_calls
            )
            tool_calls_log.extend(attempt_tool_calls)
            llm_calls_total += llm_calls
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            final_answer = answer

            unresolved_error = self._has_unresolved_tool_error(attempt_tool_calls)
            hit_step_limit = answer is None
            self_admits_failure = answer is not None and self._admits_failure(answer)
            attempt_failed = unresolved_error or hit_step_limit or self_admits_failure

            if not attempt_failed or attempt_num == self.max_attempts:
                break

            # Self-critique call. CRITICAL: never includes task.expected_answer --
            # only the failed attempt's own transcript -- so ground truth can't
            # leak into the agent's decision process.
            reflection_prompt = (
                f"Task: {task.prompt}\n"
                f"Your attempt:\n{self._render_transcript(transcript)}\n"
                "What went wrong, and what should you do differently next time? "
                "Answer in 1-3 sentences."
            )
            reflection_text, in_tok, out_tok = self._run_notool_call(
                REFLECTION_SYSTEM_PROMPT, reflection_prompt
            )
            llm_calls_total += 1
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            reflections.append(reflection_text or "(no reflection produced)")

        return AgentResult(
            task_id=task.task_id,
            architecture=self.architecture,
            final_answer=final_answer,
            transcript=transcript,
            tool_calls_made=tool_calls_log,
            num_llm_calls=llm_calls_total,
            num_tool_calls=len(tool_calls_log),
            latency_sec=time.monotonic() - start,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            reflections=reflections,
        )

    def _has_unresolved_tool_error(self, attempt_tool_calls: list[dict]) -> bool:
        """True if any call in this attempt errored and no later call to the
        *same tool name* (within this attempt) succeeded afterward."""
        for i, call in enumerate(attempt_tool_calls):
            if not call["is_error"]:
                continue
            later_success = any(
                later["tool"] == call["tool"] and not later["is_error"]
                for later in attempt_tool_calls[i + 1 :]
            )
            if not later_success:
                return True
        return False

    def _admits_failure(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(phrase in lowered for phrase in _FAILURE_PHRASES)

    def _render_transcript(self, transcript: list[dict]) -> str:
        lines = []
        for message in transcript:
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                lines.append(f"User: {message.get('content', '')}")
            elif role == "assistant":
                content = message.get("content") or ""
                if content:
                    lines.append(f"Assistant: {content}")
                for tc in message.get("tool_calls") or []:
                    lines.append(
                        f"Assistant called tool '{tc['function']['name']}' "
                        f"with input {tc['function']['arguments']}"
                    )
            elif role == "tool":
                lines.append(f"Tool result: {message.get('content', '')}")
        return "\n".join(lines)
