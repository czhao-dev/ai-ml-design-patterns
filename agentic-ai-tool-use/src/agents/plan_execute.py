"""Plan-and-Execute agent: an explicit upfront planning call with tools
deliberately unbound (so nothing can execute before a plan exists), one bounded
tool-use sub-loop per subtask, and a final synthesis call, also with tools
unbound."""

from __future__ import annotations

import json
import time

from ..tasks import Task
from ..tools.base import CallContext
from .base import AgentResult, BaseAgent

PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant. Break the user's task into a short ordered "
    "list of concrete subtasks that, if completed in order, would solve it. "
    'Respond with ONLY a JSON array of strings, e.g. ["step one", "step two"]. '
    "Do not include any other text."
)

EXECUTOR_SYSTEM_PROMPT = (
    "You are an execution assistant with access to tools: a calculator, a "
    "knowledge-base search tool, and a Python code executor. You are given one "
    "subtask of a larger plan, plus the results of any earlier subtasks. Use "
    "tools as needed to complete ONLY this subtask, then state its result "
    "directly."
)

SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a synthesis assistant. You are given the original task and the "
    "results of each subtask that was executed to solve it. Combine them into "
    "a single, direct final answer with no extra commentary."
)


class PlanExecuteAgent(BaseAgent):
    architecture = "plan_execute"

    def __init__(
        self,
        client,
        model: str,
        registry,
        max_tokens: int,
        temperature: float = 0.0,
        max_steps_per_subtask: int = 4,
    ) -> None:
        super().__init__(client, model, registry, max_tokens, temperature)
        self.max_steps_per_subtask = max_steps_per_subtask

    def run(self, task: Task) -> AgentResult:
        call_context = CallContext.for_task()
        tool_calls_log: list[dict] = []
        start = time.monotonic()
        llm_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        # 1. Planning call -- tools deliberately unbound.
        plan_text, in_tok, out_tok = self._run_notool_call(PLANNER_SYSTEM_PROMPT, task.prompt)
        llm_calls += 1
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        plan = self._parse_plan(plan_text)

        # 2. One bounded tool-use sub-loop per subtask, carrying prior results forward.
        max_steps = task.max_steps_override or self.max_steps_per_subtask
        subtask_results: list[str] = []
        full_transcript: list[dict] = []
        for subtask in plan:
            prior_results = "\n".join(f"- {r}" for r in subtask_results) or "(none yet)"
            sub_user_content = (
                f"Overall goal: {task.prompt}\n"
                f"Full plan: {plan}\n"
                f"Prior subtask results:\n{prior_results}\n"
                f"Now complete this subtask: {subtask}"
            )
            sub_messages, sub_answer, sub_llm_calls, sub_in, sub_out = self._run_tool_use_loop(
                [{"role": "user", "content": sub_user_content}],
                EXECUTOR_SYSTEM_PROMPT,
                call_context,
                max_steps,
                tool_calls_log,
            )
            full_transcript.extend(sub_messages)
            subtask_results.append(sub_answer or "(no result produced)")
            llm_calls += sub_llm_calls
            total_input_tokens += sub_in
            total_output_tokens += sub_out

        # 3. Synthesis call -- tools unbound.
        synthesis_input = "Original task: " + task.prompt + "\nSubtask results:\n" + "\n".join(subtask_results)
        final_answer, in_tok, out_tok = self._run_notool_call(SYNTHESIZER_SYSTEM_PROMPT, synthesis_input)
        llm_calls += 1
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        return AgentResult(
            task_id=task.task_id,
            architecture=self.architecture,
            final_answer=final_answer,
            transcript=full_transcript,
            tool_calls_made=tool_calls_log,
            num_llm_calls=llm_calls,
            num_tool_calls=len(tool_calls_log),
            latency_sec=time.monotonic() - start,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            plan=plan,
        )

    def _parse_plan(self, text: str | None) -> list[str]:
        """Extract a JSON array of subtask strings from the planner's response.
        Falls back to splitting into lines and stripping a leading numbering/
        bullet marker if JSON parsing fails, so a slightly malformed planner
        response still produces a usable plan rather than crashing the run."""
        if not text:
            return ["Solve the task directly."]

        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

        lines = []
        for line in text.splitlines():
            stripped = line.strip().lstrip("0123456789.-*) ").strip()
            if stripped:
                lines.append(stripped)
        return lines or ["Solve the task directly."]
