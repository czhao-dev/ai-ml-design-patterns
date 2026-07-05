#!/usr/bin/env python3
"""Run one architecture on one task and print the full step-by-step transcript.

Usage:
    python scripts/demo_single_task.py --architecture reflexion --task-id err_004
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from src import config
from src.agents.plan_execute import PlanExecuteAgent
from src.agents.react import ReActAgent
from src.agents.reflexion import ReflexionAgent
from src.metrics import score_task
from src.tasks import load_tasks
from src.tool_setup import apply_error_injection, build_registry

AGENT_FACTORIES = {
    "react": lambda client, settings, registry: ReActAgent(
        client,
        settings.model_id,
        registry,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        max_steps=settings.max_steps_react,
    ),
    "plan_execute": lambda client, settings, registry: PlanExecuteAgent(
        client,
        settings.model_id,
        registry,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        max_steps_per_subtask=settings.max_steps_per_subtask,
    ),
    "reflexion": lambda client, settings, registry: ReflexionAgent(
        client,
        settings.model_id,
        registry,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        max_steps=settings.max_steps_react,
        max_attempts=settings.max_attempts_reflexion,
    ),
}


def _print_transcript(transcript: list[dict]) -> None:
    for message in transcript:
        role = message.get("role")
        if role == "system":
            continue
        print(f"--- {role} ---")
        if message.get("content"):
            print(message["content"])
        for tool_call in message.get("tool_calls") or []:
            print(f"  tool_call: {tool_call['function']['name']}({tool_call['function']['arguments']})")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one architecture on one task and print the transcript.")
    parser.add_argument("--architecture", choices=sorted(AGENT_FACTORIES), required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()

    settings = config.get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    registry = build_registry(settings)
    tasks = load_tasks(config.TASKS_PATH)
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if task is None:
        raise SystemExit(f"No task with task_id={args.task_id!r} found in {config.TASKS_PATH}")

    agent = AGENT_FACTORIES[args.architecture](client, settings, registry)

    print(f"Task: {task.prompt}")
    print(f"Expected answer: {task.expected_answer}")
    if task.error_injection:
        print(f"Error injection: {task.error_injection}")
    print()

    with apply_error_injection(registry, task.error_injection):
        result = agent.run(task)

    _print_transcript(result.transcript)
    if result.plan:
        print(f"Plan: {result.plan}")
    if result.reflections:
        print("Reflections:")
        for i, reflection in enumerate(result.reflections, 1):
            print(f"  {i}. {reflection}")

    eval_result = score_task(result, task)
    print(f"\nFinal answer: {result.final_answer!r}")
    print(f"Success: {eval_result.success}")


if __name__ == "__main__":
    main()
