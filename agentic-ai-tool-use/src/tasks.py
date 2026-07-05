"""Task definitions loaded from data/tasks.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ErrorInjection:
    target_tool: str
    fail_on_nth_call: int
    error_message: str


@dataclass
class Task:
    task_id: str
    category: str  # "arithmetic" | "multihop_qa" | "code_exec" | "error_recovery"
    prompt: str
    expected_answer: str
    answer_normalization: str  # "numeric" | "string_ci" | "string_exact"
    expected_tools: list[str] | None = None
    error_injection: ErrorInjection | None = None
    max_steps_override: int | None = None
    notes: str = ""


def _parse_task(raw: dict) -> Task:
    error_injection = None
    if raw.get("error_injection"):
        error_injection = ErrorInjection(**raw["error_injection"])
    return Task(
        task_id=raw["task_id"],
        category=raw["category"],
        prompt=raw["prompt"],
        expected_answer=raw["expected_answer"],
        answer_normalization=raw["answer_normalization"],
        expected_tools=raw.get("expected_tools"),
        error_injection=error_injection,
        max_steps_override=raw.get("max_steps_override"),
        notes=raw.get("notes", ""),
    )


def load_tasks(path: Path) -> list[Task]:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(_parse_task(json.loads(line)))
    return tasks
