"""Tests for src/tools/code_exec.py."""

import subprocess

from src.tools.base import CallContext
from src.tools.code_exec import CodeExecutionTool, run_python


def test_simple_arithmetic_stdout():
    result = run_python("print(2 + 2)", timeout_sec=5.0, memory_limit_mb=256)
    assert not result.is_error
    assert result.content == "4"


def test_disallowed_import_blocked_before_execution(monkeypatch):
    invoked = []
    original_run = subprocess.run

    def spy_run(*args, **kwargs):
        invoked.append((args, kwargs))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)

    result = run_python("import os\nprint(os.listdir('.'))", timeout_sec=5.0, memory_limit_mb=256)

    assert result.is_error
    assert "not allowed" in result.content.lower()
    assert invoked == []  # subprocess.run was never invoked -- rejected by the static pre-check


def test_disallowed_builtin_blocked_before_execution():
    result = run_python("eval('1+1')", timeout_sec=5.0, memory_limit_mb=256)
    assert result.is_error
    assert "not allowed" in result.content.lower()


def test_allowed_import_executes_normally():
    result = run_python("import math\nprint(math.sqrt(16))", timeout_sec=5.0, memory_limit_mb=256)
    assert not result.is_error
    assert result.content == "4.0"


def test_execution_times_out():
    code = "while True:\n    pass\n"
    result = run_python(code, timeout_sec=0.5, memory_limit_mb=256)
    assert result.is_error
    assert "timed out" in result.content.lower()


def test_subprocess_env_excludes_api_key(monkeypatch):
    captured_env: dict = {}
    original_run = subprocess.run

    def spy_run(*args, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")

    run_python("print('hi')", timeout_sec=5.0, memory_limit_mb=256)

    assert "OPENAI_API_KEY" not in captured_env


def test_tool_call_delegates_to_run_python():
    tool = CodeExecutionTool(timeout_sec=5.0, memory_limit_mb=256)
    result = tool(CallContext.for_task(), code="print(1 + 1)")
    assert result.content == "2"
