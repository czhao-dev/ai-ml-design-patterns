"""Sandboxed Python code execution tool.

Threat model (stated explicitly, do not over-claim): code originates only from
the OpenAI API responding to a fixed, self-authored benchmark task set run
locally by the project author -- not from an adversarial third party over a
network. The goal is to contain accidental/hallucinated destructive behavior
(stray filesystem calls, infinite loops, runaway memory) and keep secrets out
of the child process. This is not a defense against a determined attacker.
"""

from __future__ import annotations

import ast
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import CallContext, ToolResult

_ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "itertools",
    "functools",
    "re",
    "json",
    "collections",
    "datetime",
}
_DISALLOWED_NAMES = {"eval", "exec", "compile", "__import__", "open", "input"}
_DISALLOWED_ATTRS = {
    "__globals__",
    "__subclasses__",
    "__builtins__",
    "__class__",
    "__bases__",
    "__mro__",
    "__import__",
}


class CodeSafetyError(ValueError):
    """Raised when the static pre-check rejects code before any execution occurs."""


def _static_precheck(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeSafetyError(f"Code does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    raise CodeSafetyError(f"Import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                raise CodeSafetyError(f"Import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in _DISALLOWED_NAMES:
            raise CodeSafetyError(f"Use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in _DISALLOWED_ATTRS:
            raise CodeSafetyError(f"Access to '{node.attr}' is not allowed")


def _make_preexec_fn(memory_limit_mb: int, timeout_sec: float):
    """Returns a preexec_fn (POSIX only) that caps CPU time in the child process,
    as a second layer under the subprocess.run(timeout=) wall-clock cutoff.

    RLIMIT_AS (virtual address space) is intentionally only applied on Linux.
    On macOS, capping RLIMIT_AS at a modest value is unreliable -- the Python
    interpreter's own startup and dynamic-library loading can reserve more
    virtual address space than the cap allows, causing the child process to
    fail before the sandboxed code ever runs. RLIMIT_CPU is safe and portable
    across POSIX platforms and is applied everywhere except Windows.
    """
    if platform.system() == "Windows":
        return None

    def _limit() -> None:
        import resource

        if platform.system() == "Linux":
            mem_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        cpu_sec = int(timeout_sec) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))

    return _limit


def run_python(code: str, timeout_sec: float, memory_limit_mb: int) -> ToolResult:
    try:
        _static_precheck(code)
    except CodeSafetyError as exc:
        return ToolResult(content=f"Rejected before execution: {exc}", is_error=True)

    scratch_dir = tempfile.mkdtemp(prefix="agentic_bakeoff_codeexec_")
    tmp_file = Path(scratch_dir) / "snippet.py"
    tmp_file.write_text(code)

    # Minimal, explicit environment -- never the parent's full os.environ -- so
    # OPENAI_API_KEY (and anything else in the parent process's env) cannot leak
    # into generated code's environment even if it tried to read it.
    minimal_env = {"PATH": os.environ.get("PATH", "")}
    cmd = [sys.executable, "-I", "-S", str(tmp_file)]

    try:
        completed = subprocess.run(
            cmd,
            cwd=scratch_dir,
            env=minimal_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            preexec_fn=_make_preexec_fn(memory_limit_mb, timeout_sec),
        )
    except subprocess.TimeoutExpired:
        return ToolResult(content=f"Execution timed out after {timeout_sec}s", is_error=True)
    finally:
        try:
            tmp_file.unlink(missing_ok=True)
            Path(scratch_dir).rmdir()
        except OSError:
            pass  # best-effort cleanup; a leftover temp dir is harmless

    if completed.returncode == 0:
        return ToolResult(content=completed.stdout.strip())
    return ToolResult(content=completed.stderr.strip() or "Execution failed with no output", is_error=True)


class CodeExecutionTool:
    name = "execute_python"
    description = (
        "Execute a short Python snippet in a sandboxed subprocess and return its "
        "stdout. Only math, statistics, itertools, functools, re, json, collections, "
        "and datetime may be imported. Print the final answer as the last line of output."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute."}
        },
        "required": ["code"],
    }

    def __init__(self, timeout_sec: float = 5.0, memory_limit_mb: int = 256) -> None:
        self.timeout_sec = timeout_sec
        self.memory_limit_mb = memory_limit_mb

    def __call__(self, call_context: CallContext, code: str) -> ToolResult:
        return run_python(code, timeout_sec=self.timeout_sec, memory_limit_mb=self.memory_limit_mb)
