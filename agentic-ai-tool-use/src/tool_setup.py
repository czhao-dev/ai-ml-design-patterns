"""Builds the standard ToolRegistry shared by all three agent architectures,
and a context manager for temporarily installing a FlakyToolWrapper for a
single error_recovery-category task run."""

from __future__ import annotations

from contextlib import contextmanager

from .config import KB_DIR, Settings
from .tasks import ErrorInjection
from .tools.base import ToolRegistry
from .tools.calculator import CalculatorTool
from .tools.code_exec import CodeExecutionTool
from .tools.error_injection import FlakyToolWrapper
from .tools.retrieval import BM25Index, RetrievalTool, load_knowledge_base


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        CodeExecutionTool(
            timeout_sec=settings.code_exec_timeout_sec,
            memory_limit_mb=settings.code_exec_memory_limit_mb,
        )
    )
    documents = load_knowledge_base(KB_DIR)
    index = BM25Index(documents)
    registry.register(RetrievalTool(index, default_top_k=settings.bm25_top_k))
    return registry


@contextmanager
def apply_error_injection(registry: ToolRegistry, error_injection: ErrorInjection | None):
    """No-op if error_injection is None. Otherwise temporarily swaps the target
    tool for a FlakyToolWrapper around it for the duration of the `with` block,
    then restores the original tool -- so injection state never leaks into the
    next task run."""
    if error_injection is None:
        yield
        return

    original = registry.get(error_injection.target_tool)
    if original is None:
        yield
        return

    wrapper = FlakyToolWrapper(
        original,
        target_tool_name=error_injection.target_tool,
        fail_on_nth_call=error_injection.fail_on_nth_call,
        error_message=error_injection.error_message,
    )
    registry.replace(error_injection.target_tool, wrapper)
    try:
        yield
    finally:
        registry.replace(error_injection.target_tool, original)
