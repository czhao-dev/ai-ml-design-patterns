"""Application settings loaded from environment variables, plus shared project paths."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TASKS_PATH = DATA_DIR / "tasks.jsonl"
KB_DIR = DATA_DIR / "knowledge_base"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REACT_METRICS_PATH = REPORTS_DIR / "react_metrics.json"
PLAN_EXECUTE_METRICS_PATH = REPORTS_DIR / "plan_execute_metrics.json"
REFLEXION_METRICS_PATH = REPORTS_DIR / "reflexion_metrics.json"
RESULTS_SUMMARY_PATH = REPORTS_DIR / "results_summary.md"

# $ per million tokens, (input, output). Confirm against current OpenAI pricing
# before relying on estimate_cost_usd() for a real budget decision.
MODEL_PRICING_PER_MTOK = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    model_id: str
    max_tokens: int
    temperature: float
    max_steps_react: int
    max_steps_per_subtask: int
    max_attempts_reflexion: int
    request_timeout_sec: float
    code_exec_timeout_sec: float
    code_exec_memory_limit_mb: int
    bm25_top_k: int


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if not value else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if not value else float(value)


def get_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Copy .env.example to .env and set your OpenAI API key."
        )

    return Settings(
        openai_api_key=api_key,
        model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1-mini"),
        max_tokens=_get_int("MAX_TOKENS", 1024),
        temperature=_get_float("TEMPERATURE", 0.0),
        max_steps_react=_get_int("MAX_STEPS_REACT", 8),
        max_steps_per_subtask=_get_int("MAX_STEPS_PER_SUBTASK", 4),
        max_attempts_reflexion=_get_int("MAX_ATTEMPTS_REFLEXION", 3),
        request_timeout_sec=_get_float("REQUEST_TIMEOUT_SEC", 60.0),
        code_exec_timeout_sec=_get_float("CODE_EXEC_TIMEOUT_SEC", 5.0),
        code_exec_memory_limit_mb=_get_int("CODE_EXEC_MEMORY_LIMIT_MB", 256),
        bm25_top_k=_get_int("BM25_TOP_K", 3),
    )
