"""Tests for src/config.py."""

import pytest

from src import config

_ENV_VARS = [
    "OPENAI_MODEL_ID",
    "MAX_TOKENS",
    "TEMPERATURE",
    "MAX_STEPS_REACT",
    "MAX_STEPS_PER_SUBTASK",
    "MAX_ATTEMPTS_REFLEXION",
    "REQUEST_TIMEOUT_SEC",
    "CODE_EXEC_TIMEOUT_SEC",
    "CODE_EXEC_MEMORY_LIMIT_MB",
    "BM25_TOP_K",
]


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.get_settings()


def test_defaults_applied_when_env_vars_absent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    settings = config.get_settings()

    assert settings.openai_api_key == "test-key"
    assert settings.model_id == "gpt-4.1-mini"
    assert settings.max_tokens == 1024
    assert settings.temperature == 0.0
    assert settings.max_steps_react == 8
    assert settings.max_attempts_reflexion == 3


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_ID", "some-other-model")
    monkeypatch.setenv("MAX_TOKENS", "2048")
    monkeypatch.setenv("MAX_STEPS_REACT", "12")

    settings = config.get_settings()

    assert settings.model_id == "some-other-model"
    assert settings.max_tokens == 2048
    assert settings.max_steps_react == 12
