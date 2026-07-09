"""Tests for app/config.py."""

import pytest

from app import config


def test_missing_project_id_raises_runtime_error():
    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
        config.get_settings()


def test_defaults_applied_when_env_vars_absent(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    settings = config.get_settings()

    assert settings.gcp_project_id == "test-project"
    assert settings.gcp_location == "us-central1"
    assert settings.llm_model_id == "gemini-2.5-flash"
    assert settings.embedding_model_id == "text-embedding-004"
    assert settings.max_new_tokens == 2048
    assert settings.temperature == 0.5
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 20
    assert settings.server_name == "0.0.0.0"
    assert settings.server_port == 7860


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("GCP_LOCATION", "europe-west1")
    monkeypatch.setenv("MAX_NEW_TOKENS", "512")
    monkeypatch.setenv("TEMPERATURE", "0.1")
    monkeypatch.setenv("CHUNK_SIZE", "500")

    settings = config.get_settings()

    assert settings.gcp_location == "europe-west1"
    assert settings.max_new_tokens == 512
    assert settings.temperature == 0.1
    assert settings.chunk_size == 500


def test_gradio_server_port_falls_back_to_port_env_var(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")

    settings = config.get_settings()

    assert settings.server_port == 9000


def test_port_env_var_takes_precedence_over_gradio_server_port(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("PORT", "8080")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")

    settings = config.get_settings()

    assert settings.server_port == 8080


def test_settings_are_frozen(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    settings = config.get_settings()

    with pytest.raises(AttributeError):
        settings.gcp_project_id = "changed"
