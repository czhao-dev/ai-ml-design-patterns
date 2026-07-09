from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ENV_VARS = [
    "GCP_PROJECT_ID",
    "GCP_LOCATION",
    "VERTEX_LLM_MODEL_ID",
    "VERTEX_EMBEDDING_MODEL_ID",
    "MAX_NEW_TOKENS",
    "TEMPERATURE",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "GRADIO_SERVER_NAME",
    "GRADIO_SERVER_PORT",
    "PORT",
]

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate every test from the developer's real .env / shell environment."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
