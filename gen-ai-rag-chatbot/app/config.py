"""Application settings loaded from environment variables."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    gcp_project_id: str
    gcp_location: str
    llm_model_id: str
    embedding_model_id: str
    max_new_tokens: int
    temperature: float
    chunk_size: int
    chunk_overlap: int
    server_name: str
    server_port: int


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if not value else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if not value else float(value)


def get_settings() -> AppSettings:
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "GCP_PROJECT_ID is required. Copy .env.example to .env and set your GCP project ID."
        )

    return AppSettings(
        gcp_project_id=project_id,
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        llm_model_id=os.getenv("VERTEX_LLM_MODEL_ID", "gemini-2.5-flash"),
        embedding_model_id=os.getenv("VERTEX_EMBEDDING_MODEL_ID", "text-embedding-004"),
        max_new_tokens=_get_int("MAX_NEW_TOKENS", 2048),
        temperature=_get_float("TEMPERATURE", 0.5),
        chunk_size=_get_int("CHUNK_SIZE", 1000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 20),
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=_get_int("PORT", _get_int("GRADIO_SERVER_PORT", 7860)),
    )
