"""Tests for app/rag_pipeline.py.

Only pure/local logic is exercised here (document loading from disk,
chunking, and the input-validation guard clauses in `answer_question`).
Anything that talks to Vertex AI (`build_llm`, `build_embedding_model`, and
the retrieval/QA chain itself) needs live GCP credentials and is out of
scope for these offline unit tests.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.rag_pipeline import DOCUMENT_LOADERS, answer_question, load_document, split_documents


def test_document_loaders_cover_advertised_extensions():
    assert set(DOCUMENT_LOADERS) == {".pdf", ".txt", ".md", ".csv", ".docx"}


def test_load_document_reads_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello from a text file", encoding="utf-8")

    docs = load_document(path)

    assert len(docs) == 1
    assert "hello from a text file" in docs[0].page_content


def test_load_document_reads_markdown_via_text_loader(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nSome content.", encoding="utf-8")

    docs = load_document(path)

    assert "Some content." in docs[0].page_content


def test_load_document_reads_csv_rows(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\nfoo,1\nbar,2\n", encoding="utf-8")

    docs = load_document(path)

    assert len(docs) == 2
    assert "foo" in docs[0].page_content


def test_load_document_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"not really a zip")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_load_document_error_lists_supported_types(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"not really a zip")

    with pytest.raises(ValueError, match=r"\.csv.*\.docx.*\.md.*\.pdf.*\.txt"):
        load_document(path)


class _FixedSettings:
    """Stand-in for AppSettings, only exposing the fields split_documents reads."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


def test_split_documents_respects_chunk_size():
    long_text = "word " * 500
    settings = _FixedSettings(chunk_size=100, chunk_overlap=0)

    chunks = split_documents([Document(page_content=long_text)], settings)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_split_documents_single_short_document_stays_one_chunk():
    settings = _FixedSettings(chunk_size=1000, chunk_overlap=20)

    chunks = split_documents([Document(page_content="short document")], settings)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short document"


def test_answer_question_without_file_returns_prompt():
    assert answer_question(None, "What is this about?") == "Please upload a document."


def test_answer_question_without_query_returns_prompt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("content", encoding="utf-8")

    assert answer_question(path, "") == "Please enter a question."
    assert answer_question(path, "   ") == "Please enter a question."


def test_answer_question_unsupported_file_type_returns_error_message(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    path = tmp_path / "archive.zip"
    path.write_bytes(b"not really a zip")

    result = answer_question(path, "What is in this file?")

    assert "Unsupported file type" in result
