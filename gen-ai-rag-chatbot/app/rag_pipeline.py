"""RAG pipeline for answering questions from uploaded documents."""

from functools import lru_cache
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from app.config import AppSettings, get_settings

DOCUMENT_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}


def build_llm(settings: AppSettings | None = None) -> ChatVertexAI:
    settings = settings or get_settings()
    return ChatVertexAI(
        model_name=settings.llm_model_id,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        max_output_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
    )


def build_embedding_model(settings: AppSettings | None = None) -> VertexAIEmbeddings:
    settings = settings or get_settings()
    return VertexAIEmbeddings(
        model_name=settings.embedding_model_id,
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )


def load_document(file_path: str | Path):
    suffix = Path(file_path).suffix.lower()
    loader_cls = DOCUMENT_LOADERS.get(suffix)
    if loader_cls is None:
        supported = ", ".join(sorted(DOCUMENT_LOADERS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types: {supported}.")
    loader = loader_cls(str(file_path))
    return loader.load()


def split_documents(documents, settings: AppSettings | None = None):
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def _build_retriever_cached(file_path: str, settings: AppSettings):
    documents = load_document(file_path)
    chunks = split_documents(documents, settings)
    vector_store = Chroma.from_documents(chunks, build_embedding_model(settings))
    return vector_store.as_retriever()


def build_retriever(file_path: str | Path, settings: AppSettings | None = None):
    settings = settings or get_settings()
    return _build_retriever_cached(str(file_path), settings)


def answer_question(file_path: str | Path, query: str, settings: AppSettings | None = None) -> str:
    if not file_path:
        return "Please upload a document."
    if not query or not query.strip():
        return "Please enter a question."

    settings = settings or get_settings()
    try:
        retriever = build_retriever(file_path, settings)
    except ValueError as exc:
        return str(exc)

    qa_chain = RetrievalQA.from_chain_type(
        llm=build_llm(settings),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    response = qa_chain.invoke({"query": query.strip()})
    return response["result"]
