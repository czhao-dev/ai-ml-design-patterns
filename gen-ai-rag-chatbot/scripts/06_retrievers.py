"""Retriever examples for RAG applications."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from uuid import uuid4

from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag_pipeline import build_embedding_model, build_llm
POLICY_FILE = (
    ROOT / "data" / "sample_documents" / "company_policies.txt"
)
PDF_PATH = (
    ROOT / "data" / "sample_documents" / "lora_review_sample.pdf"
)


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_chroma(documents, embedding_model, prefix: str = "retriever_demo"):
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=f"{prefix}_{uuid4().hex}",
    )


def print_docs(label: str, docs, limit: int = 220) -> None:
    print(f"\n{label}: {len(docs)} result(s)")
    for index, doc in enumerate(docs, start=1):
        print(f"{index}. {doc.page_content[:limit].replace(chr(10), ' ')}")


def vector_store_retriever_demo(chunks, embedding_model) -> None:
    vector_store = build_chroma(chunks, embedding_model, "policy_vector")

    similarity_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    print_docs("Similarity retriever", similarity_retriever.invoke("email policy"))

    mmr_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    print_docs("MMR retriever", mmr_retriever.invoke("email policy"))

    threshold_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4},
    )
    print_docs("Similarity threshold retriever", threshold_retriever.invoke("email policy"))


def multi_query_retriever_demo(embedding_model, llm) -> None:
    pdf_docs = PyPDFLoader(str(PDF_PATH)).load()
    chunks = split_documents(pdf_docs, chunk_size=500, chunk_overlap=20)
    vector_store = build_chroma(chunks, embedding_model, "pdf_multi_query")

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )
    print_docs(
        "Multi-query retriever",
        retriever.invoke("What does this paper say about efficient parameter tuning?"),
    )


def movie_documents() -> list[Document]:
    return [
        Document(
            page_content="Scientists bring back dinosaurs and mayhem breaks loose.",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="A thief enters dreams to plant an idea.",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="A detective gets lost in a series of dreams within dreams.",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="Four sisters grow up and choose different paths.",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and go on an adventure.",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone searching for meaning.",
            metadata={"year": 1979, "director": "Andrei Tarkovsky", "genre": "thriller", "rating": 9.9},
        ),
    ]


def self_query_retriever_demo(embedding_model, llm) -> None:
    metadata_fields = [
        AttributeInfo(
            name="genre",
            description="Movie genre, such as science fiction, comedy, drama, thriller, romance, action, or animated.",
            type="string",
        ),
        AttributeInfo(name="year", description="The year the movie was released", type="integer"),
        AttributeInfo(name="director", description="The movie director", type="string"),
        AttributeInfo(name="rating", description="A 1-10 movie rating", type="float"),
    ]
    movies = movie_documents()
    vector_store = build_chroma(movies, embedding_model, "movie_self_query")
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        "Brief summary of a movie.",
        metadata_fields,
    )
    print_docs(
        "Self-query retriever",
        retriever.invoke("I want to watch a movie directed by Christopher Nolan"),
    )


def parent_document_retriever_demo(chunks, embedding_model) -> None:
    child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator="\n")
    parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator="\n")
    vector_store = Chroma(
        collection_name=f"split_parents_{uuid4().hex}",
        embedding_function=embedding_model,
    )
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(chunks)

    print(f"\nParent documents stored: {len(list(store.yield_keys()))}")
    print_docs("Parent document retriever", retriever.invoke("smoking policy"))


def main() -> None:
    text_docs = TextLoader(str(POLICY_FILE)).load()
    chunks = split_documents(text_docs, chunk_size=200, chunk_overlap=20)
    print(f"Loaded {len(chunks)} policy chunks.")

    try:
        settings = get_settings()
        embedding_model = build_embedding_model(settings)
        llm = build_llm(settings)
    except RuntimeError as exc:
        print(f"Skipping retriever demos: {exc}")
        return

    vector_store_retriever_demo(chunks, embedding_model)
    multi_query_retriever_demo(embedding_model, llm)
    self_query_retriever_demo(embedding_model, llm)
    parent_document_retriever_demo(chunks, embedding_model)


if __name__ == "__main__":
    main()
