"""Store document embeddings in Chroma and FAISS, then query them."""

from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag_pipeline import build_embedding_model
POLICY_FILE = (
    ROOT / "data" / "sample_documents" / "company_policies.txt"
)


def load_chunks():
    documents = TextLoader(str(POLICY_FILE)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    return splitter.split_documents(documents)


def print_matches(label: str, docs, limit: int = 220) -> None:
    print(f"\n{label}: {len(docs)} result(s)")
    for index, doc in enumerate(docs, start=1):
        print(f"{index}. {doc.page_content[:limit].replace(chr(10), ' ')}")


def chroma_demo(chunks, embedding_model) -> None:
    ids = [str(i) for i in range(len(chunks))]
    vector_store = Chroma.from_documents(
        chunks,
        embedding_model,
        ids=ids,
        collection_name=f"policy_demo_{uuid4().hex}",
    )

    print(f"\nChroma documents stored: {vector_store._collection.count()}")
    print_matches("Chroma similarity search", vector_store.similarity_search("Email policy", k=2))

    new_id = str(len(chunks))
    new_chunk = Document(
        page_content="InstructLab is an open-source tool for tuning large language models.",
        metadata={"source": "example", "page": 1},
    )
    vector_store.add_documents([new_chunk], ids=[new_id])
    print(f"Added document id {new_id}; count is now {vector_store._collection.count()}.")

    updated_chunk = Document(
        page_content="InstructLab is an open-source project for aligning and tuning large language models.",
        metadata={"source": "example", "page": 1},
    )
    vector_store.update_document(new_id, updated_chunk)
    print(f"Updated document id {new_id}.")

    vector_store._collection.delete(ids=[new_id])
    print(f"Deleted document id {new_id}; count is now {vector_store._collection.count()}.")


def faiss_demo(chunks, embedding_model) -> None:
    ids = [str(i) for i in range(len(chunks))]
    vector_store = FAISS.from_documents(chunks, embedding_model, ids=ids)
    print_matches("FAISS similarity search", vector_store.similarity_search("Smoking policy", k=2))


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {POLICY_FILE.name}.")

    try:
        settings = get_settings()
        embedding_model = build_embedding_model(settings)
    except RuntimeError as exc:
        print(f"Skipping vector store demos: {exc}")
        return

    chroma_demo(chunks, embedding_model)
    faiss_demo(chunks, embedding_model)


if __name__ == "__main__":
    main()
