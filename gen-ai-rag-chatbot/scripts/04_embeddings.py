"""Generate query and document embeddings for local text samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag_pipeline import build_embedding_model
POLICY_FILE = (
    ROOT / "data" / "sample_documents" / "company_policies.txt"
)


def load_chunks(chunk_size: int = 100, chunk_overlap: int = 20) -> list[str]:
    document = TextLoader(str(POLICY_FILE)).load()[0]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(document.page_content)


def watsonx_embedding_demo(chunks: list[str]) -> None:
    settings = get_settings()
    embedding_model = build_embedding_model(settings)

    query_vector = embedding_model.embed_query("What is the email policy?")
    document_vectors = embedding_model.embed_documents(chunks[:5])

    print("\nWatsonx embeddings")
    print(f"query dimensions: {len(query_vector)}")
    print(f"document vectors: {len(document_vectors)}")
    print(f"first values: {query_vector[:5]}")


def huggingface_embedding_demo(chunks: list[str]) -> None:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_vector = embedding_model.embed_query("What is the email policy?")
    document_vectors = embedding_model.embed_documents(chunks[:5])

    print("\nHugging Face embeddings")
    print(f"query dimensions: {len(query_vector)}")
    print(f"document vectors: {len(document_vectors)}")
    print(f"first values: {query_vector[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-huggingface",
        action="store_true",
        help="Also run the Hugging Face embedding example. This may download a model.",
    )
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} text chunks from {POLICY_FILE.name}.")

    try:
        watsonx_embedding_demo(chunks)
    except RuntimeError as exc:
        print(f"\nSkipping Watsonx embedding call: {exc}")

    if args.include_huggingface:
        huggingface_embedding_demo(chunks)


if __name__ == "__main__":
    main()
