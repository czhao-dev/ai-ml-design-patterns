"""Show why whole-document prompting does not scale for long documents."""

from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.rag_pipeline import build_llm
POLICY_FILE = (
    ROOT / "data" / "sample_documents" / "company_policies.txt"
)

PROMPT = PromptTemplate.from_template(
    """Use only the document content below to answer the question.

Document:
{content}

Question:
{question}

If the document does not contain the answer, say so.
"""
)


def load_document_text() -> str:
    document = TextLoader(str(POLICY_FILE)).load()[0]
    return document.page_content


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def answer_from_full_document(question: str) -> str:
    settings = get_settings()
    llm = build_llm(settings)
    chain = PROMPT | llm
    return chain.invoke({"content": load_document_text(), "question": question})


def main() -> None:
    content = load_document_text()
    print(f"Loaded {len(content):,} characters, roughly {estimate_tokens(content):,} tokens.")
    print(
        "This small sample can fit in many model context windows, but larger PDFs need chunking, "
        "embedding, and retrieval so the prompt only contains relevant context."
    )

    try:
        answer = answer_from_full_document("What policy information does this document contain?")
    except RuntimeError as exc:
        print(f"\nSkipping live LLM call: {exc}")
        return

    print("\nLLM answer:")
    print(answer)


if __name__ == "__main__":
    main()
