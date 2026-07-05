"""Examples of loading common document formats with LangChain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint

from langchain_community.document_loaders import (
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader, UnstructuredCSVLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "sample_documents"
PDF_PATH = (
    ROOT / "data" / "sample_documents" / "lora_review_sample.pdf"
)


def preview(label: str, documents, chars: int = 300) -> None:
    print(f"\n{label}: {len(documents)} document(s)")
    if documents:
        print(documents[0].page_content[:chars].replace("\n", " "))
        print(f"metadata: {documents[0].metadata}")


def load_text_file():
    return TextLoader(str(SOURCE_DIR / "company_policies.txt")).load()


def load_pdf_file():
    return PyPDFLoader(str(PDF_PATH)).load()


def load_markdown_file():
    return UnstructuredMarkdownLoader(str(SOURCE_DIR / "markdown_sample.md")).load()


def load_json_file():
    json_path = SOURCE_DIR / "facebook_chat.json"
    print("\nRaw JSON sample:")
    pprint(json.loads(json_path.read_text())["messages"][:2])
    return JSONLoader(
        file_path=str(json_path),
        jq_schema=".messages[].content",
        text_content=False,
    ).load()


def load_csv_file():
    return CSVLoader(file_path=str(SOURCE_DIR / "mlb_teams_2012.csv")).load()


def load_csv_as_html_table():
    return UnstructuredCSVLoader(
        file_path=str(SOURCE_DIR / "mlb_teams_2012.csv"),
        mode="elements",
    ).load()


def load_docx_file():
    return Docx2txtLoader(str(SOURCE_DIR / "docx_sample.docx")).load()


def load_unstructured_files():
    files = [
        str(SOURCE_DIR / "markdown_sample.md"),
        str(SOURCE_DIR / "company_policies.txt"),
    ]
    return UnstructuredFileLoader(files).load()


def load_web_pages():
    return WebBaseLoader(
        [
            "https://www.ibm.com/topics/langchain",
            "https://www.redhat.com/en/topics/ai/what-is-instructlab",
        ]
    ).load()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-web",
        action="store_true",
        help="Also load live web pages. This requires network access.",
    )
    args = parser.parse_args()

    preview("TXT loader", load_text_file())
    preview("PDF loader", load_pdf_file())
    preview("Markdown loader", load_markdown_file())
    preview("JSON loader", load_json_file())
    preview("CSV loader", load_csv_file())
    preview("DOCX loader", load_docx_file())
    preview("Unstructured multi-file loader", load_unstructured_files())

    table_docs = load_csv_as_html_table()
    print("\nUnstructured CSV HTML table preview:")
    print(table_docs[0].metadata.get("text_as_html", "")[:500])

    if args.include_web:
        preview("Web loader", load_web_pages())


if __name__ == "__main__":
    main()
